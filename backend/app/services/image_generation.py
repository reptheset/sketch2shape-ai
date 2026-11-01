"""
Stage 4: sketch-guided image generation with SDXL + ControlNet Scribble, run locally.

Input:  the cleaned sketch from Stage 2 + the prompt, negative prompt and ControlNet
        scale chosen in Stage 3.
Output: generated.png, a photorealistic product-style image of the sketched object on a
        plain white background, ready for background removal (Stage 5).

Everything here is open-source and free, and runs on your own machine:
  - Base model:  stabilityai/stable-diffusion-xl-base-1.0   (CreativeML Open RAIL++-M)
  - ControlNet:  xinsir/controlnet-scribble-sdxl-1.0         (Apache-2.0)
  - VAE:         madebyollin/sdxl-vae-fp16-fix               (MIT; avoids black/NaN images in fp16)
  - Library:     Hugging Face diffusers + PyTorch

Device is picked automatically: NVIDIA GPU (cuda) > Apple Silicon GPU (mps) > CPU.
On a Mac with 24 GB+ memory a 1024x1024 image with 30 steps takes roughly 1-2 minutes.

The pipeline is loaded once (about 10 GB of weights, downloaded on first use and cached in
~/.cache/huggingface) and reused for every job. Only one image is generated at a time,
because the GPU can only hold one SDXL run comfortably.

The scribble ControlNet expects white lines on a black background, so the black-on-white
sketch from Stage 2 is inverted before it is passed in.
"""

import asyncio
import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.errors import PipelineError

logger = logging.getLogger("sketch2shape.generation")

ProgressCallback = Callable[[int, int], None]  # (step, total_steps), called from the worker thread


@dataclass
class GenerationResult:
    path: Path
    base_model: str
    controlnet_model: str
    device: str
    controlnet_scale: float
    seed: int
    steps: int
    guidance_scale: float
    width: int
    height: int
    inference_seconds: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = self.path.name
        return data


# ---------------------------------------------------------------- model loading


class _ModelState:
    """Holds the single shared pipeline and its loading status (for /api/health)."""

    def __init__(self) -> None:
        self.pipe: Any = None
        self.device: str | None = None
        self.status = "not_loaded"  # not_loaded | loading | ready | error
        self.error: str | None = None
        self.load_lock = threading.Lock()  # one loader at a time
        self.run_lock = asyncio.Lock()  # one generation at a time


state = _ModelState()


def _pick_device() -> str:
    import torch

    wanted = settings.sd_device
    if wanted != "auto":
        return wanted
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(move_to_device: bool = True) -> Any:
    """Loads SDXL + ControlNet once (thread-safe). Downloads weights on first use."""
    if state.pipe is not None:
        return state.pipe
    with state.load_lock:
        if state.pipe is not None:
            return state.pipe
        state.status, state.error = "loading", None
        started = time.perf_counter()
        try:
            import torch
            from diffusers import (
                AutoencoderKL,
                ControlNetModel,
                EulerAncestralDiscreteScheduler,
                StableDiffusionXLControlNetPipeline,
            )
        except ImportError as exc:
            state.status, state.error = "error", "PyTorch/diffusers are not installed"
            raise PipelineError(
                "PyTorch and diffusers are not installed. Run: pip install -r requirements.txt"
            ) from exc

        try:
            device = _pick_device()
            # fp16 halves memory and is fast on GPUs; CPUs need fp32.
            dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
            logger.info("Loading SDXL + ControlNet Scribble (%s, %s)...", device, dtype)

            controlnet = ControlNetModel.from_pretrained(settings.sd_controlnet_model, torch_dtype=dtype)
            vae = AutoencoderKL.from_pretrained(settings.sd_vae_model, torch_dtype=dtype)
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained(settings.sd_base_model, subfolder="scheduler")
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                settings.sd_base_model,
                controlnet=controlnet,
                vae=vae,
                scheduler=scheduler,
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,  # download only the fp16 weights
                use_safetensors=True,
            )
            if move_to_device:
                pipe = pipe.to(device)
                if device == "mps" and settings.sd_attention_slicing:
                    # Recommended by diffusers for Apple Silicon with < 64 GB memory.
                    pipe.enable_attention_slicing()
            pipe.set_progress_bar_config(disable=True)
        except Exception as exc:
            state.status, state.error = "error", str(exc)[:300]
            logger.exception("Could not load the image generation models")
            raise PipelineError(f"Could not load the image generation models: {str(exc)[:200]}") from exc

        state.pipe, state.device, state.status = pipe, device, "ready"
        logger.info("Image generation models ready on %s in %.0fs", device, time.perf_counter() - started)
        return pipe


def model_status() -> dict[str, Any]:
    return {"status": state.status, "device": state.device, "error": state.error}


# ---------------------------------------------------------------- generation


def prepare_control_image(sketch_path: Path, size: int) -> Image.Image:
    """Black-on-white sketch -> white-on-black, strictly 0/255, size x size (what the scribble ControlNet expects)."""
    gray = cv2.imread(str(sketch_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise PipelineError("The cleaned sketch could not be read for image generation.")
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    lines = np.where(gray < 128, 255, 0).astype(np.uint8)
    return Image.fromarray(lines).convert("RGB")


def _run_sync(
    control_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    scale: float,
    seed: int,
    on_progress: ProgressCallback | None,
) -> Image.Image:
    pipe = load_pipeline()  # raises a clear PipelineError if PyTorch/diffusers are missing
    import torch

    steps = settings.sd_steps

    def step_end(_pipe: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        if on_progress is not None:
            on_progress(step + 1, steps)
        return callback_kwargs

    # A CPU generator gives reproducible results for a given seed on every device.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        controlnet_conditioning_scale=scale,
        num_inference_steps=steps,
        guidance_scale=settings.sd_guidance_scale,
        width=control_image.width,
        height=control_image.height,
        generator=generator,
        callback_on_step_end=step_end,
    )
    return result.images[0]


def _looks_blank(image: Image.Image) -> bool:
    """fp16 problems show up as all-black (or uniform) images."""
    array = np.asarray(image.convert("L"), dtype=np.float32)
    return float(array.std()) < 2.0


async def generate_image(
    sketch_path: Path,
    output_path: Path,
    prompt: str,
    negative_prompt: str,
    controlnet_scale: float,
    on_progress: ProgressCallback | None = None,
) -> GenerationResult:
    """Runs Stage 4 and writes the generated image to `output_path`."""
    size = settings.image_size
    control_image = prepare_control_image(sketch_path, size)
    (output_path.parent / "control.png").parent.mkdir(parents=True, exist_ok=True)
    control_image.save(output_path.parent / "control.png")  # kept for debugging / the README

    seed = random.randint(0, 2**31 - 1)  # recorded so a good result can be reproduced
    async with state.run_lock:  # one SDXL run at a time; other jobs wait here
        started = time.perf_counter()
        try:
            image = await asyncio.to_thread(
                _run_sync, control_image, prompt, negative_prompt, controlnet_scale, seed, on_progress
            )
        except PipelineError:
            raise
        except RuntimeError as exc:
            message = str(exc)
            logger.exception("SDXL generation failed")
            if "out of memory" in message.lower() or "mps backend out of memory" in message.lower():
                raise PipelineError(
                    "The computer ran out of GPU memory while generating the image. "
                    "Close other apps, or set IMAGE_SIZE=768 in backend/.env."
                ) from exc
            raise PipelineError(f"Image generation failed: {message[:200]}") from exc
        elapsed = time.perf_counter() - started

    if _looks_blank(image):
        raise PipelineError("The generated image came out blank. Please try again (a new random seed is used).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return GenerationResult(
        path=output_path,
        base_model=settings.sd_base_model,
        controlnet_model=settings.sd_controlnet_model,
        device=state.device or "unknown",
        controlnet_scale=controlnet_scale,
        seed=seed,
        steps=settings.sd_steps,
        guidance_scale=settings.sd_guidance_scale,
        width=image.width,
        height=image.height,
        inference_seconds=round(elapsed, 1),
    )
