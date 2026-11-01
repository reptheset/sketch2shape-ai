"""
Stage 6, primary provider: Microsoft TRELLIS on a free Hugging Face Space (ZeroGPU).

TRELLIS (MIT licence) turns one object image into a textured mesh (PBR material with a
baked colour texture) and is the best open image-to-3D model we can use for free. It needs
a large NVIDIA GPU, so instead of running it locally we call the public Space
"trellis-community/TRELLIS" through gradio_client, exactly like the Space's web UI does:

  1. /start_session             creates the session's temporary folder on the Space
  2. /preprocess_image          the Space's own crop/background clean-up of our image
  3. /generate_and_extract_glb  sparse structure + structured latent sampling, then GLB export

ZeroGPU quotas (GPU time per day, reset every 24h): no token ~2 min, free account ~5 min,
PRO ~40 min. One generation uses roughly 30-60 s of GPU time, so set HF_TOKEN (a free "Read"
token) to get several models per day. When the quota runs out, the queue is full or the
Space is down, this raises ProviderError and the caller falls back to local TripoSR.
After a quota error the Space is skipped for a cool-down period (TRELLIS_QUOTA_COOLDOWN_MINUTES),
so later jobs go straight to the fallback instead of waiting for the same error again.
"""

import inspect
import logging
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import settings
from app.errors import PipelineError

logger = logging.getLogger("sketch2shape.trellis")

StatusCallback = Callable[[str], None]  # human-readable message, called from the worker thread

MAX_SEED = 2**31 - 1
QUOTA_HINTS = ("quota", "zerogpu", "gpu limit", "rate limit", "too many requests", "exceeded")
POLL_SECONDS = 1.0

_cooldown_until = 0.0
_cooldown_lock = threading.Lock()


class ProviderError(PipelineError):
    """A reconstruction provider could not produce a model. `quota` marks quota/rate-limit errors."""

    def __init__(self, message: str, quota: bool = False) -> None:
        super().__init__(message)
        self.quota = quota


def cooldown_remaining() -> float:
    """Seconds until the Space is tried again after a quota error (0 = try it)."""
    return max(0.0, _cooldown_until - time.time())


def _start_cooldown() -> None:
    global _cooldown_until
    with _cooldown_lock:
        _cooldown_until = time.time() + settings.trellis_quota_cooldown_seconds


def _is_quota_error(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in QUOTA_HINTS)


def _make_client() -> Any:
    try:
        from gradio_client import Client
    except ImportError as exc:
        raise ProviderError("gradio_client is not installed. Run: pip install -r requirements.txt") from exc

    kwargs: dict[str, Any] = {"verbose": False}
    if settings.hf_token:
        # The keyword was renamed from hf_token to token in newer gradio_client versions.
        params = inspect.signature(Client.__init__).parameters
        kwargs["hf_token" if "hf_token" in params else "token"] = settings.hf_token
    try:
        return Client(settings.trellis_space, **kwargs)
    except Exception as exc:
        raise ProviderError(f"Could not connect to the Space {settings.trellis_space}: {str(exc)[:200]}") from exc


def _find_files(value: Any, found: list[str]) -> None:
    """Collects every file path in a Gradio result (strings, dicts with path/value/video, tuples)."""
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key in ("video", "value", "path", "name"):
            if key in value:
                _find_files(value[key], found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _find_files(item, found)


def _status_message(update: Any) -> str | None:
    code = getattr(getattr(update, "code", None), "name", str(getattr(update, "code", "")))
    if code in {"IN_QUEUE", "JOINING_QUEUE"}:
        rank, size = getattr(update, "rank", None), getattr(update, "queue_size", None)
        if rank is not None:
            position = f"position {rank + 1}" + (f" of {size}" if size else "")
            return f"Waiting for a free GPU on Hugging Face ({position})"
        return "Waiting for a free GPU on Hugging Face"
    if code in {"PROCESSING", "ITERATING", "PROGRESS", "SENDING_DATA", "LOG"}:
        return "Building the 3D model with TRELLIS (free Hugging Face GPU)"
    if code == "QUEUE_FULL":
        return "The free GPU queue is full"
    return None


def run_trellis(image_path: Path, output_glb: Path, seed: int, on_status: StatusCallback) -> dict[str, Any]:
    """Blocking call (run it in a worker thread). Writes output_glb; returns details for the job."""
    try:
        from gradio_client import handle_file
    except ImportError as exc:
        raise ProviderError("gradio_client is not installed. Run: pip install -r requirements.txt") from exc

    started = time.perf_counter()
    on_status("Connecting to the free TRELLIS GPU on Hugging Face")
    client = _make_client()

    try:
        try:
            client.predict(api_name="/start_session")
        except Exception as exc:  # older/newer versions of the Space may not need it
            logger.info("TRELLIS /start_session skipped: %s", str(exc)[:120])

        on_status("Uploading the object image to TRELLIS")
        preprocessed = client.predict(image=handle_file(str(image_path)), api_name="/preprocess_image")
        files: list[str] = []
        _find_files(preprocessed, files)
        if not files:
            raise ProviderError("TRELLIS did not return a preprocessed image.")

        job = client.submit(
            image=handle_file(files[0]),
            multiimages=[],
            seed=int(seed) % MAX_SEED,
            ss_guidance_strength=settings.trellis_ss_guidance,
            ss_sampling_steps=settings.trellis_ss_steps,
            slat_guidance_strength=settings.trellis_slat_guidance,
            slat_sampling_steps=settings.trellis_slat_steps,
            multiimage_algo="stochastic",
            mesh_simplify=settings.trellis_mesh_simplify,
            texture_size=settings.trellis_texture_size,
            api_name="/generate_and_extract_glb",
        )

        deadline = time.perf_counter() + settings.trellis_timeout_seconds
        last_message = None
        while not job.done():
            if time.perf_counter() > deadline:
                try:
                    job.cancel()
                except Exception:
                    pass
                raise ProviderError(f"TRELLIS took longer than {settings.trellis_timeout_seconds:.0f}s.")
            message = _status_message(job.status())
            if message == "The free GPU queue is full":
                raise ProviderError("The free TRELLIS GPU queue is full.", quota=True)
            if message and message != last_message:
                on_status(message)
                last_message = message
            time.sleep(POLL_SECONDS)

        result = job.result()
    except ProviderError as exc:
        if exc.quota:
            _start_cooldown()
        raise
    except Exception as exc:
        text = str(exc) or exc.__class__.__name__
        quota = _is_quota_error(text)
        if quota:
            _start_cooldown()
        logger.warning("TRELLIS failed: %s", text[:300])
        raise ProviderError(f"TRELLIS failed: {text[:200]}", quota=quota) from exc

    files = []
    _find_files(result, files)
    glb = next((f for f in files if f.lower().endswith(".glb") and Path(f).is_file()), None)
    if glb is None:
        raise ProviderError("TRELLIS finished but returned no GLB file.")
    shutil.copyfile(glb, output_glb)

    preview = None
    video = next((f for f in files if f.lower().endswith(".mp4") and Path(f).is_file()), None)
    if video:
        preview_path = output_glb.with_name("trellis_preview.mp4")
        shutil.copyfile(video, preview_path)
        preview = preview_path.name

    return {
        "space": settings.trellis_space,
        "seed": int(seed) % MAX_SEED,
        "used_token": bool(settings.hf_token),
        "ss_sampling_steps": settings.trellis_ss_steps,
        "slat_sampling_steps": settings.trellis_slat_steps,
        "mesh_simplify": settings.trellis_mesh_simplify,
        "texture_size": settings.trellis_texture_size,
        "preview_video": preview,
        "seconds": round(time.perf_counter() - started, 2),
    }
