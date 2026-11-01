"""
Stage 6, fallback provider: TripoSR (Stability AI + Tripo AI, MIT licence), run locally.

TripoSR reconstructs a mesh from one image in a single forward pass (no diffusion), so it
runs on a laptop: a few seconds for the model plus 10-60 s for mesh extraction on Apple
Silicon or CPU. Quality is lower than TRELLIS and the colour is stored per vertex instead
of in a texture image, but it always works, offline and without any quota.

Setup (once):  python -m scripts.setup_triposr
  - clones github.com/VAST-AI-Research/TripoSR into backend/vendor/TripoSR (the `tsr` package)
  - downloads the stabilityai/TripoSR weights (~1.7 GB) into ~/.cache/huggingface

TripoSR normally needs `torchmcubes`, a CUDA/C++ extension that is awkward to compile on a
Mac. We register a tiny drop-in module with the same `marching_cubes(volume, level)` function
implemented with scikit-image, so nothing has to be compiled.

Device: cuda > mps (Apple GPU) > cpu with TRIPOSR_DEVICE=auto. If an operation is not
supported on mps, the run is retried once on the CPU.
"""

import inspect
import logging
import re
import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings
from app.errors import PipelineError
from app.services.reconstruction_trellis import ProviderError

logger = logging.getLogger("sketch2shape.triposr")

StatusCallback = Callable[[str], None]

FOREGROUND_RATIO = 0.85  # how much of the frame the object fills (TripoSR's own default)
BACKGROUND_GREY = 0.5  # TripoSR was trained on objects composited on 50% grey
DENSITY_THRESHOLD = 25.0  # TripoSR's default iso-level for mesh extraction


# ---------------------------------------------------------------- torchmcubes shim


def marching_cubes_numpy(volume: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes with the vertex order of torchmcubes (x = last array axis).

    scikit-image returns vertices as (axis0, axis1, axis2); torchmcubes returns (axis2,
    axis1, axis0), and TripoSR reorders them back with [..., [2, 1, 0]]. So we reverse here.
    """
    from skimage.measure import marching_cubes

    lo, hi = float(volume.min()), float(volume.max())
    if not lo < level < hi:
        raise ProviderError("TripoSR found no surface in the image (the object may be too thin or too small).")
    verts, faces, _normals, _values = marching_cubes(volume, level=level)
    verts = np.ascontiguousarray(verts[:, ::-1], dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    return verts, faces


def install_torchmcubes_shim() -> None:
    if "torchmcubes" in sys.modules:
        return
    try:
        import torchmcubes  # noqa: F401  (the real one is used if it happens to be installed)

        return
    except ImportError:
        pass

    def marching_cubes(volume: Any, level: float) -> tuple[Any, Any]:
        import torch

        verts, faces = marching_cubes_numpy(volume.detach().float().cpu().numpy(), float(level))
        return torch.from_numpy(verts), torch.from_numpy(faces)

    shim = types.ModuleType("torchmcubes")
    shim.marching_cubes = marching_cubes  # type: ignore[attr-defined]
    sys.modules["torchmcubes"] = shim


# ---------------------------------------------------------------- model loading


class _TripoState:
    def __init__(self) -> None:
        self.model: Any = None
        self.device: str | None = None
        self.load_lock = threading.Lock()
        self.run_lock = threading.Lock()  # one local reconstruction at a time


state = _TripoState()


def is_installed() -> bool:
    return (settings.triposr_dir / "tsr" / "system.py").is_file()


def _pick_device() -> str:
    import torch

    wanted = settings.triposr_device
    if wanted != "auto":
        return wanted
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# TripoSR's checkpoint was saved with transformers 4.x, whose ViT used different parameter
# names. transformers 5 renamed them (it converts old names itself in from_pretrained, but
# TripoSR loads its checkpoint with plain load_state_dict), so we apply the same renames here.
_VIT_PREFIX = "image_tokenizer.model."
_VIT_RENAMES = (
    (re.compile(r"^encoder\.layer\."), "layers."),
    (re.compile(r"attention\.attention\.query"), "attention.q_proj"),
    (re.compile(r"attention\.attention\.key"), "attention.k_proj"),
    (re.compile(r"attention\.attention\.value"), "attention.v_proj"),
    (re.compile(r"attention\.output\.dense"), "attention.o_proj"),
    (re.compile(r"intermediate\.dense"), "mlp.fc1"),
    (re.compile(r"(?<!attention\.)output\.dense"), "mlp.fc2"),
)


def upgrade_vit_keys(state_dict: dict[str, Any], expected: set[str]) -> dict[str, Any]:
    """Renames old-style ViT keys only when the model expects the new names."""
    if all(key in expected for key in state_dict):
        return state_dict
    upgraded = {}
    for key, value in state_dict.items():
        if key.startswith(_VIT_PREFIX) and key not in expected:
            rest = key[len(_VIT_PREFIX):]
            for pattern, replacement in _VIT_RENAMES:
                rest = pattern.sub(replacement, rest)
            key = _VIT_PREFIX + rest
        upgraded[key] = value
    return upgraded


def _load_tsr(tsr_class: Any) -> Any:
    """Same as TSR.from_pretrained, but tolerant of the transformers 5 ViT renames."""
    import torch
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    config_path = hf_hub_download(repo_id=settings.triposr_model, filename="config.yaml")
    weight_path = hf_hub_download(repo_id=settings.triposr_model, filename="model.ckpt")
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    model = tsr_class(cfg)
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    state_dict = upgrade_vit_keys(state_dict, set(model.state_dict().keys()))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ProviderError(
            f"TripoSR weights do not match this transformers version "
            f"(missing {missing[:3]}, unexpected {unexpected[:3]})."
        )
    return model


def load_model(move_to_device: bool = True) -> Any:
    """Loads TripoSR once (downloads ~1.7 GB of weights on first use)."""
    if state.model is not None:
        return state.model
    with state.load_lock:
        if state.model is not None:
            return state.model
        if not is_installed():
            raise ProviderError(
                "TripoSR is not set up. Run once from backend/: python -m scripts.setup_triposr"
            )
        if str(settings.triposr_dir) not in sys.path:
            sys.path.insert(0, str(settings.triposr_dir))
        install_torchmcubes_shim()
        try:
            from tsr.system import TSR
        except ImportError as exc:
            raise ProviderError(
                f"TripoSR could not be imported ({exc}). Run: pip install -r requirements.txt"
            ) from exc

        logger.info("Loading TripoSR (%s)...", settings.triposr_model)
        started = time.perf_counter()
        model = _load_tsr(TSR)
        model.renderer.set_chunk_size(settings.triposr_chunk_size)
        device = "cpu"
        if move_to_device:
            device = _pick_device()
            model.to(device)
        state.model, state.device = model, device
        logger.info("TripoSR ready on %s in %.1fs", device, time.perf_counter() - started)
        return model


# ---------------------------------------------------------------- mesh helpers


def prepare_image(rgba_path: Path) -> Image.Image:
    """RGBA cut-out -> object filling 85% of the frame, composited on 50% grey (TripoSR's input format)."""
    from tsr.utils import resize_foreground

    image = resize_foreground(Image.open(rgba_path).convert("RGBA"), FOREGROUND_RATIO)
    array = np.asarray(image).astype(np.float32) / 255.0
    rgb = array[:, :, :3] * array[:, :, 3:4] + (1.0 - array[:, :, 3:4]) * BACKGROUND_GREY
    return Image.fromarray((rgb * 255.0).round().astype(np.uint8))


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def orient_for_gltf(mesh: Any) -> Any:
    """TripoSR's z-up output -> glTF y-up, facing the viewer (same as TripoSR's demo app),
    with faces wound outward so the viewer does not render the inside of the object."""
    import trimesh

    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    if signed_volume(np.asarray(mesh.vertices), np.asarray(mesh.faces)) < 0:
        mesh.invert()
    return mesh


# ---------------------------------------------------------------- main entry point


def _reconstruct(model: Any, device: str, image: Image.Image, on_status: StatusCallback) -> Any:
    import torch

    on_status(f"Reconstructing the 3D shape locally with TripoSR ({device})")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    on_status(f"Extracting the mesh (resolution {settings.triposr_mc_resolution})")
    kwargs: dict[str, Any] = {"resolution": settings.triposr_mc_resolution, "threshold": DENSITY_THRESHOLD}
    if "has_vertex_color" in inspect.signature(model.extract_mesh).parameters:
        kwargs["has_vertex_color"] = True
    with torch.no_grad():
        meshes = model.extract_mesh(scene_codes, **kwargs)
    return meshes[0]


def run_triposr(image_path: Path, output_glb: Path, on_status: StatusCallback) -> dict[str, Any]:
    """Blocking call (run it in a worker thread). Writes output_glb; returns details for the job."""
    started = time.perf_counter()
    on_status("Loading the local TripoSR model")
    with state.run_lock:
        model = load_model()
        image = prepare_image(image_path)
        device = state.device or "cpu"
        try:
            mesh = _reconstruct(model, device, image, on_status)
        except ProviderError:
            raise
        except (RuntimeError, NotImplementedError) as exc:
            if device != "mps":
                raise
            # Some operations are missing on the Apple GPU in older PyTorch versions.
            logger.warning("TripoSR failed on mps (%s); retrying on CPU", str(exc)[:200])
            model.to("cpu")
            state.device = device = "cpu"
            mesh = _reconstruct(model, device, image, on_status)

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ProviderError("TripoSR produced an empty mesh.")
        on_status("Saving the 3D model")
        orient_for_gltf(mesh)
        mesh.export(str(output_glb), file_type="glb")

    return {
        "model": settings.triposr_model,
        "device": device,
        "mc_resolution": settings.triposr_mc_resolution,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "texture": "vertex colours",
        "seconds": round(time.perf_counter() - started, 2),
    }
