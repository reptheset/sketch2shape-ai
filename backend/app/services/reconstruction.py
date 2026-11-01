"""
Stage 6: single-image 3D reconstruction, with automatic fallback.

Input:  foreground.png from Stage 5 (RGBA object cut-out, centred, square).
Output: model.glb, a textured 3D asset.

Providers are tried in the order given by RECON_PROVIDERS (default "trellis,triposr"):
  trellis  TRELLIS on the free Hugging Face Space (best quality, PBR texture; daily GPU quota)
  triposr  TripoSR locally on this computer (lower quality, vertex colours; always available)

If a provider fails for any reason (quota used up, queue full, Space asleep or down,
timeout, missing setup), its error is recorded and the next provider is tried. The job only
fails if every provider fails. details.reconstruction in the job status shows which provider
produced the model and why any earlier one was skipped.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings
from app.errors import PipelineError
from app.services import reconstruction_trellis as trellis
from app.services import reconstruction_triposr as triposr

logger = logging.getLogger("sketch2shape.reconstruction")

StatusCallback = Callable[[str], None]

PROVIDER_LABELS = {"trellis": "TRELLIS (Hugging Face Space)", "triposr": "TripoSR (local)"}


@dataclass
class ReconstructionResult:
    glb_path: Path
    provider: str
    provider_label: str
    file_size_bytes: int
    seconds: float
    preview_video: str | None
    provider_details: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["glb_path"] = self.glb_path.name
        return data


def check_glb(path: Path) -> int:
    """Light check that a provider wrote a real binary glTF; full mesh validation is Stage 7."""
    if not path.is_file():
        raise PipelineError("No GLB file was written.")
    size = path.stat().st_size
    with path.open("rb") as handle:
        magic = handle.read(4)
    if size < 100 or magic != b"glTF":
        raise PipelineError("The returned 3D file is not a valid GLB.")
    return size


def provider_status() -> dict[str, Any]:
    """For /api/health: which providers are configured and ready."""
    return {
        "order": settings.recon_providers,
        "trellis": {
            "space": settings.trellis_space,
            "hf_token_set": bool(settings.hf_token),
            "quota_cooldown_seconds": round(trellis.cooldown_remaining()),
        },
        "triposr": {
            "installed": triposr.is_installed(),
            "loaded": triposr.state.model is not None,
            "device": triposr.state.device,
        },
    }


def _run_provider(name: str, image_path: Path, output_glb: Path, seed: int, on_status: StatusCallback) -> dict:
    if name == "trellis":
        return trellis.run_trellis(image_path, output_glb, seed, on_status)
    if name == "triposr":
        return triposr.run_triposr(image_path, output_glb, on_status)
    raise PipelineError(f"Unknown reconstruction provider '{name}'")


async def reconstruct_3d(
    image_path: Path, output_glb: Path, seed: int, on_status: StatusCallback
) -> ReconstructionResult:
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []

    for index, name in enumerate(settings.recon_providers):
        label = PROVIDER_LABELS[name]
        if name == "trellis" and trellis.cooldown_remaining() > 0:
            minutes = trellis.cooldown_remaining() / 60
            attempts.append({"provider": name, "ok": False, "skipped": True,
                             "error": f"GPU quota recently used up; retrying the Space in {minutes:.0f} min"})
            logger.info("Skipping TRELLIS (quota cool-down, %.0f min left)", minutes)
            continue

        output_glb.unlink(missing_ok=True)
        t = time.perf_counter()
        try:
            details = await asyncio.to_thread(_run_provider, name, image_path, output_glb, seed, on_status)
            size = check_glb(output_glb)
        except Exception as exc:
            if isinstance(exc, PipelineError):
                reason = str(exc)
            else:
                logger.exception("%s crashed", label)
                reason = f"{exc.__class__.__name__}: {str(exc)[:200]}"
            attempts.append({"provider": name, "ok": False, "seconds": round(time.perf_counter() - t, 2),
                             "quota": bool(getattr(exc, "quota", False)), "error": reason})
            logger.warning("%s failed: %s", label, reason)
            remaining = settings.recon_providers[index + 1:]
            if remaining:
                on_status(f"{label} unavailable, switching to {PROVIDER_LABELS[remaining[0]]}")
            continue

        seconds = round(time.perf_counter() - t, 2)
        attempts.append({"provider": name, "ok": True, "seconds": seconds})
        return ReconstructionResult(
            glb_path=output_glb,
            provider=name,
            provider_label=label,
            file_size_bytes=size,
            seconds=round(time.perf_counter() - started, 2),
            preview_video=details.get("preview_video"),
            provider_details=details,
            attempts=attempts,
        )

    reasons = "; ".join(f"{PROVIDER_LABELS[a['provider']]}: {a['error']}" for a in attempts)
    raise PipelineError(f"3D reconstruction failed. {reasons}")
