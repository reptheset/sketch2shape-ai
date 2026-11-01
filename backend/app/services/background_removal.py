"""
Stage 5: background removal with rembg (U²-Net), run locally.

Input:  generated.png from Stage 4 (object on a light studio background).
Output: foreground.png        RGBA, transparent background, object centred, square
        foreground_white.png  the same object composited on pure white (some 3D models
                              expect an RGB image)

Steps:
  1. rembg + U²-Net predicts an alpha mask for the main object.          (MIT licence)
  2. Mask clean-up with OpenCV:
       - small, separate bits are dropped (e.g. steam wisps, sparkles, dust) by keeping
         the largest connected region plus any region at least 10% of its size
       - small pin-holes inside the object are filled, but real openings (a mug handle,
         chair-back gaps) are kept because they touch the background
  3. Crop to the object, pad to a square with a margin and resize. Image-to-3D models
     expect the object centred and filling most of the frame.
  4. Sanity checks: fail clearly if nothing (or the whole image) was detected.

The U²-Net model (~170 MB) is downloaded by rembg on first use and cached (in ~/.u2net
by default). The session is created once and reused.
"""

import asyncio
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.errors import PipelineError

logger = logging.getLogger("sketch2shape.background")

ALPHA_THRESHOLD = 127  # mask value above which a pixel counts as "object"
KEEP_RATIO = 0.10  # keep extra regions at least this fraction of the main region
MAX_HOLE_FRACTION = 0.002  # fill enclosed holes smaller than this fraction of the image


@dataclass
class ForegroundResult:
    rgba_path: Path
    white_path: Path
    model: str
    object_coverage: float  # fraction of the source image covered by the object
    regions_found: int
    regions_kept: int
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rgba_path"] = self.rgba_path.name
        data["white_path"] = self.white_path.name
        return data


# ---------------------------------------------------------------- rembg session


_session: Any = None
_session_lock = threading.Lock()


def get_session() -> Any:
    """Creates the rembg session once (downloads the model on first use)."""
    global _session
    if _session is not None:
        return _session
    with _session_lock:
        if _session is None:
            try:
                from rembg import new_session
            except ImportError as exc:
                raise PipelineError("rembg is not installed. Run: pip install -r requirements.txt") from exc
            logger.info("Loading rembg model '%s'...", settings.rembg_model)
            try:
                _session = new_session(settings.rembg_model)
            except Exception as exc:
                logger.exception("Could not load the background removal model")
                raise PipelineError(f"Could not load the background removal model: {str(exc)[:200]}") from exc
    return _session


# ---------------------------------------------------------------- mask clean-up


def clean_mask(alpha: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Keeps the main object (plus large secondary parts) and fills tiny pin-holes.

    Returns (binary mask 0/255, regions found, regions kept).
    """
    binary = np.where(alpha > ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary, 0, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    keep = np.zeros(count, dtype=bool)
    keep[1:] = areas >= largest * KEEP_RATIO
    mask = np.where(keep[labels], 255, 0).astype(np.uint8)

    # Fill small enclosed holes (noise inside the object). Holes connected to the image
    # border are background, and large holes are real openings, so both are kept.
    inverse = 255 - mask
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
    height, width = mask.shape
    max_hole = MAX_HOLE_FRACTION * height * width
    for label in range(1, hole_count):
        x, y, w, h, area = hole_stats[label]
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        if not touches_border and area <= max_hole:
            mask[hole_labels == label] = 255

    return mask, int(count - 1), int(keep.sum())


def frame_object(rgba: np.ndarray, size: int, margin: float) -> np.ndarray:
    """Crops to the object, pads to a centred square with `margin` on each side, resizes."""
    alpha = rgba[:, :, 3]
    points = cv2.findNonZero(alpha)
    x, y, w, h = cv2.boundingRect(points)
    side = int(max(w, h) / (1.0 - 2.0 * margin))
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    off_x, off_y = (side - w) // 2, (side - h) // 2
    canvas[off_y : off_y + h, off_x : off_x + w] = rgba[y : y + h, x : x + w]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def composite_on_white(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    return rgb.astype(np.uint8)


# ---------------------------------------------------------------- main entry point


def _remove_sync(image_path: Path, rgba_path: Path, white_path: Path) -> ForegroundResult:
    from rembg import remove

    started = time.perf_counter()
    session = get_session()
    source = Image.open(image_path).convert("RGB")
    cutout = remove(source, session=session, post_process_mask=True)
    rgba = np.array(cutout.convert("RGBA"))

    mask, found, kept = clean_mask(rgba[:, :, 3])
    coverage = float(np.count_nonzero(mask)) / mask.size
    if found == 0 or coverage < 0.01:
        raise PipelineError(
            "Could not separate the object from the background. Try a clearer sketch or a more specific description."
        )
    if coverage > 0.95:
        raise PipelineError("The background could not be removed (the whole image was detected as the object).")

    # Use rembg's soft alpha inside the cleaned mask, so edges stay smooth.
    rgba[:, :, 3] = np.minimum(rgba[:, :, 3], mask)
    rgba[:, :, 3][(mask == 255) & (rgba[:, :, 3] < ALPHA_THRESHOLD)] = 255  # filled pin-holes become opaque
    rgba[:, :, :3][rgba[:, :, 3] == 0] = 0  # clean colour under fully transparent pixels

    framed = frame_object(rgba, settings.foreground_size, settings.foreground_margin)
    Image.fromarray(framed).save(rgba_path)
    Image.fromarray(composite_on_white(framed)).save(white_path)

    return ForegroundResult(
        rgba_path=rgba_path,
        white_path=white_path,
        model=settings.rembg_model,
        object_coverage=round(coverage, 4),
        regions_found=found,
        regions_kept=kept,
        seconds=round(time.perf_counter() - started, 2),
    )


async def remove_background(image_path: Path, rgba_path: Path, white_path: Path) -> ForegroundResult:
    """Runs Stage 5 in a worker thread (CPU-bound)."""
    try:
        return await asyncio.to_thread(_remove_sync, image_path, rgba_path, white_path)
    except PipelineError:
        raise
    except Exception as exc:
        logger.exception("Background removal failed")
        raise PipelineError(f"Background removal failed: {str(exc)[:200]}") from exc
