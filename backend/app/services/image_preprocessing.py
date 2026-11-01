"""
Sketch input preparation with OpenCV.

Turns any accepted input (a clean whiteboard export, a scanned drawing, or a phone photo
of paper with uneven lighting) into the same thing: a square, centred, black-on-white
PNG with the paper shading and specks removed. Every later stage (GPT-4o-mini analysis,
ControlNet Scribble) gets a consistent, well-framed sketch.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MIN_SIDE = 64
MAX_SIDE = 8000
WORKING_MAX_SIDE = 2048  # big phone photos are downscaled first; keeps processing fast


class InvalidImageError(ValueError):
    """The upload is not a usable sketch. The message is safe to show to the user."""


@dataclass
class PreprocessResult:
    width: int
    height: int
    original_width: int
    original_height: int
    ink_ratio: float  # share of pixels that are ink in the final image
    stroke_components: int  # number of separate connected strokes kept
    inverted: bool  # True if the input was light lines on a dark background


def decode_image(data: bytes) -> np.ndarray:
    """Decodes uploaded bytes and checks the image is a sensible size."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise InvalidImageError("The file could not be read as an image. Please upload a valid PNG or JPG.")
    height, width = image.shape[:2]
    if min(height, width) < MIN_SIDE:
        raise InvalidImageError(f"The image is too small ({width}×{height}). Use at least {MIN_SIDE}×{MIN_SIDE} pixels.")
    if max(height, width) > MAX_SIDE:
        raise InvalidImageError(f"The image is too large ({width}×{height}). Keep both sides under {MAX_SIDE} pixels.")
    return image


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts to 8-bit grayscale, putting transparent areas on a white background."""
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _limit_size(gray: np.ndarray, max_side: int) -> np.ndarray:
    height, width = gray.shape
    longest = max(height, width)
    if longest <= max_side:
        return gray
    scale = max_side / longest
    return cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def _remove_specks(ink: np.ndarray) -> tuple[np.ndarray, int]:
    """Drops tiny connected blobs (dust, paper grain, JPEG noise). `ink` is 255 where there is ink."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(8, int(ink.size * 0.00002))
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area  # label 0 is the background
    return np.where(keep[labels], 255, 0).astype(np.uint8), int(keep.sum())


def preprocess_sketch(src: Path, dst: Path, size: int = 1024) -> PreprocessResult:
    """Cleans the sketch at `src` and writes a size×size black-on-white PNG to `dst`."""
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise InvalidImageError("The uploaded sketch could not be opened.")
    original_height, original_width = image.shape[:2]

    gray = _limit_size(_to_grayscale(image), WORKING_MAX_SIDE)

    # Sketches are expected to be dark lines on light paper. A mostly dark image
    # (chalkboard, white-on-black drawing) is inverted first.
    inverted = float(np.median(gray)) < 110
    if inverted:
        gray = 255 - gray

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive thresholding compares each pixel with its neighbourhood, so shadows and
    # uneven lighting in phone photos do not turn into black patches.
    block = max(15, (max(gray.shape) // 40) | 1)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)
    ink, components = _remove_specks(255 - binary)

    points = cv2.findNonZero(ink)
    if points is None or components == 0:
        raise InvalidImageError("No sketch lines were found. Make sure the drawing is clearly visible.")

    # Crop to the drawing, pad it into a square and centre it.
    x, y, w, h = cv2.boundingRect(points)
    side = int(max(w, h) * 1.3) + 2
    canvas = np.zeros((side, side), dtype=np.uint8)
    off_x, off_y = (side - w) // 2, (side - h) // 2
    canvas[off_y : off_y + h, off_x : off_x + w] = ink[y : y + h, x : x + w]

    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    final_ink = np.where(resized > 40, 255, 0).astype(np.uint8)

    # Very thin strokes almost disappear after downscaling; thicken them slightly so
    # ControlNet Scribble still sees clear outlines.
    if np.count_nonzero(final_ink) / final_ink.size < 0.02:
        final_ink = cv2.dilate(final_ink, np.ones((3, 3), np.uint8), iterations=1)

    output = 255 - final_ink  # black lines on white
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), output):
        raise OSError(f"Could not write {dst}")

    return PreprocessResult(
        width=size,
        height=size,
        original_width=original_width,
        original_height=original_height,
        ink_ratio=round(float(np.count_nonzero(final_ink)) / final_ink.size, 4),
        stroke_components=components,
        inverted=inverted,
    )
