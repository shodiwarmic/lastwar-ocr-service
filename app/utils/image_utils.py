"""
app/utils/image_utils.py

PIL/Pillow helper functions used across the pipeline.

Responsibilities:
- Converting Flask FileStorage objects to PIL Images
- Handling EXIF rotation so phone screenshots always arrive upright
- Serialising PIL Images to bytes for the Vision API
- Providing dimension helpers used by the stitcher

All functions are pure (no side effects, no I/O beyond the inputs) so they
are straightforward to unit test without mocking.
"""

import io
from typing import Optional

from PIL import Image, ImageOps, ExifTags
from werkzeug.datastructures import FileStorage

from app.utils.logger import get_logger

logger = get_logger(__name__)


def pil_from_file_storage(file_storage: FileStorage) -> Optional[Image.Image]:
    """
    Converts a Flask FileStorage object (multipart upload) to a PIL Image.

    Applies EXIF-based auto-rotation so that screenshots taken in portrait mode
    on any phone arrive correctly oriented regardless of the device's EXIF data.
    Most game screenshots will have no EXIF rotation, but applying ImageOps.exif_transpose
    unconditionally is safe and free.

    Args:
        file_storage: A file object from Flask's request.files.

    Returns:
        A PIL Image in RGB mode, or None if the file cannot be opened.

    Example:
        for f in request.files.getlist("images"):
            img = pil_from_file_storage(f)
    """
    try:
        img = Image.open(file_storage.stream)
        img = ImageOps.exif_transpose(img)  # Correct EXIF rotation
        img = img.convert("RGB")            # Normalise to RGB (handles RGBA PNGs)
        return img
    except Exception as exc:
        logger.error(
            "Failed to open image from FileStorage",
            extra={"image_filename": getattr(file_storage, "filename", "unknown"), "error": str(exc)},
        )
        return None


def pil_from_bytes(data: bytes) -> Optional[Image.Image]:
    """
    Opens a PIL Image from a raw bytes object.

    Used in tests to load fixture images without a Flask request context.

    Args:
        data: Raw image bytes (PNG, JPEG, etc.)

    Returns:
        A PIL Image in RGB mode, or None on failure.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return img
    except Exception as exc:
        logger.error("Failed to open image from bytes", extra={"error": str(exc)})
        return None


def pil_to_bytes(pil_image: Image.Image, fmt: str = "PNG") -> bytes:
    """
    Serialises a PIL Image to a bytes object for submission to the Vision API.

    PNG is the default because it is lossless and Vision API handles it well.
    JPEG can be used for stitched images to reduce payload size, but may
    introduce compression artefacts around small text characters.

    Args:
        pil_image: Any PIL Image object.
        fmt:       Output format string — "PNG" or "JPEG".

    Returns:
        Image data as bytes.
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format=fmt)
    return buffer.getvalue()


def get_image_dimensions(pil_image: Image.Image) -> tuple[int, int]:
    """
    Returns the (width, height) of a PIL Image.

    Used by the stitcher to group images by resolution and validate that
    images being concatenated share the same width.

    Args:
        pil_image: Any PIL Image object.

    Returns:
        (width, height) tuple of integers.
    """
    return pil_image.width, pil_image.height


def crop_top_bottom(
    pil_image: Image.Image,
    top_fraction: float = 0.0,
    bottom_fraction: float = 0.0,
) -> Image.Image:
    """
    Crops a fraction from the top and/or bottom of an image.

    Used by the stitcher to remove repeated UI chrome (nav tabs, header,
    self-player highlighted row) before concatenating screenshots.

    Fractions are relative to image height so results are resolution-agnostic.
    For example, top_fraction=0.18 removes the top 18% of the image.

    Args:
        pil_image:       Source image.
        top_fraction:    Proportion of image height to remove from the top (0.0–1.0).
        bottom_fraction: Proportion of image height to remove from the bottom (0.0–1.0).

    Returns:
        Cropped PIL Image. Returns original image unchanged if both fractions are 0.
    """
    if top_fraction == 0.0 and bottom_fraction == 0.0:
        return pil_image

    w, h = pil_image.size
    top_px = int(h * top_fraction)
    bottom_px = int(h * (1.0 - bottom_fraction))
    return pil_image.crop((0, top_px, w, bottom_px))


def sample_color_region(
    pil_image: Image.Image,
    x_fraction: float,
    y_fraction: float,
    sample_size: int = 10,
) -> tuple[int, int, int]:
    """
    Samples the average RGB colour of a small region at a relative position.

    Used by the classifier's pre-OCR pass to detect the orange active-tab
    highlight without parsing text. Relative coordinates make this
    resolution-independent.

    Args:
        pil_image:   Source image.
        x_fraction:  Horizontal centre of sample region (0.0 = left, 1.0 = right).
        y_fraction:  Vertical centre of sample region (0.0 = top, 1.0 = bottom).
        sample_size: Width and height of the sampled region in pixels.

    Returns:
        (R, G, B) tuple of the average colour in the sampled region.

    Example:
        r, g, b = sample_color_region(img, x_fraction=0.5, y_fraction=0.12)
        is_orange = r > 200 and g < 140 and b < 80
    """
    w, h = pil_image.size
    cx = int(w * x_fraction)
    cy = int(h * y_fraction)
    half = sample_size // 2

    left   = max(0, cx - half)
    upper  = max(0, cy - half)
    right  = min(w, cx + half)
    lower  = min(h, cy + half)

    region = pil_image.crop((left, upper, right, lower))
    pixels = list(region.getdata())

    if not pixels:
        return (0, 0, 0)

    r = sum(p[0] for p in pixels) // len(pixels)
    g = sum(p[1] for p in pixels) // len(pixels)
    b = sum(p[2] for p in pixels) // len(pixels)
    return (r, g, b)


def is_orange(rgb: tuple[int, int, int]) -> bool:
    """
    Returns True if an RGB colour falls within the orange range used by the
    Last War UI for active tabs and highlighted rows.

    Tuned against the sample screenshots. The orange tab colour is approximately
    (220–255, 100–160, 0–80) in RGB. Adjust thresholds if OCR fixture analysis
    reveals different values on other devices.

    Args:
        rgb: (R, G, B) tuple.

    Returns:
        True if the colour matches the active-tab orange.
    """
    r, g, b = rgb
    return r > 200 and 80 <= g <= 170 and b < 90
