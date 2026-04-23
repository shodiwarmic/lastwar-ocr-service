"""
app/pipeline/stitcher.py

Groups screenshots by resolution, stitches each group into one tall image
with black separator bands, and recursively splits any group that would
exceed Vision API size limits.

Design:
    The stitch-first pipeline skips pre-OCR classification.  Every uploaded
    image goes directly into a resolution group; images of the same size are
    stitched into one tall image and sent as a single Vision API call.
    Classification and player extraction happen per section after OCR,
    using only the text blocks whose bounding-box Y falls within each source
    image's slice of the stitched output.

Separator bands:
    A SEPARATOR_HEIGHT-pixel black band is pasted between each source image.
    This creates a clean visual boundary: no OCR word can have a bounding box
    that spans the separator, so per-section block filtering by Y range is clean.

API limit handling:
    Vision API rejects payloads > 20 MB (JPEG-encoded) or > 75 megapixels.
    _split_until_within_limits() recursively bisects oversized groups until
    every sub-batch fits, degrading gracefully to one call per image if needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from app.utils.image_utils import pil_to_bytes
from app.utils.logger import get_logger
from app.utils.window_detect import crop_to_window, detect_window_by_black_borders

logger = get_logger(__name__)

# Black separator band inserted between source images in the stitched output.
SEPARATOR_HEIGHT = 10  # pixels

# Vision API hard limits
_MAX_BYTES      = 20 * 1024 * 1024   # 20 MB encoded JPEG
_MAX_MEGAPIXELS = 75_000_000         # 75 MP uncompressed


@dataclass
class ImageRegion:
    """
    Y-coordinate slice of one source image within a stitched batch.

    After OCR, text blocks with avg_y in [y_start, y_end) belong to this
    source image and are classified and player-extracted independently.

    Attributes:
        filename: Original upload filename, used for logging and classification.
        y_start:  Top edge of this image's slice in the stitched output (px).
        y_end:    Bottom edge (exclusive) — equal to y_start + source image height.
    """
    filename: str
    y_start:  int
    y_end:    int


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def prepare_stitched_batches(
    images: list[tuple[Image.Image, str]],
) -> list[tuple[Image.Image, list[ImageRegion]]]:
    """
    Groups images by resolution, stitches each group, splits if oversized.

    This is the sole stitcher entry point called by the route handler.

    Args:
        images: (pil_image, filename) pairs in submission order.

    Returns:
        List of (stitched_image, [ImageRegion, ...]) tuples — one per Vision
        API call.  ImageRegion.y_start / y_end delimit each source image's
        pixel rows within the stitched output.
    """
    # Pre-step: crop each input image to its detected game window. This
    # neutralises the inside-landscape split-screen positioning issue
    # documented as B5 in the per-fixture verification report — the game
    # runs in a portrait sub-window inside the landscape canvas, and YAML
    # normalised fractions (x_hint, search_region, etc.) all assume the
    # game spans the full image. After cropping, downstream classifier
    # and extractor see the game window directly and the fractions land
    # in the right pixels regardless of where on the canvas Android put
    # the window.
    #
    # detect_window_by_black_borders returns None when the game already
    # fills the image (no letterboxing) — in that case the original
    # image passes through unchanged.
    cropped: list[tuple[Image.Image, str]] = []
    for img, filename in images:
        rect = detect_window_by_black_borders(img)
        if rect is None:
            cropped.append((img, filename))
            continue
        new_img = crop_to_window(img, rect)
        logger.info(
            "Cropped letterboxed image to detected game window",
            extra={
                "image_filename": filename,
                "from":           f"{img.width}x{img.height}",
                "to":             f"{new_img.width}x{new_img.height}",
                "window_rect":    rect.as_tuple(),
            },
        )
        cropped.append((new_img, filename))

    groups: dict[tuple[int, int], list[tuple[Image.Image, str]]] = {}
    for img, filename in cropped:
        key = (img.width, img.height)
        groups.setdefault(key, []).append((img, filename))

    for (w, h), items in groups.items():
        logger.info(
            "Resolution group formed",
            extra={
                "resolution":  f"{w}x{h}",
                "image_count": len(items),
                "filenames":   [f for _, f in items],
            },
        )

    batches: list[tuple[Image.Image, list[ImageRegion]]] = []
    for image_list in groups.values():
        batches.extend(_split_until_within_limits(image_list))

    logger.info(
        "Batch preparation complete",
        extra={
            "total_input_images":     len(images),
            "total_ocr_calls_needed": len(batches),
        },
    )

    return batches


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _stitch_with_separators(
    image_list: list[tuple[Image.Image, str]],
) -> tuple[Image.Image, list[ImageRegion]]:
    """
    Concatenates images vertically with SEPARATOR_HEIGHT-pixel black bands.

    Returns the stitched PIL image and the ImageRegion list that records
    each source image's Y slice within it.
    """
    width = image_list[0][0].width
    n     = len(image_list)

    total_h  = sum(img.height for img, _ in image_list) + SEPARATOR_HEIGHT * (n - 1)
    stitched = Image.new("RGB", (width, total_h), color=(0, 0, 0))

    regions:  list[ImageRegion] = []
    y_offset = 0

    for i, (img, filename) in enumerate(image_list):
        y_start = y_offset
        stitched.paste(img, (0, y_offset))
        y_offset += img.height
        regions.append(ImageRegion(filename=filename, y_start=y_start, y_end=y_offset))
        if i < n - 1:
            y_offset += SEPARATOR_HEIGHT   # advance past separator band

    logger.debug(
        "Stitched images with separators",
        extra={
            "image_count":   n,
            "stitched_size": f"{width}x{total_h}",
            "filenames":     [f for _, f in image_list],
        },
    )

    return stitched, regions


def _within_api_limits(image: Image.Image) -> bool:
    """Returns True if the image fits within Vision API pixel and byte limits."""
    if image.width * image.height > _MAX_MEGAPIXELS:
        return False
    jpeg_bytes = pil_to_bytes(image, fmt="JPEG")
    return len(jpeg_bytes) <= _MAX_BYTES


def _split_until_within_limits(
    image_list: list[tuple[Image.Image, str]],
) -> list[tuple[Image.Image, list[ImageRegion]]]:
    """
    Recursively bisects image_list until every sub-batch fits API limits.

    Base cases:
        - Single image: returned as-is (cannot split further).
        - Stitched batch within limits: returned directly.
    Recursive case:
        Bisect at midpoint; recurse on both halves independently.
    """
    if not image_list:
        return []

    stitched, regions = _stitch_with_separators(image_list)

    if _within_api_limits(stitched) or len(image_list) == 1:
        return [(stitched, regions)]

    mid   = len(image_list) // 2
    left  = _split_until_within_limits(image_list[:mid])
    right = _split_until_within_limits(image_list[mid:])

    logger.info(
        "Split stitched batch to stay within API limits",
        extra={
            "original_count": len(image_list),
            "split_into":     len(left) + len(right),
        },
    )

    return left + right
