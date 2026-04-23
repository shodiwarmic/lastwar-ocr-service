"""
app/utils/window_detect.py

Detect the actual game-window rectangle inside a screenshot.

Background — the YAMLs in lastwar-screen-definitions describe layouts in
**normalised fractions of the source image** (e.g. `tabs.x_hint = 0.15` is
"15% across the image"). That assumption holds when the game fills the
screen edge-to-edge — true for the Pixel 10 Pro XL baseline and the Pixel
Fold front/inside-portrait orientations. It breaks on the Pixel Fold's
inside-landscape orientation, where Android renders the game in a
portrait-shaped split-screen window inside a landscape canvas. The window
can sit at the left, centre, or right of the canvas; the rest is black
bars or system chrome ("Double-tap to move this app").

This module finds that window so the rest of the pipeline can crop to it
before applying any normalised-fraction logic.

Two detection strategies — call ``detect_game_window`` for the composite:

  1. Black-border scan (preferred, runs pre-OCR). Sample columns/rows from
     each edge inward; the first non-near-black column is the edge of the
     window.
  2. OCR bounding-box union (post-OCR fallback). After OCR has run, take
     the min/max x/y across all detected text bboxes and pad slightly. Use
     this when the borders aren't uniformly dark (e.g. system chrome panel
     in split-screen mode is dark grey, not black).

Both functions return ``(left, top, right, bottom)`` in pixel coordinates,
or ``None`` if the strategy was inconclusive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from PIL import Image

# A pixel is "near black" if every channel is at or below this. Tuned wide
# enough to catch the slightly-grey bars Android renders around split-screen
# windows; tight enough to reject anything resembling actual UI chrome.
NEAR_BLACK_MAX_CHANNEL = 30

# A column/row counts as a "border" when at least this fraction of its
# sampled pixels are near-black. Allows a few stray noisy pixels.
BORDER_COVERAGE_THRESHOLD = 0.95

# Sanity floor — the detected window must be at least this fraction of the
# original dimension to be considered legitimate. Prevents an over-aggressive
# crop on a frame that genuinely has wide dark UI elements (e.g. a fully
# black background screen during transitions).
MIN_WINDOW_FRACTION = 0.20

# How many pixels to sample per column/row when scanning for borders. More =
# slower but more reliable. 64 is enough to detect the borders without being
# fooled by sparse non-black pixels in nominally-dark UI.
SAMPLE_COUNT = 64

# OCR-bbox fallback padding — expand the union-of-bboxes by this fraction
# of the original dimension to capture UI chrome that has no text (e.g.
# the back-arrow button at the bottom).
BBOX_PADDING_FRACTION = 0.03


@dataclass(frozen=True)
class WindowRect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


# ---------------------------------------------------------------------------
# Strategy 1 — Black-border scan
# ---------------------------------------------------------------------------

def _column_is_border(image: Image.Image, x: int, samples: int) -> bool:
    """True if at least BORDER_COVERAGE_THRESHOLD of `samples` pixels in
    column `x` are near-black."""
    h = image.height
    if h <= 0:
        return False
    near_black = 0
    for i in range(samples):
        y = (i * (h - 1)) // max(samples - 1, 1)
        r, g, b = image.getpixel((x, y))[:3]
        if r <= NEAR_BLACK_MAX_CHANNEL and g <= NEAR_BLACK_MAX_CHANNEL and b <= NEAR_BLACK_MAX_CHANNEL:
            near_black += 1
    return near_black >= int(samples * BORDER_COVERAGE_THRESHOLD)


def _row_is_border(image: Image.Image, y: int, samples: int) -> bool:
    w = image.width
    if w <= 0:
        return False
    near_black = 0
    for i in range(samples):
        x = (i * (w - 1)) // max(samples - 1, 1)
        r, g, b = image.getpixel((x, y))[:3]
        if r <= NEAR_BLACK_MAX_CHANNEL and g <= NEAR_BLACK_MAX_CHANNEL and b <= NEAR_BLACK_MAX_CHANNEL:
            near_black += 1
    return near_black >= int(samples * BORDER_COVERAGE_THRESHOLD)


def detect_window_by_black_borders(
    image: Image.Image,
    samples: int = SAMPLE_COUNT,
) -> Optional[WindowRect]:
    """Find the largest non-border rectangle inside ``image`` by scanning
    each edge inward until a non-border column/row is found.

    Returns None when the detected window is implausibly small (smaller
    than MIN_WINDOW_FRACTION of either dimension), which usually means the
    image isn't letterboxed and the caller should treat the full image as
    the window.

    Always converts to RGB internally so transparency / palette images work.
    """
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    w, h = rgb.size
    if w == 0 or h == 0:
        return None

    # Scan from left
    left = 0
    while left < w and _column_is_border(rgb, left, samples):
        left += 1
    # Scan from right
    right = w - 1
    while right > left and _column_is_border(rgb, right, samples):
        right -= 1
    # Scan from top
    top = 0
    while top < h and _row_is_border(rgb, top, samples):
        top += 1
    # Scan from bottom
    bottom = h - 1
    while bottom > top and _row_is_border(rgb, bottom, samples):
        bottom -= 1

    # Convert to half-open [left, right) and [top, bottom)
    rect = WindowRect(left=left, top=top, right=right + 1, bottom=bottom + 1)

    # Sanity check — reject implausibly-narrow detections. A window that
    # is smaller than 20% of either dimension probably means the borders
    # logic was confused (e.g. the screenshot was a near-black loading frame).
    if rect.width < int(w * MIN_WINDOW_FRACTION) or rect.height < int(h * MIN_WINDOW_FRACTION):
        return None

    # If the detection didn't actually shrink anything, return None so the
    # caller knows borders weren't found and can skip the crop.
    if rect.width == w and rect.height == h:
        return None

    return rect


# ---------------------------------------------------------------------------
# Strategy 2 — OCR bounding-box union (post-OCR fallback)
# ---------------------------------------------------------------------------

def detect_window_by_ocr_bboxes(
    text_blocks: Sequence[dict],
    image_size: tuple[int, int],
    padding_fraction: float = BBOX_PADDING_FRACTION,
) -> Optional[WindowRect]:
    """Take the union of every text-block bounding box, pad slightly, clamp
    to the image. Use this when ``detect_window_by_black_borders`` returned
    None but the image is still letterboxed by non-black chrome (e.g. the
    "Double-tap to move this app" panel in split-screen mode).

    Returns None when there are no text blocks.
    """
    if not text_blocks:
        return None

    img_w, img_h = image_size
    min_x, min_y = img_w, img_h
    max_x = max_y = 0
    found_any = False
    for block in text_blocks:
        bbox = block.get("bbox")
        if not bbox:
            continue
        # bbox is either {"vertices": [{"x":..,"y":..}, ...]} (dict form
        # produced by capture_ocr_fixture.py) or a Vision API proto with
        # .vertices. Handle the dict form here — proto-form callers should
        # convert before calling this.
        verts = bbox.get("vertices") if isinstance(bbox, dict) else getattr(bbox, "vertices", None)
        if not verts:
            continue
        for v in verts:
            x = v.get("x") if isinstance(v, dict) else v.x
            y = v.get("y") if isinstance(v, dict) else v.y
            if x is None or y is None:
                continue
            min_x = min(min_x, int(x))
            min_y = min(min_y, int(y))
            max_x = max(max_x, int(x))
            max_y = max(max_y, int(y))
            found_any = True

    if not found_any:
        return None

    pad_x = int(img_w * padding_fraction)
    pad_y = int(img_h * padding_fraction)
    left   = max(0,            min_x - pad_x)
    top    = max(0,            min_y - pad_y)
    right  = min(img_w,        max_x + pad_x)
    bottom = min(img_h,        max_y + pad_y)

    # Sanity: must still cover at least MIN_WINDOW_FRACTION of each dim.
    if right - left < int(img_w * MIN_WINDOW_FRACTION):
        return None
    if bottom - top < int(img_h * MIN_WINDOW_FRACTION):
        return None
    if right - left == img_w and bottom - top == img_h:
        return None

    return WindowRect(left=left, top=top, right=right, bottom=bottom)


# ---------------------------------------------------------------------------
# Composite — call this; pick the best available strategy
# ---------------------------------------------------------------------------

def detect_game_window(
    image: Image.Image,
    text_blocks: Optional[Sequence[dict]] = None,
) -> Optional[WindowRect]:
    """Returns the game window rectangle inside ``image``, or ``None`` when
    the full image already is the window (no letterbox detected).

    When ``text_blocks`` is supplied, falls back to the OCR-bbox strategy
    if the black-border scan was inconclusive.
    """
    rect = detect_window_by_black_borders(image)
    if rect is not None:
        return rect
    if text_blocks is not None:
        return detect_window_by_ocr_bboxes(text_blocks, image.size)
    return None


def crop_to_window(image: Image.Image, rect: WindowRect) -> Image.Image:
    """Return a new image cropped to ``rect``. Raises ``ValueError`` if the
    rect is outside the image bounds."""
    if rect.left < 0 or rect.top < 0 or rect.right > image.width or rect.bottom > image.height:
        raise ValueError(
            f"Window rect {rect.as_tuple()} exceeds image size {image.size}"
        )
    return image.crop(rect.as_tuple())
