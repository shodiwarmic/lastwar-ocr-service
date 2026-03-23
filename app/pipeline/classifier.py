"""
app/pipeline/classifier.py

Two-pass screenshot classification for Last War: Survival ranking screens.

Pass 1 — Pre-OCR (fast path):
    Uses colour sampling of the tab area to detect the active UI tab without
    calling the Vision API. Returns a confidence score. If confidence is below
    CONFIDENCE_THRESHOLD the image is routed to Pass 2.

Pass 2 — OCR-assisted (fallback):
    The image is sent individually (unstitched) to the Vision API. The returned
    text blocks are parsed to identify UI labels that unambiguously identify the
    screen type.

    For Daily Rank screens, Pass 2 uses the OCR bounding boxes to locate each
    day tab precisely in the image, then samples the background colour of that
    crop using Pillow. The active tab has an orange background; inactive tabs
    are grey. This is more reliable than text-pattern analysis because it reads
    a physical pixel signal directly from the image rather than inferring from
    OCR text characteristics.

Classification priority order (prevents mis-routing):
    1. Strength Ranking  — unique header + different tab set, caught first
    2. Weekly Rank       — "Weekly Rank" tab is orange, no day tabs active
    3. Daily Rank day    — bounding-box colour sampling of each day tab region
    4. Daily Rank text   — scoring fallback when image not available (tests)

Coordinates used in blind colour sampling (Pass 1) are expressed as fractions
of image dimensions so results are consistent across all device resolutions.
"""

from __future__ import annotations

import colorsys
from typing import Optional

from PIL import Image

from app.utils.image_utils import sample_color_region, is_orange
from app.utils.text_utils import normalize_day_label
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.80

TAB_BAR_Y_FRACTION   = 0.20
HEADER_Y_FRACTION    = 0.10
WEEKLY_TAB_X_FRACTION = 0.65

DAY_TAB_X_POSITIONS = {
    "monday":    0.10,
    "tuesday":   0.26,
    "wednesday": 0.42,
    "thursday":  0.58,
    "friday":    0.74,
    "saturday":  0.90,
}

STRENGTH_POWER_TAB_X = 0.15

# Pixels to expand around a day-tab bounding box when sampling background
# colour. The text bbox is tight around the characters; expanding ensures
# we sample the tab background rather than the text glyphs themselves.
TAB_BBOX_PADDING = 8

# HSV thresholds for the orange active-tab background.
# H is normalised 0.0–1.0 (0.0=red, 0.167=orange, 0.333=yellow).
# Orange range corresponds to roughly 15–40 degrees on the 360° wheel.
ORANGE_H_MIN = 0.040   # ~15°
ORANGE_H_MAX = 0.115   # ~41°
ORANGE_S_MIN = 0.55    # must be saturated (not grey)
ORANGE_V_MIN = 0.65    # must be bright (not dark)

# Known text markers used in Pass 2 OCR-based classification
_STRENGTH_MARKERS = {"strength ranking", "power", "kills", "donation"}
_DAY_ABBREVIATIONS = {"mon.", "tues.", "wed.", "thur.", "fri.", "sat."}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_screenshot(
    image: Image.Image,
    filename: str = "",
) -> tuple[Optional[str], float]:
    """
    Classifies a screenshot using Pass 1 colour sampling only.

    This is the fast-path classifier called before any Vision API interaction.
    It returns a (category, confidence) tuple. If confidence is below
    CONFIDENCE_THRESHOLD the caller should invoke classify_from_ocr_text()
    after obtaining OCR results.

    Args:
        image:    PIL Image of the screenshot to classify.
        filename: Original filename, used only for logging.

    Returns:
        Tuple of (category_string_or_None, confidence_float).
        category is one of: "monday"–"saturday", "weekly", "power", or None.
        confidence is 0.0–1.0.
    """
    result, confidence = _detect_strength_ranking(image)
    if result:
        logger.debug("Pass 1: Strength Ranking detected",
                     extra={"image_filename": filename, "confidence": confidence})
        return "power", confidence

    result, confidence = _detect_weekly_rank(image)
    if result:
        logger.debug("Pass 1: Weekly Rank detected",
                     extra={"image_filename": filename, "confidence": confidence})
        return "weekly", confidence

    day, confidence = _detect_active_day(image)
    if day:
        logger.debug("Pass 1: Daily Rank detected",
                     extra={"image_filename": filename, "day": day, "confidence": confidence})
        return day, confidence

    logger.debug("Pass 1: Classification ambiguous",
                 extra={"image_filename": filename, "confidence": 0.0})
    return None, 0.0


def classify_from_ocr_text(
    text_blocks: list[dict],
    image: Optional[Image.Image] = None,
    filename: str = "",
) -> tuple[Optional[str], float]:
    """
    Classifies a screenshot using text blocks returned by the Vision API.

    This is the Pass 2 fallback called when Pass 1 confidence is below
    CONFIDENCE_THRESHOLD. When a PIL image is also provided, day-tab
    identification uses bounding-box colour sampling for high accuracy.
    When no image is provided (e.g. in unit tests using fixture text blocks
    only), a text-scoring fallback is used instead.

    Args:
        text_blocks: List of OCR text block dicts sorted top-to-bottom.
        image:       Optional PIL Image. When present, enables precise
                     colour-based day tab detection using OCR bounding boxes.
        filename:    Original filename for logging.

    Returns:
        Tuple of (category_string_or_None, confidence_float).
        Confidence is 1.0 on a definitive match, 0.8 on a colour-based match,
        or 0.0 on failure.
    """
    all_text_lower = {block["text"].strip().lower() for block in text_blocks}

    # 1. Strength Ranking — unique header text
    if _ocr_detect_strength(all_text_lower):
        logger.debug("Pass 2: Strength Ranking detected via OCR",
                     extra={"image_filename": filename})
        return "power", 1.0

    # 2. Weekly Rank — weekly token without day tab abbreviations
    if _ocr_detect_weekly(all_text_lower):
        logger.debug("Pass 2: Weekly Rank detected via OCR",
                     extra={"image_filename": filename})
        return "weekly", 1.0

    # 3. Daily Rank — colour sampling using OCR bounding boxes (preferred)
    if image is not None:
        day = _detect_active_day_by_color(image, text_blocks)
        if day:
            logger.debug("Pass 2: Daily Rank detected via bounding-box colour sampling",
                         extra={"image_filename": filename, "day": day})
            return day, 0.95

    # 4. Daily Rank — text scoring fallback (used in tests without image)
    day = _ocr_detect_active_day_by_text(text_blocks)
    if day:
        logger.debug("Pass 2: Daily Rank detected via text scoring fallback",
                     extra={"image_filename": filename, "day": day})
        return day, 0.75

    logger.warning(
        "Pass 2: OCR classification failed — no definitive markers found",
        extra={"image_filename": filename, "detected_tokens": list(all_text_lower)[:20]},
    )
    return None, 0.0


# ---------------------------------------------------------------------------
# Pass 1 — blind colour-sampling helpers
# ---------------------------------------------------------------------------

def _detect_strength_ranking(image: Image.Image) -> tuple[bool, float]:
    """Samples the Power tab position unique to the Strength Ranking screen."""
    rgb = sample_color_region(image, x_fraction=STRENGTH_POWER_TAB_X,
                              y_fraction=TAB_BAR_Y_FRACTION)
    if is_orange(rgb):
        return True, 0.90
    return False, 0.0


def _detect_weekly_rank(image: Image.Image) -> tuple[bool, float]:
    """Samples the right-side Weekly Rank tab position for orange."""
    rgb = sample_color_region(image, x_fraction=WEEKLY_TAB_X_FRACTION,
                              y_fraction=TAB_BAR_Y_FRACTION - 0.05)
    if is_orange(rgb):
        return True, 0.88
    return False, 0.0


def _detect_active_day(image: Image.Image) -> tuple[Optional[str], float]:
    """
    Samples each day tab's expected horizontal position for orange background.
    Used in Pass 1 before OCR text blocks are available.
    """
    for day, x_frac in DAY_TAB_X_POSITIONS.items():
        rgb = sample_color_region(image, x_fraction=x_frac,
                                  y_fraction=TAB_BAR_Y_FRACTION)
        if is_orange(rgb):
            return day, 0.85
    return None, 0.0


# ---------------------------------------------------------------------------
# Pass 2 — colour sampling using OCR bounding boxes
# ---------------------------------------------------------------------------

def _detect_active_day_by_color(
    image: Image.Image,
    text_blocks: list[dict],
) -> Optional[str]:
    """
    Identifies the active day tab by cropping to each day token's bounding box
    and sampling the background colour of that region.

    Why this works:
        The active tab renders as white/dark text on an orange background.
        Inactive tabs have the same text on a grey/white background.
        We don't read the text — we read the colour of the area around
        the text. Orange → active. Grey → inactive.

    Process:
        1. Find every text block whose text is a day abbreviation.
        2. For each, expand the bounding box by TAB_BBOX_PADDING pixels to
           capture the tab background (not just the tight text glyph area).
        3. Crop that region from the PIL image.
        4. Compute the average RGB of the crop.
        5. Convert to HSV and check against orange thresholds.
        6. Return the first day whose tab is orange, or None if none match.

    Args:
        image:       PIL Image of the screenshot.
        text_blocks: OCR text block dicts containing bounding box data.

    Returns:
        Canonical day string (e.g. "friday") or None if no orange tab found.
    """
    img_w, img_h = image.size

    for block in text_blocks:
        text = block["text"].strip()
        canonical = normalize_day_label(text)
        if canonical is None:
            continue

        bbox = block.get("bbox")
        if not bbox:
            continue

        # Extract pixel coordinates from bounding box
        left, top, right, bottom = _bbox_to_pixel_coords(bbox)
        if left is None:
            continue

        # Expand bounding box to capture tab background, clamped to image bounds
        left   = max(0,     left   - TAB_BBOX_PADDING)
        top    = max(0,     top    - TAB_BBOX_PADDING)
        right  = min(img_w, right  + TAB_BBOX_PADDING)
        bottom = min(img_h, bottom + TAB_BBOX_PADDING)

        if right <= left or bottom <= top:
            continue

        # Crop and sample average colour
        crop = image.crop((left, top, right, bottom))
        avg_rgb = _average_rgb(crop)
        is_active = _is_orange_hsv(avg_rgb)

        logger.debug(
            "Tab colour sample",
            extra={
                "day": canonical,
                "crop_box": f"({left},{top},{right},{bottom})",
                "avg_rgb": avg_rgb,
                "is_orange": is_active,
            },
        )

        if is_active:
            return canonical

    return None


def _is_orange_hsv(rgb: tuple[int, int, int]) -> bool:
    """
    Returns True if an RGB colour falls within the orange range of the
    Last War active-tab background using HSV thresholds.

    HSV is used instead of raw RGB because it separates hue (colour identity)
    from saturation and brightness, making the check robust to lighting
    variation and compression artefacts across different devices.

    Args:
        rgb: (R, G, B) tuple with values 0–255.

    Returns:
        True if the colour is orange (active tab background).
    """
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return ORANGE_H_MIN <= h <= ORANGE_H_MAX and s >= ORANGE_S_MIN and v >= ORANGE_V_MIN


def _average_rgb(crop: Image.Image) -> tuple[int, int, int]:
    """
    Computes the average RGB colour of a PIL Image crop.

    Converts to RGB first to handle any mode (RGBA, P, etc.) that might
    appear in uploaded screenshots.

    Args:
        crop: Any PIL Image.

    Returns:
        (R, G, B) tuple of average pixel values, each 0–255.
    """
    rgb_crop = crop.convert("RGB")
    pixels = list(rgb_crop.getdata())
    if not pixels:
        return (0, 0, 0)
    r = sum(p[0] for p in pixels) // len(pixels)
    g = sum(p[1] for p in pixels) // len(pixels)
    b = sum(p[2] for p in pixels) // len(pixels)
    return (r, g, b)


def _bbox_to_pixel_coords(bbox) -> tuple:
    """
    Extracts (left, top, right, bottom) pixel coordinates from a bounding box.

    Handles both Vision API proto objects (with .vertices attribute) and
    plain dicts (from JSON fixtures). Returns (None, None, None, None) on
    any failure so the caller can skip gracefully.

    Args:
        bbox: BoundingPoly proto or dict with 'vertices' key.

    Returns:
        (left, top, right, bottom) integers or (None, None, None, None).
    """
    try:
        # Proto object from live Vision API
        vertices = list(bbox.vertices)
        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]
        return min(xs), min(ys), max(xs), max(ys)
    except AttributeError:
        pass

    try:
        # Dict form from JSON fixtures
        vertices = bbox.get("vertices", [])
        if not vertices:
            return None, None, None, None
        xs = [v.get("x", 0) for v in vertices]
        ys = [v.get("y", 0) for v in vertices]
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, AttributeError):
        return None, None, None, None


# ---------------------------------------------------------------------------
# Pass 2 — OCR text helpers
# ---------------------------------------------------------------------------

def _ocr_detect_strength(all_text_lower: set[str]) -> bool:
    """Returns True if Strength Ranking markers are present in the OCR text set."""
    if "strength ranking" in all_text_lower:
        return True
    if "power" in all_text_lower and "kills" in all_text_lower and "donation" in all_text_lower:
        return True
    return False


def _ocr_detect_weekly(all_text_lower: set[str]) -> bool:
    """
    Returns True if Weekly Rank markers are present without active day tabs.

    Both Weekly and Daily screens show both tab labels as text, so the key
    discriminator is the absence of day tab abbreviations (Mon./Tues./etc.)
    which only appear as a visible row on Daily screens.
    """
    if "weekly" not in all_text_lower:
        return False

    day_tabs_visible = any(
        abbr in all_text_lower for abbr in _DAY_ABBREVIATIONS
    )
    return not day_tabs_visible


def _ocr_detect_active_day_by_text(text_blocks: list[dict]) -> Optional[str]:
    """
    Text-only fallback for day detection when no PIL image is available.

    Used in unit tests that pass OCR fixture text blocks without an image.
    Scores each day token: +2 for no trailing period (active tab signal),
    +1 for with period (inactive tab). Returns the highest-scoring day.

    This is less reliable than colour sampling on real screenshots because
    the no-period signal can appear in other OCR contexts (e.g. announcement
    banners). Only used when image is not available.

    Args:
        text_blocks: OCR text block dicts.

    Returns:
        Canonical day string or None.
    """
    DAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    NO_PERIOD = {"mon", "tues", "wed", "thur", "fri", "sat"}

    scores: dict[str, int] = {}

    for block in text_blocks:
        raw = block["text"].strip()
        canonical = normalize_day_label(raw)
        if canonical is None:
            continue
        is_no_period = raw.rstrip(".").lower() in NO_PERIOD and not raw.endswith(".")
        points = 2 if is_no_period else 1
        scores[canonical] = scores.get(canonical, 0) + points

    if not scores:
        return None

    return max(
        scores.keys(),
        key=lambda d: (scores[d], -DAY_ORDER.index(d) if d in DAY_ORDER else 0),
    )


def _avg_y_from_bbox(bbox) -> float:
    """Computes average Y from a bounding box proto or dict."""
    try:
        vertices = list(bbox.vertices)
        return sum(v.y for v in vertices) / len(vertices)
    except AttributeError:
        pass
    try:
        vertices = bbox.get("vertices", [])
        return sum(v.get("y", 0) for v in vertices) / len(vertices)
    except (TypeError, ZeroDivisionError):
        return 0.0
