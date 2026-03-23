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
    screen type. This pass is also reused to bootstrap fixture data via the
    capture_ocr_fixture tool.

Classification priority order (prevents mis-routing):
    1. Strength Ranking  — unique header + different tab set, caught first
    2. Weekly Rank       — "Weekly Rank" tab is orange, no day tabs active
    3. Daily Rank        — specific day tab is orange

Coordinates used in colour sampling are expressed as fractions of image
dimensions so results are consistent across all device resolutions.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from app.utils.image_utils import sample_color_region, is_orange
from app.utils.text_utils import normalize_day_label
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum confidence from Pass 1 required to skip the OCR fallback.
# Lowering this improves accuracy at the cost of more Vision API calls.
CONFIDENCE_THRESHOLD = 0.80

# Approximate vertical position of the tab bar as a fraction of image height.
# Tuned against sample screenshots — the tab row sits at roughly 18–22% down.
TAB_BAR_Y_FRACTION = 0.20

# Approximate vertical position of the top header (title text area)
HEADER_Y_FRACTION = 0.10

# Horizontal positions of each UI element as fractions of image width.
# "Weekly Rank" tab is approximately centred at 65% when it is the active orange tab.
# Day tabs span from ~10% (Mon) to ~90% (Sat) across the tab bar.
WEEKLY_TAB_X_FRACTION = 0.65

DAY_TAB_X_POSITIONS = {
    "monday":    0.10,
    "tuesday":   0.26,
    "wednesday": 0.42,
    "thursday":  0.58,
    "friday":    0.74,
    "saturday":  0.90,
}

# Horizontal position of the "Power / Kills / Donation" tab set on the
# Strength Ranking screen. Only that screen has a tab at ~0.15 labelled "Power".
STRENGTH_POWER_TAB_X = 0.15

# Known text markers used in Pass 2 OCR-based classification
_STRENGTH_MARKERS = {"strength ranking", "strength  ranking", "power", "kills", "donation"}
_WEEKLY_MARKERS   = {"weekly rank"}
_DAY_MARKERS      = set(normalize_day_label(d) for d in [
    "Mon", "Tues", "Wed", "Thur", "Fri", "Sat",
    "Mon.", "Tues.", "Wed.", "Thur.", "Fri.", "Sat.",
] if normalize_day_label(d))


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
    # --- Try Strength Ranking first (most distinct UI) ---
    result, confidence = _detect_strength_ranking(image)
    if result:
        logger.debug(
            "Pass 1: Strength Ranking detected",
            extra={"image_filename": filename, "confidence": confidence},
        )
        return "power", confidence

    # --- Try Weekly Rank ---
    result, confidence = _detect_weekly_rank(image)
    if result:
        logger.debug(
            "Pass 1: Weekly Rank detected",
            extra={"image_filename": filename, "confidence": confidence},
        )
        return "weekly", confidence

    # --- Try Daily Rank day tabs ---
    day, confidence = _detect_active_day(image)
    if day:
        logger.debug(
            "Pass 1: Daily Rank detected",
            extra={"image_filename": filename, "day": day, "confidence": confidence},
        )
        return day, confidence

    logger.debug(
        "Pass 1: Classification ambiguous",
        extra={"image_filename": filename, "confidence": 0.0},
    )
    return None, 0.0


def classify_from_ocr_text(
    text_blocks: list[dict],
    filename: str = "",
) -> tuple[Optional[str], float]:
    """
    Classifies a screenshot using text blocks returned by the Vision API.

    This is the Pass 2 fallback called when Pass 1 confidence is below
    CONFIDENCE_THRESHOLD. It is also used directly in tests via OCR fixtures
    so the full classification logic can be exercised without colour sampling.

    Text blocks are expected in the format produced by ocr_client.extract_text_blocks():
        [{"text": "some string", "bbox": BoundingBox}, ...]

    Classification priority is identical to Pass 1: Strength → Weekly → Daily.

    Args:
        text_blocks: List of OCR text block dicts sorted top-to-bottom.
        filename:    Original filename for logging.

    Returns:
        Tuple of (category_string_or_None, confidence_float).
        Confidence is always 1.0 on a text match or 0.0 on failure — OCR
        evidence is treated as definitive.
    """
    all_text_lower = {block["text"].strip().lower() for block in text_blocks}

    # Strength Ranking: unique header text present
    if _ocr_detect_strength(all_text_lower):
        logger.debug("Pass 2: Strength Ranking detected via OCR", extra={"image_filename": filename})
        return "power", 1.0

    # Weekly Rank: "weekly rank" text present and no active day tab
    if _ocr_detect_weekly(all_text_lower):
        logger.debug("Pass 2: Weekly Rank detected via OCR", extra={"image_filename": filename})
        return "weekly", 1.0

    # Daily Rank: find active day
    day = _ocr_detect_active_day(text_blocks, all_text_lower)
    if day:
        logger.debug("Pass 2: Daily Rank detected via OCR", extra={"image_filename": filename, "day": day})
        return day, 1.0

    logger.warning(
        "Pass 2: OCR classification failed — no definitive markers found",
        extra={"image_filename": filename, "detected_tokens": list(all_text_lower)[:20]},
    )
    return None, 0.0


# ---------------------------------------------------------------------------
# Pass 1 — colour-sampling helpers
# ---------------------------------------------------------------------------

def _detect_strength_ranking(image: Image.Image) -> tuple[bool, float]:
    """
    Checks for the Strength Ranking screen by sampling the orange Power tab.

    The Strength Ranking screen has a unique horizontal tab set at the top:
    Power | Kills | Donation. The active "Power" tab sits at approximately
    x=15%, y=20% and is orange. This position has no orange element on
    Daily or Weekly screens, making it a strong discriminator.

    Returns:
        (True, confidence) if the region is orange, (False, 0.0) otherwise.
    """
    rgb = sample_color_region(image, x_fraction=STRENGTH_POWER_TAB_X, y_fraction=TAB_BAR_Y_FRACTION)
    if is_orange(rgb):
        return True, 0.90
    return False, 0.0


def _detect_weekly_rank(image: Image.Image) -> tuple[bool, float]:
    """
    Checks for the Weekly Rank screen by sampling the Weekly Rank tab position.

    On Weekly Rank screens the right-side tab ("Weekly Rank") is orange.
    On Daily screens the left-side tab ("Daily Rank") is orange instead.
    Sampling the right-tab position for orange gives a strong signal.

    Returns:
        (True, confidence) if the right tab is orange, (False, 0.0) otherwise.
    """
    rgb = sample_color_region(image, x_fraction=WEEKLY_TAB_X_FRACTION, y_fraction=TAB_BAR_Y_FRACTION - 0.05)
    if is_orange(rgb):
        return True, 0.88
    return False, 0.0


def _detect_active_day(image: Image.Image) -> tuple[Optional[str], float]:
    """
    Identifies the active day tab by scanning each day's expected position
    for an orange colour.

    Day tabs span the full width of the screen. Each day's approximate
    horizontal position is defined in DAY_TAB_X_POSITIONS. The active tab
    is orange; inactive tabs are grey/white.

    Returns:
        (day_string, confidence) for the first orange day tab found, or
        (None, 0.0) if no orange day tab is detected.
    """
    for day, x_frac in DAY_TAB_X_POSITIONS.items():
        rgb = sample_color_region(image, x_fraction=x_frac, y_fraction=TAB_BAR_Y_FRACTION)
        if is_orange(rgb):
            return day, 0.85

    return None, 0.0


# ---------------------------------------------------------------------------
# Pass 2 — OCR text helpers
# ---------------------------------------------------------------------------

def _ocr_detect_strength(all_text_lower: set[str]) -> bool:
    """
    Returns True if Strength Ranking markers are present in the OCR text set.

    Requires "strength ranking" OR both "power" and "kills" to appear together
    (since "power" alone could appear in other contexts).
    """
    if "strength ranking" in all_text_lower:
        return True
    if "power" in all_text_lower and "kills" in all_text_lower and "donation" in all_text_lower:
        return True
    return False


def _ocr_detect_weekly(all_text_lower: set[str]) -> bool:
    """
    Returns True if Weekly Rank markers are present without active day tabs.

    OCR returns individual words, so we check for "weekly" as a token rather
    than "weekly rank" as a combined string. Both Weekly and Daily screens
    show both tab labels, so the key discriminator is the absence of individual
    day tab abbreviations (Mon./Tues./etc.) which only appear on Daily screens.

    Decision logic:
        - "weekly" present + no day tab abbreviations → Weekly Rank screen
        - "weekly" present + day tab abbreviations present → Daily Rank screen
        - "weekly" absent → not a Weekly Rank screen
    """
    has_weekly = "weekly" in all_text_lower
    has_daily  = "daily" in all_text_lower

    if not has_weekly:
        return False

    # Day tab abbreviations only appear on Daily screens
    day_tabs_visible = any(
        day_abbr in all_text_lower
        for day_abbr in ["mon.", "tues.", "wed.", "thur.", "fri.", "sat."]
    )

    if day_tabs_visible:
        return False

    # Weekly present, no day tabs — this is the Weekly Rank screen
    return True


def _ocr_detect_active_day(
    text_blocks: list[dict],
    all_text_lower: set[str],
) -> Optional[str]:
    """
    Identifies the active day tab from OCR text blocks using a scoring approach.

    Why Y-position does not work on real screenshots:
        All 6 day tabs sit at the same Y coordinate in a single horizontal bar.
        A y_spread guard always fires and returns None on real data.

    Scoring strategy:
        The active tab renders as white text on orange background. Vision API
        tends to read it without a trailing period (e.g. "Fri" rather than
        "Fri.") while inactive tabs retain the period ("Mon.", "Thur.").
        We score each day +2 for a no-period occurrence and +1 for a
        with-period occurrence. The highest-scoring day is returned.

        Tiebreaker: among equal-scoring days, prefer the one with the
        highest raw score, then earlier in Mon-Sat order as a last resort.

    Args:
        text_blocks:    Full list of OCR text block dicts.
        all_text_lower: Pre-computed set of lowercased text values.

    Returns:
        Canonical day string (e.g. "thursday") or None if no day tokens found.
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

    best_day = max(
        scores.keys(),
        key=lambda d: (scores[d], -DAY_ORDER.index(d) if d in DAY_ORDER else 0),
    )
    return best_day


def _avg_y_from_bbox(bbox) -> float:
    """
    Computes the average Y coordinate of a Vision API bounding box.

    The Vision API bounding box can be either a BoundingPoly object with
    .vertices or a plain dict. This helper handles both for flexibility
    across production and test fixture contexts.

    Args:
        bbox: BoundingPoly object or dict with 'vertices' key.

    Returns:
        Average Y value as a float.
    """
    try:
        # Proto object from live Vision API
        vertices = list(bbox.vertices)
        return sum(v.y for v in vertices) / len(vertices)
    except AttributeError:
        pass

    try:
        # Dict form from JSON fixtures
        vertices = bbox.get("vertices", [])
        return sum(v.get("y", 0) for v in vertices) / len(vertices)
    except (TypeError, ZeroDivisionError):
        return 0.0
