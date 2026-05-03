"""
app/pipeline/classifier.py

Screenshot classification for Last War: Survival ranking screens.

Single entry point — `classify_from_ocr_text(blocks, image)` — runs after the
stitch-first OCR step. Looks at the OCR text first to identify the screen
type, then samples the source image to resolve which tab is active. The
per-frame `pre_ocr_hint` colour-sample fast path was removed when the
pipeline switched to stitch-first; it is still documented in the
screen-definition schema as an optional fast-skip for the Android scanner,
but the OCR service no longer consumes it.

Classification priority order (prevents mis-routing):
    1. Strength Ranking      — unique header + different tab set, caught first
    2. Alliance Contribution — multi-row tab groups joined as `{cat}_{period}`
    3. Weekly Rank           — "Weekly Rank" tab is orange, no day tabs active
    4. Daily Rank day        — bounding-box colour sampling of each day tab
    5. Daily Rank text       — scoring fallback when image not available (tests)

Coordinates used inside per-tab colour sampling come from OCR bounding boxes
(not normalised image fractions), so the classifier degrades gracefully when
the game UI is letterboxed inside a wider screen.
"""

from __future__ import annotations

import colorsys
from typing import Optional

from PIL import Image

from app.pipeline.screen_definitions import get_definition
from app.utils.text_utils import normalize_day_label
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_from_ocr_text(
    text_blocks: list[dict],
    image: Optional[Image.Image] = None,
    filename: str = "",
) -> tuple[Optional[str], float]:
    """
    Classifies a screenshot using text blocks returned by the OCR engine.

    The stitch-first pipeline always supplies a PIL image, enabling bounding-
    box colour sampling for all tab detection.

    Args:
        text_blocks: List of OCR text block dicts sorted top-to-bottom.
        image:       PIL Image of the section being classified.
        filename:    Original filename for logging.

    Returns:
        Tuple of (category_string_or_None, confidence_float).
        Confidence is 1.0 on a definitive match, 0.95 on a colour-based
        daily-rank match, or 0.0 on failure.
    """
    all_text_lower = {block["text"].strip().lower() for block in text_blocks}

    # 1. Strength Ranking — unique header text; colour-sample the active tab
    if _ocr_detect_strength(all_text_lower):
        tab = _detect_active_strength_tab(image, text_blocks) if image is not None else None
        category = tab or "power"
        logger.debug(
            "Pass 2: Strength Ranking detected via OCR",
            extra={"image_filename": filename, "active_tab": category},
        )
        return category, 1.0

    # 2. Alliance Contribution — unique title; row-1 (orange) + row-2 (brightest)
    if _ocr_detect_alliance_contribution(all_text_lower):
        category = _detect_active_ac_tab(image, text_blocks) if image is not None else None
        if category:
            logger.debug(
                "Pass 2: Alliance Contribution detected via OCR",
                extra={"image_filename": filename, "active_tab": category},
            )
            return category, 1.0
        logger.warning(
            "Pass 2: Alliance Contribution detected but tab resolution failed",
            extra={"image_filename": filename},
        )
        return None, 0.0

    # 3. Weekly Rank — weekly token without day tab abbreviations
    if _ocr_detect_weekly(all_text_lower):
        logger.debug("Pass 2: Weekly Rank detected via OCR",
                     extra={"image_filename": filename})
        return "weekly", 1.0

    # 4. Daily Rank — bounding-box colour sampling of each day tab region
    if image is not None:
        day = _detect_active_day_by_color(image, text_blocks)
        if day:
            logger.debug("Pass 2: Daily Rank detected via bounding-box colour sampling",
                         extra={"image_filename": filename, "day": day})
            return day, 0.95

    # 5. Daily Rank — text scoring fallback when no image is available
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
# HSV / colour helpers
# ---------------------------------------------------------------------------

def _is_orange_hsv_thresholds(
    rgb: tuple[int, int, int],
    h_min: float,
    h_max: float,
    s_min: float,
    v_min: float,
) -> bool:
    """HSV orange check using caller-supplied thresholds."""
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h_min <= h <= h_max and s >= s_min and v >= v_min


# ---------------------------------------------------------------------------
# Pass 2 — colour sampling using OCR bounding boxes
# ---------------------------------------------------------------------------

def _detect_active_day_by_color(
    image: Image.Image,
    text_blocks: list[dict],
) -> Optional[str]:
    """
    Identifies the active day tab by comparing background brightness across
    all day tab regions.

    Observation from real screenshots:
        The active day tab has a WHITE/LIGHT background (high V, low S).
        Inactive tabs have a GREY background (lower V, slightly warmer).
        The orange colour is on the top-level "Daily Rank" tab, NOT the day row.

        Active tab:   V ≈ 0.74–0.81, S ≈ 0.019–0.025  (white/light)
        Inactive tab: V ≈ 0.67–0.70, S ≈ 0.090–0.094  (warm grey)

    Strategy:
        Sample the background colour of every day tab bounding box.
        Return the day whose crop has the highest brightness (V value).
        To avoid false positives from image noise, require that the brightest
        tab is at least BRIGHTNESS_GAP brighter than the second-brightest.

    Args:
        image:       PIL Image of the screenshot.
        text_blocks: OCR text block dicts containing bounding box data.

    Returns:
        Canonical day string (e.g. "friday") or None if no clear winner.
    """
    img_w, img_h = image.size

    daily_defn = get_definition("daily_ranking")
    if daily_defn and daily_defn.tabs:
        ai = daily_defn.tabs.active_indicator
        BRIGHTNESS_GAP = ai.min_gap
        PAD = max(1, round(img_w * ai.bbox_padding_fraction))
    else:
        BRIGHTNESS_GAP = 0.04
        PAD = max(1, round(img_w * 0.007))

    day_brightness: dict[str, float] = {}

    for block in text_blocks:
        text = block["text"].strip()
        canonical = normalize_day_label(text)
        if canonical is None:
            continue

        bbox = block.get("bbox")
        if not bbox:
            continue

        left, top, right, bottom = _bbox_to_pixel_coords(bbox)
        if left is None:
            continue

        left   = max(0,     left   - PAD)
        top    = max(0,     top    - PAD)
        right  = min(img_w, right  + PAD)
        bottom = min(img_h, bottom + PAD)

        if right <= left or bottom <= top:
            continue

        crop = image.crop((left, top, right, bottom))
        r, g, b = _average_rgb(crop)
        _, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

        logger.debug(
            "Tab colour sample",
            extra={
                "day": canonical,
                "crop_box": f"({left},{top},{right},{bottom})",
                "avg_rgb": f"rgb({r},{g},{b})",
                "hsv": f"s={s:.3f} v={v:.3f}",
                "brightness": round(v, 3),
            },
        )

        # Keep the highest brightness seen per day (handles duplicate tokens)
        if canonical not in day_brightness or v > day_brightness[canonical]:
            day_brightness[canonical] = v

    if not day_brightness:
        return None

    # Sort by brightness descending
    ranked = sorted(day_brightness.items(), key=lambda x: x[1], reverse=True)

    best_day, best_v = ranked[0]

    # Require a clear brightness gap to avoid noise returning a wrong result
    if len(ranked) > 1:
        second_v = ranked[1][1]
        if best_v - second_v < BRIGHTNESS_GAP:
            logger.debug(
                "Tab colour sampling inconclusive — brightness gap too small",
                extra={"best": best_day, "best_v": round(best_v, 3),
                       "second_v": round(second_v, 3), "gap": round(best_v - second_v, 3)},
            )
            return None

    return best_day


# ---------------------------------------------------------------------------
# Strength Ranking — active tab detection
# ---------------------------------------------------------------------------

def _detect_active_strength_tab(
    image: Image.Image,
    text_blocks: list[dict],
) -> Optional[str]:
    """
    Identifies the active Strength Ranking tab.

    Strength Ranking is split into two definitions: `strength_donation`
    (Daily / Weekly sub-tabs visible) and `strength_metrics` (Power / Kills
    only). We try strength_donation first because its more-specific tab
    items only match when the Donation sub-tab row is rendered; if no
    sub-tab tab fires, fall through to strength_metrics for Power / Kills.
    """
    for screen_id in _STRENGTH_SCREEN_IDS:
        defn = get_definition(screen_id)
        if defn is None or not defn.tabs:
            continue
        category = _detect_active_tab_in_definition(image, text_blocks, defn)
        if category is not None:
            return category
    return None


def _detect_active_tab_in_definition(
    image: Image.Image,
    text_blocks: list[dict],
    defn,
) -> Optional[str]:
    """
    Identifies the active tab inside a single Strength-style definition by
    locating each tab item's signal text in the OCR blocks, sampling the
    surrounding region for orange, and (when a top-level signal maps to
    multiple sub-tabs) selecting the active sub-tab by brightness.

    All colour thresholds and padding are read from the definition so that
    tuning the YAML is the only change needed.
    """
    img_w, img_h = image.size
    ai  = defn.tabs.active_indicator
    pad = max(4, round(img_w * ai.bbox_padding_fraction))

    # Orange thresholds from active_indicator colour definition
    if ai.color and ai.color.hsv_override:
        o = ai.color.hsv_override
        h_min, h_max, s_min, v_min = o.h_min, o.h_max, o.s_min, o.v_min
    else:
        h_min, h_max, s_min, v_min = 0.014, 0.153, 0.40, 0.55

    # Group tabs by first signal — unique first signals are the top-level tabs
    tab_groups: dict[str, list] = {}
    for tab in defn.tabs.items:
        if not tab.signals:
            continue
        top_label = tab.signals[0].lower()
        tab_groups.setdefault(top_label, []).append(tab)

    # Keep only the topmost (tab-bar) occurrence of each top-level label
    top_tab_blocks: dict[str, dict] = {}
    for block in text_blocks:
        t = block["text"].strip().lower()
        if t in tab_groups:
            if t not in top_tab_blocks or block["avg_y"] < top_tab_blocks[t]["avg_y"]:
                top_tab_blocks[t] = block

    for tab_label, block in top_tab_blocks.items():
        left, top, right, bottom = _bbox_to_pixel_coords(block.get("bbox"))
        if left is None:
            continue

        crop = image.crop((
            max(0,     left  - pad),
            max(0,     top   - pad),
            min(img_w, right + pad),
            min(img_h, bottom + pad),
        ))
        avg_rgb = _average_rgb(crop)
        if not _is_orange_hsv_thresholds(avg_rgb, h_min, h_max, s_min, v_min):
            continue

        tabs_in_group = tab_groups[tab_label]
        if len(tabs_in_group) == 1:
            return tabs_in_group[0].category

        # Multiple tabs share this top-level label — detect active sub-tab by brightness
        sub = _detect_active_subtab(
            image, text_blocks, img_w, img_h, pad, tabs_in_group, ai.min_gap
        )
        return sub if sub else tabs_in_group[0].category

    return None


def _detect_active_subtab(
    image: Image.Image,
    text_blocks: list[dict],
    img_w: int,
    img_h: int,
    pad: int,
    sub_tab_items: list,
    brightness_gap: float,
) -> Optional[str]:
    """
    Determines which sub-tab is active by comparing background brightness.

    Used when multiple tab items share the same first-signal top-level label
    (e.g. donation_daily and donation_weekly both start with "Donation").
    The second signal of each item is the sub-tab label (e.g. "Daily", "Weekly").

    The active sub-tab has a brighter (whiter) background than inactive ones.
    Returns None if the brightness gap is below the definition's threshold.

    Args:
        sub_tab_items:  TabItem objects sharing the same top-level first signal.
        brightness_gap: Minimum V-channel gap required to pick a winner;
                        sourced from active_indicator.min_gap in the definition.
    """
    # Map second signal (lowercased) → category, derived from definition
    sub_label_to_category = {
        tab.signals[1].lower(): tab.category
        for tab in sub_tab_items
        if len(tab.signals) >= 2
    }
    if not sub_label_to_category:
        return None

    sub_brightness: dict[str, float] = {}

    for block in text_blocks:
        t = block["text"].strip().lower()
        if t not in sub_label_to_category:
            continue

        left, top, right, bottom = _bbox_to_pixel_coords(block.get("bbox"))
        if left is None:
            continue

        crop = image.crop((
            max(0,     left  - pad),
            max(0,     top   - pad),
            min(img_w, right + pad),
            min(img_h, bottom + pad),
        ))
        avg_rgb = _average_rgb(crop)
        r, g, b  = avg_rgb
        _, _s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

        if t not in sub_brightness or v > sub_brightness[t]:
            sub_brightness[t] = v

    if not sub_brightness:
        return None

    ranked = sorted(sub_brightness.items(), key=lambda x: x[1], reverse=True)
    best_key, best_v = ranked[0]

    if len(ranked) > 1 and best_v - ranked[1][1] < brightness_gap:
        logger.debug(
            "Sub-tab brightness gap too small — inconclusive",
            extra={"best": best_key, "gap": round(best_v - ranked[1][1], 3)},
        )
        return None

    return sub_label_to_category[best_key]


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

    Handles three formats:
    1. Vision API BoundingPoly proto object (live API calls)
    2. Dict with "vertices" list of {"x": int, "y": int} dicts (ideal JSON)
    3. Proto string representation — produced when capture_ocr_fixture.py
       falls back to str() serialisation, e.g.:
           "vertices {\n  x: 406\n  y: 451\n}\n..."
       This format is parsed with regex.

    Returns (None, None, None, None) on any failure so the caller skips
    gracefully rather than crashing.

    Args:
        bbox: BoundingPoly proto, dict with vertices, or proto string.

    Returns:
        (left, top, right, bottom) integers or (None, None, None, None).
    """
    import re

    # 1. Proto object from live Vision API
    try:
        vertices = list(bbox.vertices)
        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]
        if xs and ys:
            return min(xs), min(ys), max(xs), max(ys)
    except AttributeError:
        pass

    # 2. Dict form {"vertices": [{"x": int, "y": int}, ...]}
    if isinstance(bbox, dict):
        try:
            vertices = bbox.get("vertices", [])
            if vertices and isinstance(vertices[0], dict):
                xs = [v.get("x", 0) for v in vertices]
                ys = [v.get("y", 0) for v in vertices]
                if xs and ys:
                    return min(xs), min(ys), max(xs), max(ys)
        except (TypeError, IndexError):
            pass

    # 3. Proto string: "vertices {\n  x: 406\n  y: 451\n}\n..."
    if isinstance(bbox, str) and "vertices" in bbox:
        try:
            xs = [int(v) for v in re.findall(r'x:\s*(-?\d+)', bbox)]
            ys = [int(v) for v in re.findall(r'y:\s*(-?\d+)', bbox)]
            if xs and ys:
                return min(xs), min(ys), max(xs), max(ys)
        except (ValueError, TypeError):
            pass

    return None, None, None, None


# ---------------------------------------------------------------------------
# Pass 2 — OCR text helpers
# ---------------------------------------------------------------------------

_STRENGTH_SCREEN_IDS = ("strength_donation", "strength_metrics")


def _ocr_detect_strength(all_text_lower: set[str]) -> bool:
    """
    Returns True if any Strength Ranking variant matches the OCR text set.

    Strength Ranking is split into two screen definitions in the catalog —
    `strength_metrics` (Power / Kills row-1 tabs) and `strength_donation`
    (Donation row-1 tab + Daily / Weekly sub-tabs). Either variant matching
    counts as detection.

    Per-variant rule (mirrors the Consumer Contract in
    lastwar-screen-definitions/README.md):
    - Match if any page_signal's words all appear in the token set.
    - Reject if any negative_signal's words all appear in the token set.
    """
    for screen_id in _STRENGTH_SCREEN_IDS:
        defn = get_definition(screen_id)
        if defn is None:
            continue
        if any(
            all(w in all_text_lower for w in neg.lower().split())
            for neg in defn.negative_signals
        ):
            continue
        if any(
            all(w in all_text_lower for w in sig.lower().split())
            for sig in defn.page_signals
        ):
            return True
    return False


def _ocr_detect_weekly(all_text_lower: set[str]) -> bool:
    """
    Returns True if Weekly Rank markers are present without active day tabs.

    Driven by the weekly_ranking screen definition:
    - page_signals provide the "weekly" discriminator words
    - negative_signals list the day-tab abbreviations (Mon., Tues., etc.) whose
      presence means this is a Daily screen, not Weekly
    """
    defn = get_definition("weekly_ranking")
    if defn is None:
        return False

    # Match if any page signal has ALL its words present as individual tokens
    # (avoids false-positives from common words like "ranking" that appear on
    # other screens; "weekly" combined with "rank" is a reliable discriminator)
    if not any(
        all(w in all_text_lower for w in s.lower().split())
        for s in defn.page_signals
    ):
        return False

    neg_signals = {s.lower() for s in defn.negative_signals}
    return not (neg_signals & all_text_lower)


def _ocr_detect_active_day_by_text(
    text_blocks: list[dict],
    all_text_lower: set[str] = None,  # accepted but unused — kept for backward compat
) -> Optional[str]:
    """
    Text-only fallback for day detection when no PIL image is available.

    Used in unit tests that pass OCR fixture text blocks without an image.
    Scores each day token: +2 for no trailing period (active tab signal),
    +1 for with period (inactive tab). Returns the highest-scoring day.

    This is less reliable than colour sampling on real screenshots because
    the no-period signal can appear in other OCR contexts (e.g. announcement
    banners). Only used when image is not available.

    Args:
        text_blocks:    OCR text block dicts.
        all_text_lower: Ignored — accepted for backward compatibility with
                        older call sites that passed a pre-computed token set.

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


# ---------------------------------------------------------------------------
# Alliance Contribution — page detection + active tab resolution
# ---------------------------------------------------------------------------

def _ocr_detect_alliance_contribution(all_text_lower: set[str]) -> bool:
    """
    Returns True if Alliance Contribution markers are present in the OCR text set.
    Mirrors the rule documented in lastwar-screen-definitions/README.md
    Consumer Contract — every page_signal word must appear, no negative_signal
    word-set may all appear.
    """
    defn = get_definition("alliance_contribution")
    if defn is None:
        return False
    if any(
        all(w in all_text_lower for w in neg.lower().split())
        for neg in defn.negative_signals
    ):
        return False
    return any(
        all(w in all_text_lower for w in sig.lower().split())
        for sig in defn.page_signals
    )


def _detect_active_ac_tab(
    image: Image.Image,
    text_blocks: list[dict],
) -> Optional[str]:
    """
    Identifies the active Alliance Contribution category+period combo.

    The screen has two independent tab rows declared via `tabs.groups` in the
    YAML — by convention `category` (Mutual Assistance / Siege / Rare Soil War /
    Defeat, orange-fill `color_fraction`) and `period` (Daily / Weekly /
    Season Total Ranking, white-text `brightest`). Each group's winner is
    determined independently and the winners are joined with `_` in the
    declaration order of `tabs.groups`, matching the wire-format category
    documented in the Consumer Contract (e.g. `siege_daily`).

    Returns None if either group cannot be resolved confidently.
    """
    defn = get_definition("alliance_contribution")
    if defn is None or not defn.tabs or not defn.tabs.groups:
        return None

    img_w, img_h = image.size
    ai = defn.tabs.active_indicator
    pad = max(4, round(img_w * ai.bbox_padding_fraction))

    # Orange thresholds, with documented RGB-fallback equivalents
    if ai.color and ai.color.hsv_override:
        o = ai.color.hsv_override
        h_min, h_max, s_min, v_min = o.h_min, o.h_max, o.s_min, o.v_min
    else:
        h_min, h_max, s_min, v_min = 0.014, 0.153, 0.40, 0.55

    # Group items by their `group` field
    items_by_group: dict[str, list] = {}
    for tab in defn.tabs.items:
        if tab.group:
            items_by_group.setdefault(tab.group, []).append(tab)

    winners: list[str] = []
    for group_id in defn.tabs.groups.keys():
        group_cfg = defn.tabs.groups[group_id]
        items = items_by_group.get(group_id, [])
        if not items:
            return None
        category = _detect_winner_in_ac_group(
            image, text_blocks, img_w, img_h, pad, items, group_cfg,
            h_min, h_max, s_min, v_min,
        )
        if category is None:
            return None
        winners.append(category)
    return "_".join(winners)


def _detect_winner_in_ac_group(
    image: Image.Image,
    text_blocks: list[dict],
    img_w: int,
    img_h: int,
    pad: int,
    items: list,
    group_cfg,
    h_min: float, h_max: float, s_min: float, v_min: float,
) -> Optional[str]:
    """
    Picks the active tab within a single AC group. Strategy is one of
    "color_fraction" (active tab has solid orange background — pick the
    candidate whose averaged crop is orange) or "brightest" (active tab
    has whiter text/background — pick the candidate with the highest V).
    """
    candidates: list[tuple[str, float]] = []  # (category, score)

    for tab in items:
        if not tab.signals:
            continue
        # Locate the OCR-region this tab corresponds to. Two strategies:
        #   1. Single-block match — a block whose text contains every word of
        #      one of the tab's signals.
        #   2. Multi-block match (fallback for OCR that splits a multi-word
        #      signal across adjacent blocks on the same row, e.g.
        #      "Season Total Ranking" rendered as separate "Season" / "Total"
        #      / "Ranking" blocks). Returns the union bbox so colour sampling
        #      covers the full pill background, not just one word's glyphs.
        rect = _find_tab_region(tab, text_blocks)
        if rect is None:
            continue
        left, top, right, bottom = rect
        crop = image.crop((
            max(0,     left  - pad),
            max(0,     top   - pad),
            min(img_w, right + pad),
            min(img_h, bottom + pad),
        ))
        avg_rgb = _average_rgb(crop)

        if group_cfg.strategy == "color_fraction":
            # Average-RGB orange test — same approach Strength uses.
            if _is_orange_hsv_thresholds(avg_rgb, h_min, h_max, s_min, v_min):
                # Score is "1.0" — any orange match wins; ties broken in declaration order.
                candidates.append((tab.category, 1.0))
        elif group_cfg.strategy == "brightest":
            # Count the fraction of near-white pixels in the crop. Active
            # pill backgrounds are bright; inactive tabs are mostly dark
            # text on a slightly-grey background. The fraction signal is
            # sharper than averaging the whole crop's V — it doesn't get
            # diluted by the (usually wider) bbox of multi-word matches
            # like the union rect for "Season Total Ranking".
            candidates.append((tab.category, _white_pixel_fraction(crop)))

    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[1])
    # For brightest, require a small min_gap between winner and runner-up to
    # avoid noise. Period sub-tabs are text-only (no solid fill), so the
    # active vs inactive V difference is small — measured ~0.039 on the
    # alliance_contribution period row. min_fraction (already ~0.02 for
    # text-only rows in YAML) is a reasonable proxy for the expected gap.
    if group_cfg.strategy == "brightest" and len(candidates) > 1:
        gap = candidates[0][1] - candidates[1][1]
        if gap < group_cfg.min_fraction:
            logger.debug(
                "AC sub-tab brightness gap too small — inconclusive",
                extra={"best": candidates[0][0], "gap": round(gap, 3),
                       "threshold": group_cfg.min_fraction},
            )
            return None
    return candidates[0][0]


# White-pixel detection threshold for the period-row brightest strategy.
# A pixel counts as "white" when every channel exceeds this. Calibrated
# against alliance_contribution period sub-tab fixtures: active pill
# backgrounds register ~50%+ near-white pixels, inactive ~5%.
_WHITE_PIXEL_MIN_CHANNEL = 215


def _white_pixel_fraction(crop: Image.Image) -> float:
    """Returns the fraction of pixels in ``crop`` whose every RGB channel
    exceeds _WHITE_PIXEL_MIN_CHANNEL — the canonical white-pill signal
    used to disambiguate active vs inactive period sub-tabs.

    Returns 0.0 for an empty crop. Converts to RGB internally.
    """
    rgb = crop.convert("RGB") if crop.mode != "RGB" else crop
    pixels = list(rgb.getdata())
    if not pixels:
        return 0.0
    threshold = _WHITE_PIXEL_MIN_CHANNEL
    white = sum(1 for r, g, b in pixels if r > threshold and g > threshold and b > threshold)
    return white / len(pixels)


# Same-row tolerance for the multi-block signal match below — two text
# blocks count as "on the same row" when their average-Y values are within
# this fraction of either's bbox height. 0.6 is loose enough to handle
# OCR's slight per-word baseline jitter without crossing into adjacent rows.
_SAME_ROW_FRACTION = 0.6


def _find_tab_region(tab, text_blocks: list[dict]) -> Optional[tuple[int, int, int, int]]:
    """
    Locate the OCR-text rectangle for ``tab``. Returns
    (left, top, right, bottom) pixel coordinates or None.

    Tries each of ``tab.signals`` in order. A signal matches when:

      1. **Single-block** — some OCR block's text contains every word of
         the signal (case-insensitive). The match block's bbox is returned.
      2. **Multi-block fallback** (only for multi-word signals, only used
         if no single block matched any signal) — for the signal's first
         word, find every block containing it; for each, look on the same
         OCR row for blocks containing each remaining word in the signal.
         If all words are present on one row, return the union bbox of all
         matched blocks.

    The fallback is what makes the alliance_contribution Season Total
    Ranking tab detectable: Cloud Vision often renders that label as three
    separate word-blocks ("Season", "Total", "Ranking") rather than one,
    so the single-block matcher silently misses it.
    """
    # Strategy 1 — single-block.
    for signal in tab.signals:
        sig_words = signal.lower().split()
        for block in text_blocks:
            btext = block["text"].strip().lower()
            if all(w in btext for w in sig_words):
                return _bbox_to_pixel_coords(block.get("bbox"))

    # Strategy 2 — multi-block on the same row, in left-to-right reading order.
    # Each subsequent word's block must sit to the right of the previous one
    # so we don't accidentally pull in a same-named word from a different tab
    # (e.g. the "Ranking" that belongs to "Weekly Ranking" when we're trying
    # to match the "Ranking" in "Season Total Ranking").
    for signal in tab.signals:
        sig_words = signal.lower().split()
        if len(sig_words) < 2:
            continue
        first_word = sig_words[0]
        for first_block in text_blocks:
            ftext = first_block["text"].strip().lower()
            if first_word not in ftext:
                continue
            # Compute row tolerance in pixels from this block's bbox height.
            f_left, f_top, f_right, f_bottom = _bbox_to_pixel_coords(first_block.get("bbox"))
            if f_left is None:
                continue
            row_tol = max(8, int((f_bottom - f_top) * _SAME_ROW_FRACTION))
            f_y = first_block["avg_y"]
            # Track the right-edge of the previously-matched word — the next
            # word's left edge must be at or beyond this (with a small slack
            # to allow OCR jitter in adjacent-block boundaries).
            row_slack = max(4, int(row_tol / 2))
            min_next_left = f_right - row_slack
            others: list[dict] = []
            ok = True
            for word in sig_words[1:]:
                hit = None
                hit_right = 0
                for b in text_blocks:
                    if b is first_block or b in others:
                        continue
                    if word not in b["text"].strip().lower():
                        continue
                    if abs(b["avg_y"] - f_y) > row_tol:
                        continue
                    bl, _bt, br, _bb = _bbox_to_pixel_coords(b.get("bbox"))
                    if bl is None or bl < min_next_left:
                        continue
                    # Take the *closest-to-the-left* match of those eligible —
                    # that's the next word in reading order.
                    if hit is None or bl < hit_right:
                        hit = b
                        hit_right = bl
                if hit is None:
                    ok = False
                    break
                others.append(hit)
                # Update min_next_left to this block's right edge.
                _hl, _ht, hr, _hb = _bbox_to_pixel_coords(hit.get("bbox"))
                if hr is not None:
                    min_next_left = hr - row_slack
            if not ok:
                continue
            # Union bbox of first_block + all `others`.
            union_left, union_top = f_left, f_top
            union_right, union_bottom = f_right, f_bottom
            for o in others:
                ol, ot, orr, ob = _bbox_to_pixel_coords(o.get("bbox"))
                if ol is None:
                    continue
                union_left   = min(union_left,   ol)
                union_top    = min(union_top,    ot)
                union_right  = max(union_right,  orr)
                union_bottom = max(union_bottom, ob)
            return (union_left, union_top, union_right, union_bottom)
    return None
