"""
tests/test_classifier.py

Unit tests for app/pipeline/classifier.py.

All tests use either synthetic OCR blocks (from conftest fixtures) or
real JSON fixtures auto-discovered from tests/fixtures/ocr_responses/.

The Vision API is never called. classify_from_ocr_text() is tested directly
with pre-built block lists that match the format ocr_client produces.

Real fixture tests infer the expected category from the fixture filename.
Fixture files should include a day name or screen type in their name, e.g.:
    Friday-215600.json   → expected category: friday
    Power-214600.json    → expected category: power
    Weekly-220909.json   → expected category: weekly
"""

import pytest
from pathlib import Path

from app.pipeline.classifier import (
    classify_from_ocr_text,
    _detect_active_day_by_color,
    _ocr_detect_strength,
    _ocr_detect_weekly,
    _ocr_detect_active_day_by_text as _ocr_detect_active_day,
)
from tests.conftest import FIXTURE_DIR, get_text_blocks, load_fixture, make_block


# ---------------------------------------------------------------------------
# Pass 2: Strength Ranking detection
# ---------------------------------------------------------------------------

class TestOcrDetectStrength:

    def test_detects_metrics_screen_header(self, strength_metrics_blocks):
        """Power/Kills variant matches via the broad 'Strength Ranking' page_signal."""
        all_lower = {b["text"].strip().lower() for b in strength_metrics_blocks}
        assert _ocr_detect_strength(all_lower) is True

    def test_detects_donation_screen_via_sub_tab_signals(self, strength_donation_blocks):
        """Donation variant matches via the 'Strength Daily Weekly' multi-word page_signal."""
        all_lower = {b["text"].strip().lower() for b in strength_donation_blocks}
        assert _ocr_detect_strength(all_lower) is True

    def test_rejects_power_alone(self):
        """Just 'power' + 'ranking' lacks the 'strength' token; no variant matches."""
        assert _ocr_detect_strength({"power", "ranking"}) is False

    def test_weekly_rank_not_detected_as_strength(self, weekly_rank_blocks):
        all_lower = {b["text"].strip().lower() for b in weekly_rank_blocks}
        assert _ocr_detect_strength(all_lower) is False


# ---------------------------------------------------------------------------
# Pass 2: Weekly Rank detection
# ---------------------------------------------------------------------------

class TestOcrDetectWeekly:

    def test_detects_weekly_rank_without_day_tabs(self, weekly_rank_blocks):
        all_lower = {b["text"].strip().lower() for b in weekly_rank_blocks}
        assert _ocr_detect_weekly(all_lower) is True

    def test_does_not_detect_weekly_when_day_tabs_present(self, friday_daily_blocks):
        all_lower = {b["text"].strip().lower() for b in friday_daily_blocks}
        assert _ocr_detect_weekly(all_lower) is False

    def test_does_not_detect_strength_metrics_as_weekly(self, strength_metrics_blocks):
        all_lower = {b["text"].strip().lower() for b in strength_metrics_blocks}
        assert _ocr_detect_weekly(all_lower) is False

    def test_does_not_detect_strength_donation_as_weekly(self, strength_donation_blocks):
        """Donation variant has 'Weekly' as a sub-tab label; ensure weekly_ranking
        doesn't latch onto it (its 'Mon.'-'Sat.' negatives don't cover this case,
        but 'Weekly Rank' as the page_signal requires both words)."""
        all_lower = {b["text"].strip().lower() for b in strength_donation_blocks}
        assert _ocr_detect_weekly(all_lower) is False


# ---------------------------------------------------------------------------
# Pass 2: Active day detection
# ---------------------------------------------------------------------------

class TestOcrDetectActiveDay:
    """
    The text fallback only resolves a day when a *single* day tab is present.
    A full Mon.–Sat. tab bar is ambiguous from text alone (the active tab is a
    colour signal, not a text one), so it returns None. The previous heuristic
    scored a +2 for any day OCR'd without a trailing period; that systematically
    mis-picked Thursday because Cloud Vision reliably drops the period on "Thur"
    regardless of which day is active (batch 7038e97).
    """

    def test_full_tab_bar_is_ambiguous_returns_none(self, friday_daily_blocks):
        all_lower = {b["text"].strip().lower() for b in friday_daily_blocks}
        day = _ocr_detect_active_day(friday_daily_blocks, all_lower)
        assert day is None

    def test_thursday_period_drop_does_not_win_over_other_days(self):
        """A full tab bar where only 'Thur' lost its period (the real OCR
        quirk) must NOT resolve to Thursday — it is ambiguous → None."""
        blocks = [
            make_block("Mon.",   75, 260),
            make_block("Tues.", 185, 260),
            make_block("Wed.",  295, 260),
            make_block("Thur",  405, 260),   # period dropped by OCR
            make_block("Fri.",  515, 260),
            make_block("Sat.",  620, 260),
        ]
        assert _ocr_detect_active_day(blocks) is None

    def test_returns_none_for_no_day_blocks(self, weekly_rank_blocks):
        all_lower = {b["text"].strip().lower() for b in weekly_rank_blocks}
        day = _ocr_detect_active_day(weekly_rank_blocks, all_lower)
        assert day is None

    def test_single_day_tab_resolves(self):
        """A degenerate crop showing exactly one day tab resolves to that day."""
        cases = [
            ("Mon.",  "monday"),
            ("Tues.", "tuesday"),
            ("Wed.",  "wednesday"),
            ("Thur.", "thursday"),
            ("Fri.",  "friday"),
            ("Sat.",  "saturday"),
        ]
        for abbr, expected in cases:
            blocks = [make_block(abbr, 300, 250)]
            result = _ocr_detect_active_day(blocks)
            assert result == expected, f"Expected {expected} for '{abbr}', got {result}"


# ---------------------------------------------------------------------------
# Full classify_from_ocr_text() integration
# ---------------------------------------------------------------------------

class TestClassifyFromOcrText:
    """
    Tests for classify_from_ocr_text() without a PIL image.

    In the stitch-first pipeline a real image is always supplied, enabling
    colour-based tab detection. These tests exercise the OCR text-detection
    layer only (pass 2 OCR markers).  Strength Ranking tab detection (which
    tab is active) requires a real image and is covered by TestRealFixtures.
    """

    def test_classifies_strength_metrics_screen(self, strength_metrics_blocks):
        """Strength Metrics screen is identified; active tab defaults to 'power' without image."""
        category, confidence = classify_from_ocr_text(strength_metrics_blocks)
        assert category == "power"
        assert confidence >= 0.75

    def test_classifies_weekly_rank(self, weekly_rank_blocks):
        category, confidence = classify_from_ocr_text(weekly_rank_blocks)
        assert category == "weekly"
        assert confidence >= 0.75

    def test_daily_tab_bar_without_image_is_unresolved(self, friday_daily_blocks):
        """Without a PIL image the active day cannot be told from a full tab
        bar (it is a colour signal), so classification declines rather than
        guessing. Real day detection is covered by the colour-based fixture
        tests in TestRealFixtures."""
        category, confidence = classify_from_ocr_text(friday_daily_blocks)
        assert category is None
        assert confidence == 0.0

    def test_returns_none_for_empty_blocks(self):
        category, confidence = classify_from_ocr_text([])
        assert category is None
        assert confidence == 0.0

    def test_returns_none_for_unrecognised_layout(self):
        noise_blocks = [
            make_block("Hello", 100, 100),
            make_block("World", 200, 100),
        ]
        category, confidence = classify_from_ocr_text(noise_blocks)
        assert category is None
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# Real fixture tests — auto-discovered from tests/fixtures/ocr_responses/
# ---------------------------------------------------------------------------

# Day/screen keywords mapped to expected output category. Keys are matched
# against the lowercased fixture stem; the **longest matching key** wins so
# specific names (e.g. "siege_weekly") take precedence over generic ones
# ("weekly"). Add new keys as new screens land — the order in this dict is
# only for human readability.
_FILENAME_TO_CATEGORY = {
    # Daily VS day tabs
    "monday":           "monday",
    "tuesday":          "tuesday",
    "wednesday":        "wednesday",
    "thursday":         "thursday",
    "friday":           "friday",
    "saturday":         "saturday",
    # Weekly VS
    "weekly":           "weekly",
    # Strength Ranking
    "power":            "power",
    "strength":         "power",
    "kills":            "kills",
    "donation_daily":   "donation_daily",
    "donation_weekly":  "donation_weekly",
    # Alliance Contribution — category × period
    "mutual_assistance_daily":  "mutual_assistance_daily",
    "mutual_assistance_weekly": "mutual_assistance_weekly",
    "mutual_assistance_season": "mutual_assistance_season",
    "siege_daily":              "siege_daily",
    "siege_weekly":             "siege_weekly",
    "siege_season":             "siege_season",
    "rare_soil_war_daily":      "rare_soil_war_daily",
    "rare_soil_war_weekly":     "rare_soil_war_weekly",
    "rare_soil_war_season":     "rare_soil_war_season",
    "defeat_daily":             "defeat_daily",
    "defeat_weekly":            "defeat_weekly",
    "defeat_season":            "defeat_season",
}


def _infer_category(fixture_name: str):
    """
    Infer expected category from fixture filename, or None if unrecognised.

    Uses **longest-match-wins**: when multiple keys appear in the filename,
    the longest (most specific) one is selected. This keeps generic keys
    like "weekly" usable while making compound keys like "siege_weekly"
    take precedence on filenames that contain both.
    """
    lower = fixture_name.lower()
    matches = [(k, v) for k, v in _FILENAME_TO_CATEGORY.items() if k in lower]
    if not matches:
        return None
    matches.sort(key=lambda kv: -len(kv[0]))
    return matches[0][1]


def _discovered_fixtures():
    """Return list of fixture stem names found on disk, or a placeholder if none."""
    stems = sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))
    return stems if stems else ["__no_fixtures__"]


class TestRealFixtures:
    """
    Classification tests against real Vision API responses.
    Fixtures are auto-discovered from tests/fixtures/ocr_responses/.

    Expected category is inferred from the filename — include a day name or
    screen type in the screenshot filename before capturing:
        Friday-215600.png  → friday
        Power-214600.png   → power
        Weekly-220909.png  → weekly
    """

    @pytest.mark.parametrize("fixture_name", _discovered_fixtures())
    def test_classification_from_real_fixture(self, fixture_name, skip_if_no_fixture):
        skip_if_no_fixture(fixture_name)

        expected = _infer_category(fixture_name)
        if expected is None:
            pytest.skip(
                f"Cannot infer expected category from fixture name '{fixture_name}'. "
                f"Rename the screenshot to include a day or screen type before capturing."
            )

        fixture_data = load_fixture(fixture_name)
        blocks = fixture_data["text_blocks"]

        # Load the original screenshot for colour-based day detection if available
        image = _try_load_source_image(fixture_data.get("source_file", ""))

        # Which day tab is active is a colour signal (the active tab is a
        # desaturated white pill), not a text one — so a day fixture cannot be
        # verified without its screenshot. Skip rather than fall back to a
        # text heuristic, which can only guess (see the de-biased fallback in
        # classifier._ocr_detect_active_day_by_text).
        _DAY_CATEGORIES = {
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        }
        if expected in _DAY_CATEGORIES and image is None:
            pytest.skip(
                f"Day detection needs the source screenshot (colour signal); "
                f"image for '{fixture_name}' not available."
            )

        category, confidence = classify_from_ocr_text(
            blocks, image=image, filename=f"{fixture_name}.png"
        )
        assert category == expected, (
            f"Fixture '{fixture_name}': expected '{expected}', got '{category}' "
            f"(image_available={image is not None})"
        )
        assert confidence > 0.0


class TestDailyRankMisclassificationRegression:
    """
    Regression for archived batch 7038e97 (2026-06-16): a Monday Daily-Rank
    screenshot (the last page, ranks 89–95) was classified as Thursday.

    Two-stage failure, both fixed:
      1. The active tab was identified by *brightness*. The active Monday pill
         carries bold dark glyphs, so its sampled mean V landed within ~0.02 of
         an inactive tab — under min_gap once stitched (0.0235 vs the 0.032 it
         measured standalone) — so colour sampling declared itself inconclusive.
      2. It then fell to the text fallback, which awarded +2 to any day OCR'd
         without a trailing period. Cloud Vision reliably drops the period on
         "Thur", so the fallback picked Thursday on every daily screen.

    The fix switches colour sampling to *saturation* (the active white pill is
    desaturated, S ≈ 0.02, vs warm-grey inactive tabs at S ≈ 0.09 — a ~0.07 gap
    that does not invert) and de-biases the text fallback to None on an
    ambiguous tab bar.
    """

    FIXTURE = "monday_IMG_4358"

    def _load(self, skip_if_no_fixture):
        skip_if_no_fixture(self.FIXTURE)
        data = load_fixture(self.FIXTURE)
        image_path = (
            Path(__file__).parent / "fixtures" / "screenshots" / data["source_file"]
        )
        if not image_path.is_file():
            pytest.skip(f"Source image not committed: {image_path}")
        from app.utils.image_utils import pil_from_bytes
        return data["text_blocks"], pil_from_bytes(image_path.read_bytes())

    def test_classifies_as_monday_not_thursday(self, skip_if_no_fixture):
        blocks, image = self._load(skip_if_no_fixture)
        category, confidence = classify_from_ocr_text(
            blocks, image=image, filename=f"{self.FIXTURE}.png"
        )
        assert category == "monday", f"regressed to {category!r}"
        assert confidence == 0.95  # resolved by colour sampling, not the fallback

    def test_saturation_colour_sampling_picks_monday(self, skip_if_no_fixture):
        blocks, image = self._load(skip_if_no_fixture)
        assert _detect_active_day_by_color(image, blocks) == "monday"

    def test_text_fallback_no_longer_picks_thursday(self, skip_if_no_fixture):
        """The smoking gun: on this real tab bar the old fallback returned
        'thursday'. De-biased, it must decline (None) rather than guess."""
        blocks, _image = self._load(skip_if_no_fixture)
        assert _ocr_detect_active_day(blocks) is None


def _try_load_source_image(source_file: str):
    """
    Attempts to load the original screenshot for colour-based classification.
    Returns None if not found — tests degrade gracefully to text-only mode.
    """
    if not source_file:
        return None

    from pathlib import Path
    from app.utils.image_utils import pil_from_bytes

    search_dirs = [
        Path("tests/fixtures/screenshots"),
        Path.home() / "lastwar-screenshots",
        Path.home() / "Pictures",
        Path.home() / "Downloads",
    ]

    for directory in search_dirs:
        if not directory.is_dir():
            continue
        # Walk subdirectories — lastwar-screenshots is now organised by
        # device/configuration (pixel_10_pro_xl/, pixel_fold_*/) rather
        # than a flat layout.
        candidates = [directory / source_file, *directory.rglob(source_file)]
        for candidate in candidates:
            if candidate.is_file():
                try:
                    img = pil_from_bytes(candidate.read_bytes())
                except Exception:
                    continue
                # Mirror the production stitcher's pre-processing step: if
                # the source image is letterboxed (e.g. Pixel Fold inside-
                # landscape split-screen capture), crop to the detected game
                # window so that bbox coordinates from re-captured fixtures
                # line up with the image the classifier samples colours from.
                from app.utils.window_detect import (
                    crop_to_window,
                    detect_window_by_black_borders,
                )
                rect = detect_window_by_black_borders(img)
                if rect is not None:
                    img = crop_to_window(img, rect)
                return img

    return None
