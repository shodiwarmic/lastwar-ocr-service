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

    def test_detects_friday_from_day_tabs(self, friday_daily_blocks):
        all_lower = {b["text"].strip().lower() for b in friday_daily_blocks}
        day = _ocr_detect_active_day(friday_daily_blocks, all_lower)
        assert day == "friday"

    def test_returns_none_for_no_day_blocks(self, weekly_rank_blocks):
        all_lower = {b["text"].strip().lower() for b in weekly_rank_blocks}
        day = _ocr_detect_active_day(weekly_rank_blocks, all_lower)
        assert day is None

    def test_various_day_abbreviations(self):
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
            all_lower = {abbr.lower()}
            result = _ocr_detect_active_day(blocks, all_lower)
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

    def test_classifies_friday_daily(self, friday_daily_blocks):
        category, confidence = classify_from_ocr_text(friday_daily_blocks)
        assert category == "friday"
        assert confidence >= 0.75

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

# Day/screen keywords mapped to expected output category
_FILENAME_TO_CATEGORY = {
    "monday":           "monday",
    "tuesday":          "tuesday",
    "wednesday":        "wednesday",
    "thursday":         "thursday",
    "friday":           "friday",
    "saturday":         "saturday",
    "weekly":           "weekly",
    "power":            "power",
    "strength":         "power",
    "kills":            "kills",
    "donation_daily":   "donation_daily",
    "donation_weekly":  "donation_weekly",
}


def _infer_category(fixture_name: str):
    """Infer expected category from fixture filename, or None if unrecognised."""
    lower = fixture_name.lower()
    for keyword, category in _FILENAME_TO_CATEGORY.items():
        if keyword in lower:
            return category
    return None


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

        category, confidence = classify_from_ocr_text(
            blocks, image=image, filename=f"{fixture_name}.png"
        )
        assert category == expected, (
            f"Fixture '{fixture_name}': expected '{expected}', got '{category}' "
            f"(image_available={image is not None})"
        )
        assert confidence > 0.0


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
                    return pil_from_bytes(candidate.read_bytes())
                except Exception:
                    pass

    return None
