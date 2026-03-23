"""
tests/test_classifier.py

Unit tests for app/pipeline/classifier.py.

All tests use either synthetic OCR blocks (from conftest fixtures) or
real JSON fixtures loaded from tests/fixtures/ocr_responses/.

The Vision API is never called. classify_from_ocr_text() is tested directly
with pre-built block lists that match the format ocr_client produces.

Test categories:
    - OCR-based classification (Pass 2) using synthetic blocks
    - OCR-based classification using real captured fixtures (skipped if absent)
    - Day label normalisation edge cases
    - Confidence threshold behaviour
"""

import pytest

from app.pipeline.classifier import (
    CONFIDENCE_THRESHOLD,
    classify_from_ocr_text,
    _ocr_detect_strength,
    _ocr_detect_weekly,
    _ocr_detect_active_day,
)
from tests.conftest import get_text_blocks, make_block


# ---------------------------------------------------------------------------
# Pass 2: Strength Ranking detection
# ---------------------------------------------------------------------------

class TestOcrDetectStrength:

    def test_detects_strength_ranking_header(self, strength_ranking_blocks):
        all_lower = {b["text"].strip().lower() for b in strength_ranking_blocks}
        assert _ocr_detect_strength(all_lower) is True

    def test_detects_via_power_kills_donation(self):
        blocks = {"power", "kills", "donation", "ranking", "commander"}
        assert _ocr_detect_strength(blocks) is True

    def test_requires_both_power_and_kills(self):
        # power alone should not match
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

    def test_does_not_detect_strength_as_weekly(self, strength_ranking_blocks):
        all_lower = {b["text"].strip().lower() for b in strength_ranking_blocks}
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
        """Each day abbreviation variant should resolve correctly."""
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
            assert result == expected, f"Expected {expected} for abbreviation '{abbr}', got {result}"


# ---------------------------------------------------------------------------
# Full classify_from_ocr_text() integration
# ---------------------------------------------------------------------------

class TestClassifyFromOcrText:

    def test_classifies_strength_ranking(self, strength_ranking_blocks):
        category, confidence = classify_from_ocr_text(strength_ranking_blocks)
        assert category == "power"
        assert confidence == 1.0

    def test_classifies_weekly_rank(self, weekly_rank_blocks):
        category, confidence = classify_from_ocr_text(weekly_rank_blocks)
        assert category == "weekly"
        assert confidence == 1.0

    def test_classifies_friday_daily(self, friday_daily_blocks):
        category, confidence = classify_from_ocr_text(friday_daily_blocks)
        assert category == "friday"
        assert confidence == 1.0

    def test_returns_none_for_empty_blocks(self):
        category, confidence = classify_from_ocr_text([])
        assert category is None
        assert confidence == 0.0

    def test_returns_none_for_unrecognised_layout(self):
        noise_blocks = [
            make_block("Hello",  100, 100),
            make_block("World",  200, 100),
        ]
        category, confidence = classify_from_ocr_text(noise_blocks)
        assert category is None
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# Real fixture tests (skipped if fixtures not yet captured)
# ---------------------------------------------------------------------------

class TestRealFixtures:
    """
    Tests against real Vision API responses captured from the sample screenshots.
    Run tools/capture_ocr_fixture.py first to generate the fixture files.
    """

    FIXTURE_EXPECTED = [
        ("8851", "weekly"),    # Weekly Rank tab active
        ("8836", "friday"),    # Friday Daily tab active
        ("8822", "thursday"),  # Thursday Daily tab active
        ("8803", "thursday"),  # Thursday Daily tab active (duplicate screenshot)
        ("8778", "monday"),    # Monday Daily tab active
        ("8763", "tuesday"),   # Tuesday Daily tab active
        ("8754", "thursday"),  # Thursday (different week)
        ("8738", "wednesday"), # Wednesday Daily tab active
        ("8725", "power"),     # Strength Ranking screen
        ("8722", "tuesday"),   # Tuesday Daily tab active
    ]

    @pytest.mark.parametrize("fixture_name,expected_category", FIXTURE_EXPECTED)
    def test_classification_from_real_fixture(
        self, fixture_name, expected_category, skip_if_no_fixture
    ):
        skip_if_no_fixture(fixture_name)
        blocks = get_text_blocks(fixture_name)
        category, confidence = classify_from_ocr_text(blocks, filename=f"{fixture_name}.png")
        assert category == expected_category, (
            f"Fixture {fixture_name}: expected '{expected_category}', got '{category}'"
        )
        assert confidence == 1.0
