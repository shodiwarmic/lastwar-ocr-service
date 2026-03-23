"""
tests/test_extractor.py

Unit tests for app/pipeline/extractor.py.

Tests cover:
    - Row clustering with Y-coordinate tolerance
    - Name/score parsing from a single row
    - Full clean_player_name pipeline (via text_utils)
    - Validation filtering (UI labels, min score, empty names)
    - End-to-end extract_players() using synthetic blocks
    - End-to-end extract_players() using real fixtures (skipped if absent)
"""

import pytest

from app.pipeline.extractor import (
    build_rows_from_blocks,
    extract_players,
    is_valid_player_row,
    parse_player_row,
)
from app.utils.text_utils import clean_player_name
from tests.conftest import FIXTURE_DIR, get_text_blocks, load_fixture, make_block, skip_if_no_fixture


# ---------------------------------------------------------------------------
# Row clustering
# ---------------------------------------------------------------------------

class TestBuildRowsFromBlocks:

    def test_groups_same_y_into_one_row(self):
        blocks = [
            make_block("1",            60, 400),
            make_block("SirBucksALot", 300, 402),   # within tolerance
            make_block("45,635,206",   580, 400),
        ]
        rows = build_rows_from_blocks(blocks, image_height=2400)
        assert len(rows) == 1
        assert len(rows[0]) == 3

    def test_separates_distant_y_into_separate_rows(self):
        blocks = [
            make_block("SirBucksALot", 300, 400),
            make_block("45,635,206",   580, 400),
            make_block("Crazy Carol",  300, 500),   # >43px apart at 2400px height
            make_block("33,871,230",   580, 500),
        ]
        rows = build_rows_from_blocks(blocks, image_height=2400)
        assert len(rows) == 2

    def test_row_sorted_left_to_right(self):
        blocks = [
            make_block("45,635,206",   580, 400),   # score first in input
            make_block("SirBucksALot", 300, 400),   # name second
        ]
        rows = build_rows_from_blocks(blocks, image_height=2400)
        assert rows[0][0]["text"] == "SirBucksALot"
        assert rows[0][1]["text"] == "45,635,206"

    def test_empty_input_returns_empty(self):
        assert build_rows_from_blocks([], image_height=2400) == []

    def test_score_anchored_clustering(self):
        """Score-anchored approach groups name tokens above the score."""
        blocks = [
            make_block("SirBucksALot", 300, 400),   # name — above score
            make_block("45,635,206",   580, 430),   # score — 30px below name
            make_block("Pantheon",     300, 460),   # alliance — 30px below score
        ]
        rows = build_rows_from_blocks(blocks, image_height=2400)
        # Should find 1 row (1 score anchor)
        assert len(rows) == 1
        # Row should contain name + score but NOT alliance (below score)
        texts = [b["text"] for b in rows[0]]
        assert "SirBucksALot" in texts
        assert "45,635,206" in texts
        assert "Pantheon" not in texts


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------

class TestParsePlayerRow:

    def test_extracts_name_and_score(self, player_row_blocks):
        result = parse_player_row(player_row_blocks)
        assert result is not None
        raw_name, raw_score = result
        assert "SirBucksALot" in raw_name
        assert raw_score == "45,635,206"

    def test_returns_none_for_row_without_score(self):
        blocks = [
            make_block("SirBucksALot", 300, 400),
            make_block("[PoWr]",        160, 400),
        ]
        assert parse_player_row(blocks) is None

    def test_handles_multi_word_name(self):
        blocks = [
            make_block("gabriel",     200, 400),
            make_block("garage",      310, 400),
            make_block("19,714,200",  580, 400),
        ]
        result = parse_player_row(blocks)
        assert result is not None
        raw_name, _ = result
        assert "gabriel" in raw_name
        assert "garage" in raw_name

    def test_returns_none_for_empty_row(self):
        assert parse_player_row([]) is None

    def test_leftward_constraint_excludes_right_side_tokens(self):
        """
        Tokens to the right of the score should be excluded from the name.
        This prevents alliance display names from bleeding into player names
        when OCR places them on the same row after the score.
        """
        blocks = [
            make_block("[PoWr]",           60, 400),
            make_block("SirBucksALot",    200, 400),
            make_block("45,635,206",       500, 400),
            # These appear to the right of the score — should be excluded
            make_block("Pantheon",         620, 400),
            make_block("of",               690, 400),
            make_block("Wrath",            720, 400),
        ]
        result = parse_player_row(blocks, image_width=1080)
        assert result is not None
        raw_name, raw_score = result
        assert "Pantheon" not in raw_name
        assert "Wrath" not in raw_name
        assert "SirBucksALot" in raw_name


# ---------------------------------------------------------------------------
# Name cleaning
# ---------------------------------------------------------------------------

class TestCleanPlayerName:

    def test_strips_alliance_tag(self):
        assert clean_player_name("[PoWr] SirBucksALot") == "SirBucksALot"

    def test_strips_leading_rank_number(self):
        assert clean_player_name("48 ShodiWarmic") == "ShodiWarmic"

    def test_strips_r_badge(self):
        assert clean_player_name("R4 ShodiWarmic") == "ShodiWarmic"

    def test_handles_combined_noise(self):
        assert clean_player_name("48 R4 [PoWr] ShodiWarmic") == "ShodiWarmic"

    def test_preserves_numbers_in_names(self):
        assert clean_player_name("Charlie9042") == "Charlie9042"

    def test_preserves_spaces_in_names(self):
        assert clean_player_name("[PoWr] gabriel garage") == "gabriel garage"

    def test_handles_tag_without_space(self):
        # "[PoWr]Pantheon" — bracket tag stripped, display name also stripped
        result = clean_player_name("[PoWr]Pantheon of Wrath")
        assert "[PoWr]" not in result

    def test_strips_alliance_display_name(self):
        # Alliance display name appears as plain text after the bracket tag is stripped
        assert clean_player_name("[PoWr] SirBucksALot Pantheon of Wrath") == "SirBucksALot"

    def test_strips_alliance_display_name_case_insensitive(self):
        assert clean_player_name("SirBucksALot pantheon of wrath") == "SirBucksALot"

    def test_strips_thai_characters(self):
        # Thai OCR noise from rank badge icons on Strength Ranking screen
        result = clean_player_name("รๆ3 ShodiWarmic")
        assert "ShodiWarmic" in result
        assert "ร" not in result

    def test_preserves_accented_characters(self):
        # Accented names like Pàcha must survive cleaning
        assert clean_player_name("Pàcha") == "Pàcha"

    def test_full_noisy_row(self):
        # Simulate a full raw string as it arrives from OCR in the real pipeline
        result = clean_player_name("1 R4 [PoWr] SirBucksALot Pantheon of Wrath")
        assert result == "SirBucksALot"

    def test_strips_bare_tag_without_brackets(self):
        # OCR sometimes returns PoWr without brackets as a standalone token
        assert clean_player_name("PoWr SirBucksALot") == "SirBucksALot"

    def test_strips_bare_tag_combined_with_suffix(self):
        # Full production scenario: bracket stripped leaving bare tag + suffix
        result = clean_player_name("PoWr SirBucksALot Pantheon of Wrath")
        assert result == "SirBucksALot"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestIsValidPlayerRow:

    def test_valid_row_passes(self):
        assert is_valid_player_row("SirBucksALot", 45_635_206) is True

    def test_empty_name_fails(self):
        assert is_valid_player_row("", 45_635_206) is False

    def test_none_score_fails(self):
        assert is_valid_player_row("SirBucksALot", None) is False

    def test_zero_score_fails(self):
        assert is_valid_player_row("SirBucksALot", 0) is False

    def test_score_below_minimum_fails(self):
        # Rank numbers (1-100) should be filtered
        assert is_valid_player_row("SirBucksALot", 1) is False

    def test_ui_label_name_fails(self):
        assert is_valid_player_row("Commander", 45_635_206) is False
        assert is_valid_player_row("Ranking", 45_635_206) is False
        assert is_valid_player_row("Points", 45_635_206) is False


# ---------------------------------------------------------------------------
# Full extraction pipeline (synthetic)
# ---------------------------------------------------------------------------

class TestExtractPlayersSynthetic:

    def test_extracts_players_from_multiple_rows(self):
        # Score-anchored: name must be ABOVE the score (lower Y value)
        # Alliance subtitle is BELOW the score and should be excluded
        blocks = [
            # Row 1: SirBucksALot — name at Y=400, score at Y=430
            make_block("SirBucksALot", 300, 400),
            make_block("45,635,206",   580, 430),
            # Row 2: Crazy Carol — name at Y=530, score at Y=560
            make_block("Crazy",        270, 530),
            make_block("Carol",        350, 530),
            make_block("33,871,230",   580, 560),
        ]
        players = extract_players(blocks, screen_type="friday", image_height=2400)
        assert len(players) == 2
        names = [p.player_name for p in players]
        assert "SirBucksALot" in names
        assert "Crazy Carol" in names

    def test_filters_ui_header_row(self):
        blocks = [
            make_block("Ranking",   80, 340),
            make_block("Commander", 300, 340),
            make_block("Points",    550, 340),
            make_block("SirBucksALot", 300, 400),
            make_block("45,635,206",   580, 400),
        ]
        players = extract_players(blocks, screen_type="friday", image_height=2400)
        assert len(players) == 1
        assert players[0].player_name == "SirBucksALot"

    def test_returns_empty_for_no_valid_rows(self):
        blocks = [make_block("Ranking", 80, 340), make_block("Commander", 300, 340)]
        players = extract_players(blocks, screen_type="friday", image_height=2400)
        assert players == []

    def test_score_is_integer(self):
        blocks = [
            make_block("SirBucksALot", 300, 400),
            make_block("45,635,206",   580, 400),
        ]
        players = extract_players(blocks, screen_type="friday", image_height=2400)
        assert len(players) == 1
        assert isinstance(players[0].score, int)
        assert players[0].score == 45_635_206



# ---------------------------------------------------------------------------
# Real fixture extraction tests — auto-discovered
# ---------------------------------------------------------------------------

_FILENAME_TO_CATEGORY = {
    "monday":    "monday",
    "tuesday":   "tuesday",
    "wednesday": "wednesday",
    "thursday":  "thursday",
    "friday":    "friday",
    "saturday":  "saturday",
    "weekly":    "weekly",
    "power":     "power",
    "strength":  "power",
}


def _infer_category(fixture_name: str):
    lower = fixture_name.lower()
    for keyword, category in _FILENAME_TO_CATEGORY.items():
        if keyword in lower:
            return category
    return None


def _discovered_fixtures():
    stems = sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))
    return stems if stems else ["__no_fixtures__"]


class TestExtractPlayersRealFixtures:
    """
    Verifies extract_players() returns at least one valid player from every
    captured fixture. Category is inferred from the fixture filename.

    These are smoke tests — they confirm the extractor produces output
    rather than asserting specific player names and scores (which vary
    per screenshot). Add specific assertions once you know the expected
    values from your fixtures.
    """

    @pytest.mark.parametrize("fixture_name", _discovered_fixtures())
    def test_extracts_at_least_one_player(self, fixture_name, skip_if_no_fixture):
        skip_if_no_fixture(fixture_name)

        screen_type = _infer_category(fixture_name)
        if screen_type is None:
            pytest.skip(
                f"Cannot infer screen type from fixture name '{fixture_name}'. "
                f"Rename the screenshot to include a day or screen type."
            )

        blocks = get_text_blocks(fixture_name)
        fixture_data = load_fixture(fixture_name)
        image_height = fixture_data.get("image_height", 2400)

        players = extract_players(blocks, screen_type=screen_type, image_height=image_height)

        assert len(players) > 0, (
            f"Fixture '{fixture_name}': expected at least one player, got none. "
            f"Check the OCR output in tests/fixtures/ocr_responses/{fixture_name}.json"
        )
        for p in players:
            assert p.player_name.strip(), f"Empty player name in fixture '{fixture_name}'"
            assert p.score > 0, f"Non-positive score in fixture '{fixture_name}'"
