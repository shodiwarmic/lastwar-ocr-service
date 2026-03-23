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
from tests.conftest import get_text_blocks, make_block, skip_if_no_fixture


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

    def test_tolerance_scales_with_image_height(self):
        """Lower resolution image should use tighter absolute tolerance."""
        # At 1200px height, tolerance = 1200 * 0.018 = ~21.6px
        blocks = [
            make_block("Player1", 300, 400),
            make_block("Score1",  580, 425),  # 25px apart — should be separate rows at 1200px
        ]
        rows_low_res = build_rows_from_blocks(blocks, image_height=1200)
        rows_hi_res  = build_rows_from_blocks(blocks, image_height=2400)
        # At low res, 25px > tolerance (21.6px) → 2 rows
        assert len(rows_low_res) == 2
        # At high res, 25px < tolerance (43.2px) → 1 row
        assert len(rows_hi_res) == 1


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
        # "[PoWr]Pantheon" — no space between tag and name
        result = clean_player_name("[PoWr]Pantheon of Wrath")
        assert "[PoWr]" not in result
        assert "Pantheon" in result


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
        blocks = [
            # Row 1: SirBucksALot
            make_block("1",             60, 400),
            make_block("[PoWr]",        160, 400),
            make_block("SirBucksALot", 300, 401),
            make_block("45,635,206",   580, 400),
            # Row 2: Crazy Carol
            make_block("2",             60, 490),
            make_block("[PoWr]",        160, 490),
            make_block("Crazy",        270, 491),
            make_block("Carol",        350, 490),
            make_block("33,871,230",   580, 490),
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
# Real fixture extraction tests (skipped if fixtures absent)
# ---------------------------------------------------------------------------

class TestExtractPlayersRealFixtures:
    """
    Verifies extract_players() against known player data from sample screenshots.
    Adjust expected values to match your actual screenshots if they differ.
    """

    def test_extracts_weekly_rank_top_player(self, skip_if_no_fixture):
        skip_if_no_fixture("8851")
        blocks = get_text_blocks("8851")
        players = extract_players(blocks, screen_type="weekly", image_height=2400)
        assert len(players) > 0
        # Top player in weekly rank screenshot
        top = players[0]
        assert top.player_name == "SirBucksALot"
        assert top.score == 161_528_090

    def test_extracts_power_ranking_top_player(self, skip_if_no_fixture):
        skip_if_no_fixture("8725")
        blocks = get_text_blocks("8725")
        players = extract_players(blocks, screen_type="power", image_height=2400)
        assert len(players) > 0
        top = players[0]
        assert top.player_name == "MOJO DUDE"
        assert top.score == 218_478_394

    def test_extracts_self_player_row(self, skip_if_no_fixture):
        """The highlighted self-player row (ShodiWarmic) should also be extracted."""
        skip_if_no_fixture("8836")
        blocks = get_text_blocks("8836")
        players = extract_players(blocks, screen_type="friday", image_height=2400)
        names = [p.player_name for p in players]
        assert "ShodiWarmic" in names
