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
    _is_score_block,
    build_rows_from_blocks,
    extract_players,
    is_valid_player_row,
    parse_player_row,
)
from app.models.schemas import ScoreCandidate
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
# Crash token detection (_is_score_block)
# ---------------------------------------------------------------------------

class TestIsScoreBlock:

    def test_pure_numeric_score(self):
        assert _is_score_block(make_block("3,045,000", 600, 400)) is True

    def test_crash_token_is_score_block(self):
        # Merged name+score must still anchor a row
        assert _is_score_block(make_block("Ruthless54323,045,000", 400, 400)) is True

    def test_plain_name_is_not_score_block(self):
        assert _is_score_block(make_block("ShodiWarmic", 300, 400)) is False

    def test_small_number_filtered_by_min_score(self):
        assert _is_score_block(make_block("42", 60, 400)) is False

    def test_crash_token_below_min_score_is_not_score_block(self):
        # Name + score < min_score should not anchor a row
        assert _is_score_block(make_block("Name999", 400, 400)) is False


# ---------------------------------------------------------------------------
# Crash token row parsing (parse_player_row)
# ---------------------------------------------------------------------------

class TestParsePlayerRowCrash:

    def test_fully_merged_crash_token(self):
        """OCR merges name+score into one block — must still extract both."""
        blocks = [
            make_block("[PoWr]",                  80, 400),
            make_block("Ruthless54323,045,000",  400, 400),
        ]
        result = parse_player_row(blocks, image_width=1080)
        assert result is not None
        raw_name, raw_score = result
        assert "Ruthless5432" in raw_name
        assert raw_score == "3,045,000"

    def test_crash_with_multi_token_name(self):
        """Non-crash name tokens to the left of the crash block are included."""
        blocks = [
            make_block("[PoWr]",           80, 400),
            make_block("Louie",           220, 400),
            make_block("MW2,648,640",     450, 400),
        ]
        result = parse_player_row(blocks, image_width=1080)
        assert result is not None
        raw_name, raw_score = result
        assert "Louie" in raw_name
        assert "MW" in raw_name
        assert raw_score == "2,648,640"

    def test_crash_returns_none_without_score(self):
        """A crash block with no comma-grouped suffix should not extract."""
        blocks = [make_block("Name12345", 400, 400)]
        assert parse_player_row(blocks) is None


# ---------------------------------------------------------------------------
# extract_players — crash candidates and bounds-based correction
# ---------------------------------------------------------------------------

class TestExtractPlayersCrash:
    """
    Tests the two-pass extraction: heuristic split in pass 1, monotonicity
    bounds validation in pass 2, and candidates field on output.
    """

    # -- fixture helpers --

    def _mutual_assistance_weekly_blocks(self):
        """
        Simulates the Mutual Assistance weekly screenshot where rank 2 and 4
        have digit-ending names that crash into their scores.

        Rank 1: Sheffie      3,779,860  (clean)
        Rank 2: Ruthless5432 3,045,000  (crash: Ruthless54323,045,000)
        Rank 3: Louie MW     2,648,640  (clean)
        Rank 4: CheeseKillers2 2,622,000 (crash: CheeseKillers22,622,000)
        """
        return [
            make_block("[PoWr]",                    80,  200),
            make_block("Sheffie",                  300,  200),
            make_block("3,779,860",                700,  200),

            make_block("[PoWr]",                    80,  350),
            make_block("Ruthless54323,045,000",    400,  350),

            make_block("[PoWr]",                    80,  500),
            make_block("Louie",                    280,  500),
            make_block("MW",                       340,  500),
            make_block("2,648,640",                700,  500),

            make_block("[PoWr]",                    80,  650),
            make_block("CheeseKillers22,622,000",  400,  650),
        ]

    # -- candidates field --

    def test_crash_row_has_candidates(self):
        players = extract_players(
            self._mutual_assistance_weekly_blocks(),
            screen_type="mutual_assistance_weekly",
            image_height=2400,
        )
        names = {p.player_name: p for p in players}
        assert "Ruthless5432" in names
        entry = names["Ruthless5432"]
        assert entry.candidates is not None
        assert len(entry.candidates) >= 2

    def test_candidates_are_score_candidates(self):
        players = extract_players(
            self._mutual_assistance_weekly_blocks(),
            screen_type="mutual_assistance_weekly",
            image_height=2400,
        )
        entry = next(p for p in players if p.player_name == "Ruthless5432")
        for c in entry.candidates:
            assert isinstance(c, ScoreCandidate)
            assert c.player_name
            assert c.score > 0

    def test_candidates_ordered_smallest_score_first(self):
        players = extract_players(
            self._mutual_assistance_weekly_blocks(),
            screen_type="mutual_assistance_weekly",
            image_height=2400,
        )
        entry = next(p for p in players if p.player_name == "Ruthless5432")
        scores = [c.score for c in entry.candidates]
        assert scores == sorted(scores)

    def test_heuristic_at_candidates_index_zero(self):
        players = extract_players(
            self._mutual_assistance_weekly_blocks(),
            screen_type="mutual_assistance_weekly",
            image_height=2400,
        )
        entry = next(p for p in players if p.player_name == "Ruthless5432")
        assert entry.candidates[0].player_name == entry.player_name
        assert entry.candidates[0].score       == entry.score

    def test_clean_row_has_no_candidates(self):
        players = extract_players(
            self._mutual_assistance_weekly_blocks(),
            screen_type="mutual_assistance_weekly",
            image_height=2400,
        )
        sheffie = next(p for p in players if p.player_name == "Sheffie")
        assert sheffie.candidates is None

    def test_all_four_players_extracted(self):
        players = extract_players(
            self._mutual_assistance_weekly_blocks(),
            screen_type="mutual_assistance_weekly",
            image_height=2400,
        )
        assert len(players) == 4
        names = {p.player_name for p in players}
        assert names == {"Sheffie", "Ruthless5432", "Louie MW", "CheeseKillers2"}

    # -- bounds-based correction --

    def test_bounds_correction_fixes_wrong_heuristic(self):
        """
        When the heuristic split falls outside the monotonicity window
        the pass-2 bounds check picks the next alternative that fits.

        Crash token "Name23,000" produces two candidates:
            heuristic → ("Name2", 3,000)   — smallest score, rightmost split
            alternative → ("Name",  23,000) — next larger

        With rank 1 = 500,000 and rank 3 = 10,000 the window is [10,000, 500,000].
        3,000 < 10,000 → fails lower bound.
        23,000 ∈ [10,000, 500,000] → accepted.

        After correction the entry should be player_name="Name", score=23,000.
        """
        blocks = [
            make_block("RankOne",    300,  200),
            make_block("500,000",    700,  200),

            make_block("Name23,000", 400,  350),   # crash: heuristic=3,000, correct=23,000

            make_block("RankThree",  300,  500),
            make_block("10,000",     700,  500),
        ]
        players = extract_players(blocks, screen_type="siege_weekly", image_height=2400)
        score_map = {p.player_name: p.score for p in players}

        # Bounds: [10,000, 500,000] — heuristic 3,000 fails; 23,000 fits
        assert score_map.get("Name") == 23_000

    # -- serialisation --

    def test_candidates_excluded_when_none(self):
        players = extract_players(
            self._mutual_assistance_weekly_blocks(),
            screen_type="mutual_assistance_weekly",
            image_height=2400,
        )
        sheffie = next(p for p in players if p.player_name == "Sheffie")
        d = sheffie.model_dump(exclude_none=True)
        assert "candidates" not in d

    def test_candidates_included_in_dump_for_crash_row(self):
        players = extract_players(
            self._mutual_assistance_weekly_blocks(),
            screen_type="mutual_assistance_weekly",
            image_height=2400,
        )
        entry = next(p for p in players if p.player_name == "Ruthless5432")
        d = entry.model_dump(exclude_none=True)
        assert "candidates" in d
        assert isinstance(d["candidates"], list)
        assert d["candidates"][0] == {"player_name": "Ruthless5432", "score": 3_045_000}


# ---------------------------------------------------------------------------
# Kills screen extraction
# ---------------------------------------------------------------------------

class TestExtractKills:

    def test_extracts_expected_player_count(self, kills_ranking_blocks):
        players = extract_players(kills_ranking_blocks, screen_type="kills", image_height=1000)
        assert len(players) == 8

    def test_top_player_correct(self, kills_ranking_blocks):
        players = extract_players(kills_ranking_blocks, screen_type="kills", image_height=1000)
        names = [p.player_name for p in players]
        scores = {p.player_name: p.score for p in players}
        assert "Charlie9042" in names
        assert scores["Charlie9042"] == 17_886_167

    def test_multi_token_name_joined(self, kills_ranking_blocks):
        players = extract_players(kills_ranking_blocks, screen_type="kills", image_height=1000)
        names = [p.player_name for p in players]
        assert "Cloud FF7" in names

    def test_r_badge_stripped(self, kills_ranking_blocks):
        players = extract_players(kills_ranking_blocks, screen_type="kills", image_height=1000)
        for p in players:
            assert not p.player_name.startswith("R"), (
                f"R-badge not stripped from '{p.player_name}'"
            )

    def test_scores_positive(self, kills_ranking_blocks):
        players = extract_players(kills_ranking_blocks, screen_type="kills", image_height=1000)
        for p in players:
            assert p.score > 0


# ---------------------------------------------------------------------------
# Donation Daily extraction
# ---------------------------------------------------------------------------

class TestExtractDonationDaily:

    def test_extracts_expected_player_count(self, donation_daily_blocks):
        players = extract_players(donation_daily_blocks, screen_type="donation_daily", image_height=1000)
        assert len(players) == 8

    def test_top_player_correct(self, donation_daily_blocks):
        players = extract_players(donation_daily_blocks, screen_type="donation_daily", image_height=1000)
        scores = {p.player_name: p.score for p in players}
        assert "BlackIce2" in scores
        assert scores["BlackIce2"] == 14_800

    def test_multi_token_names_joined(self, donation_daily_blocks):
        players = extract_players(donation_daily_blocks, screen_type="donation_daily", image_height=1000)
        names = [p.player_name for p in players]
        assert "Cloud FF7" in names
        assert "Crazy Carol" in names
        assert "Davilson Pirani" in names
        assert "Doc Hollagoon" in names

    def test_scores_above_minimum(self, donation_daily_blocks):
        players = extract_players(donation_daily_blocks, screen_type="donation_daily", image_height=1000)
        for p in players:
            assert p.score >= 1_000


# ---------------------------------------------------------------------------
# Donation Weekly extraction
# ---------------------------------------------------------------------------

class TestExtractDonationWeekly:

    def test_extracts_expected_player_count(self, donation_weekly_blocks):
        players = extract_players(donation_weekly_blocks, screen_type="donation_weekly", image_height=1000)
        assert len(players) == 8

    def test_top_player_correct(self, donation_weekly_blocks):
        players = extract_players(donation_weekly_blocks, screen_type="donation_weekly", image_height=1000)
        scores = {p.player_name: p.score for p in players}
        assert "CaptTrickster727" in scores
        assert scores["CaptTrickster727"] == 28_300

    def test_multi_token_names_joined(self, donation_weekly_blocks):
        players = extract_players(donation_weekly_blocks, screen_type="donation_weekly", image_height=1000)
        names = [p.player_name for p in players]
        assert "Crazy Carol" in names
        assert "Davilson Pirani" in names
        assert "Cloud FF7" in names

    def test_scores_above_minimum(self, donation_weekly_blocks):
        players = extract_players(donation_weekly_blocks, screen_type="donation_weekly", image_height=1000)
        for p in players:
            assert p.score >= 1_000


# ---------------------------------------------------------------------------
# Real fixture extraction tests — auto-discovered
# ---------------------------------------------------------------------------

_FILENAME_TO_CATEGORY = {
    "monday":                    "monday",
    "tuesday":                   "tuesday",
    "wednesday":                 "wednesday",
    "thursday":                  "thursday",
    "friday":                    "friday",
    "saturday":                  "saturday",
    "weekly":                    "weekly",
    "power":                     "power",
    "strength":                  "power",
    "kills":                     "kills",
    "donation_daily":            "donation_daily",
    "donation_weekly":           "donation_weekly",
    # Season Contribution categories
    "mutual_assistance_daily":   "mutual_assistance_daily",
    "mutual_assistance_weekly":  "mutual_assistance_weekly",
    "mutual_assistance_season":  "mutual_assistance_season",
    "siege_daily":               "siege_daily",
    "siege_weekly":              "siege_weekly",
    "siege_season":              "siege_season",
    "rare_soil_war_daily":       "rare_soil_war_daily",
    "rare_soil_war_weekly":      "rare_soil_war_weekly",
    "rare_soil_war_season":      "rare_soil_war_season",
    "defeat_daily":              "defeat_daily",
    "defeat_weekly":             "defeat_weekly",
    "defeat_season":             "defeat_season",
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
