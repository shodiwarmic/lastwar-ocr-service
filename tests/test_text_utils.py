"""
tests/test_text_utils.py

Unit tests for app/utils/text_utils.py.

Covers:
    - split_name_score_crash  — OCR crash token splitting
    - all_crash_splits        — full candidate enumeration and ordering
    - Existing helpers smoke-checked (clean_player_name, parse_score, etc.
      are exercised more thoroughly via test_extractor.py; only edge cases
      specific to these functions live here)
"""

import pytest

from app.utils.text_utils import (
    all_crash_splits,
    parse_score,
    split_name_score_crash,
)


# ---------------------------------------------------------------------------
# split_name_score_crash
# ---------------------------------------------------------------------------

class TestSplitNameScoreCrash:
    """
    Tests the function that detects and splits OCR tokens where a digit-ending
    player name is rendered flush against the score, e.g. "Ruthless54323,045,000".
    """

    # --- Real crash cases from the mutual assistance weekly screenshot ---

    def test_digit_ending_name_rank2(self):
        result = split_name_score_crash("Ruthless54323,045,000")
        assert result == ("Ruthless5432", "3,045,000")

    def test_digit_ending_name_rank4(self):
        result = split_name_score_crash("CheeseKillers22,622,000")
        assert result == ("CheeseKillers2", "2,622,000")

    def test_letter_ending_name(self):
        result = split_name_score_crash("Splendiddragon2,552,780")
        assert result == ("Splendiddragon", "2,552,780")

    def test_digit_ending_name_short_score(self):
        result = split_name_score_crash("Blindman032,291,620")
        assert result == ("Blindman03", "2,291,620")

    def test_letter_ending_name_short(self):
        result = split_name_score_crash("LailaFa2,483,820")
        assert result == ("LailaFa", "2,483,820")

    # --- Tokens that should NOT be split ---

    def test_pure_numeric_returns_none(self):
        assert split_name_score_crash("3,045,000") is None

    def test_pure_name_returns_none(self):
        assert split_name_score_crash("ShodiWarmic") is None

    def test_name_with_no_score_suffix_returns_none(self):
        # Ends in digits but no comma-grouped number
        assert split_name_score_crash("Charlie9042") is None

    def test_score_without_commas_returns_none(self):
        # Bare integer — is_numeric_token handles this; no comma group to find
        assert split_name_score_crash("Name3045000") is None

    # --- Name-prefix comma exclusion ---

    def test_rejects_split_when_prefix_contains_comma(self):
        # "Alpha3,000X5,000" — the rightmost valid score suffix is "5,000",
        # but that leaves "Alpha3,000X" which contains a comma → rejected.
        # The only comma-free prefix is "Alpha3,000X" → no valid split exists.
        # Verifies the comma-in-prefix guard actually fires.
        result = split_name_score_crash("Alpha3,000X5,000")
        # "X" comes after the comma-group so "Alpha3,000X" still has a comma
        # in the prefix; no valid split exists.
        assert result is None

    def test_pure_digit_comma_string_returns_none(self):
        # "54323,045,000" has no alpha characters so is treated as a pure
        # numeric token — the function does not split it.
        assert split_name_score_crash("54323,045,000") is None

    # --- Edge cases ---

    def test_single_char_name_prefix(self):
        result = split_name_score_crash("X1,000")
        assert result == ("X", "1,000")

    def test_empty_string_returns_none(self):
        assert split_name_score_crash("") is None

    def test_only_commas_and_digits_returns_none(self):
        # No alpha character — treated as pure numeric
        assert split_name_score_crash("1,234,567") is None


# ---------------------------------------------------------------------------
# all_crash_splits
# ---------------------------------------------------------------------------

class TestAllCrashSplits:
    """
    Tests the function that returns ALL valid crash splits ordered smallest
    score first (heuristic at index 0).
    """

    def test_returns_all_valid_splits_for_ruthless(self):
        splits = all_crash_splits("Ruthless54323,045,000")
        assert len(splits) == 3

        names  = [s[0] for s in splits]
        scores = [parse_score(s[1]) for s in splits]

        assert "Ruthless5432" in names
        assert "Ruthless543"  in names
        assert "Ruthless54"   in names

    def test_ordered_ascending_by_score(self):
        splits = all_crash_splits("Ruthless54323,045,000")
        scores = [parse_score(s[1]) for s in splits]
        assert scores == sorted(scores)

    def test_heuristic_is_index_zero(self):
        """Index 0 must match the rightmost-split heuristic (smallest score)."""
        heuristic = split_name_score_crash("Ruthless54323,045,000")
        all_s = all_crash_splits("Ruthless54323,045,000")
        assert all_s[0] == heuristic

    def test_two_splits_cheese_killers(self):
        splits = all_crash_splits("CheeseKillers22,622,000")
        assert len(splits) == 2
        assert splits[0] == ("CheeseKillers2", "2,622,000")
        assert splits[1] == ("CheeseKillers",  "22,622,000")

    def test_one_split_letter_ending_name(self):
        splits = all_crash_splits("Splendiddragon2,552,780")
        assert len(splits) == 1
        assert splits[0] == ("Splendiddragon", "2,552,780")

    def test_pure_numeric_returns_empty(self):
        assert all_crash_splits("3,045,000") == []

    def test_plain_name_returns_empty(self):
        assert all_crash_splits("ShodiWarmic") == []

    def test_no_duplicates(self):
        splits = all_crash_splits("Ruthless54323,045,000")
        assert len(splits) == len(set(splits))

    def test_scores_match_split_name_score_crash(self):
        """Every entry in all_crash_splits should be parseable."""
        for name_prefix, score_str in all_crash_splits("CheeseKillers22,622,000"):
            assert parse_score(score_str) is not None
            assert name_prefix  # non-empty
