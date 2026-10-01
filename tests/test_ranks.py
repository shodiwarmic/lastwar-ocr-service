"""
tests/test_ranks.py

The rank column as a row-completeness checksum (app/pipeline/ranks.py), and
the guarantee that reading ranks changed nothing else: every recording's
(player_name, score) rows are compared with a snapshot taken before ranks
existed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.pipeline.extractor import extract_players
from app.pipeline.ranks import apply_checksum, as_rank, rank_from_name_token
from tests.conftest import PRIVATE_FIXTURES, discover_fixtures, load_fixture, make_block
from tests.test_extractor import _infer_category

COLUMN = (0.05, 0.26)  # the ranking screens' rank column


def _rows(*ranks, scores=None):
    scores = scores or [1000 * (100 - i) for i in range(len(ranks))]
    return [
        {"rank": r, "rank_source": "column" if r is not None else None, "score": s}
        for r, s in zip(ranks, scores)
    ]


class TestChecksum:

    def test_clean_run(self):
        out = apply_checksum(_rows(8, 9, 10, 11))
        assert out["ranks"] == {"read": 4, "from_name": 0, "inferred": 0, "expected_start": 8,
                                "gaps": [], "duplicates": [], "out_of_order": []}
        assert out["order_violations"] == []

    def test_unread_between_two_read_is_inferred(self):
        rows = _rows(3, None, 5)
        out = apply_checksum(rows)
        assert rows[1]["rank"] == 4 and rows[1]["rank_source"] == "inferred"
        assert out["ranks"]["inferred"] == 1

    def test_unread_at_the_edge_is_inferred_from_one_side(self):
        rows = _rows(None, 5, 6)  # a medal rank OCR could not read
        apply_checksum(rows)
        assert rows[0]["rank"] == 4

    def test_neighbours_that_disagree_infer_nothing(self):
        rows = _rows(3, None, 9, 10)  # a row is missing somewhere
        out = apply_checksum(rows)
        assert rows[1]["rank"] is None
        assert out["ranks"]["gaps"] == [4, 5, 6, 7, 8]

    def test_a_misread_rank_is_reported_not_repaired(self):
        rows = _rows(8, 9, 10, 10, 12, 13)  # 11 read as 10
        out = apply_checksum(rows)
        assert [r["rank"] for r in rows] == [8, 9, 10, 10, 12, 13]
        assert out["ranks"]["duplicates"] == [10]
        assert out["ranks"]["gaps"] == [11]

    def test_out_of_order(self):
        out = apply_checksum(_rows(43, 33, 45, 46))
        assert out["ranks"]["out_of_order"] == [33]

    def test_pinned_own_row_is_set_apart(self):
        rows = _rows(1, 2, 3, None, 5, 23, scores=[90, 80, 70, 60, 50, 99])
        out = apply_checksum(rows)
        assert out["ranks"]["pinned"] == 23
        assert out["ranks"]["gaps"] == []
        assert rows[3]["rank"] == 4
        assert out["order_violations"] == []  # the pinned row's score is its own

    def test_no_ranks_read(self):
        out = apply_checksum(_rows(None, None))
        assert out["ranks"] is None

    def test_order_violation_ties_allowed(self):
        out = apply_checksum(_rows(1, 2, 3, scores=[500, 500, 600]))
        assert [v["index"] for v in out["order_violations"]] == [2]


class TestRankTokens:

    @pytest.mark.parametrize("text,expected", [
        ("1", 1), ("48", 48), ("999", 999), ("12.", 12),
        ("0", None), ("1000", None), ("B", None), ("1,234", None), ("4.6K", None),
    ])
    def test_as_rank(self, text, expected):
        assert as_rank(text) == expected

    def test_merged_into_the_name_inside_the_column(self):
        block = make_block("48KeldaVornic", 200, 500)  # left edge 180 px of 1080
        assert rank_from_name_token(block, COLUMN, 1080) == 48

    def test_a_name_that_starts_with_digits_is_left_alone(self):
        block = make_block("100Max", 420, 500)  # starts in the name column
        assert rank_from_name_token(block, COLUMN, 1080) is None

    def test_a_crash_token_is_not_a_rank(self):
        block = make_block("54323,045,000", 200, 500)
        assert rank_from_name_token(block, COLUMN, 1080) is None


def _row(rank_text, rank_x, name, score, y, rank_dy=0):
    blocks = [make_block(name, 450, y), make_block(score, 850, y)]
    if rank_text is not None:
        blocks.append(make_block(rank_text, rank_x, y + rank_dy))
    return blocks


class TestExtractionWithRanks:
    """The regression matrix: ranks read beside names that tempt a misread."""

    def _extract(self, blocks, report=None):
        blocks = sorted(blocks, key=lambda b: (b["avg_y"], b["avg_x"]))
        return extract_players(blocks, screen_type="weekly", image_height=2400,
                               image_width=1080, report=report)

    def test_digit_leading_and_digit_trailing_names(self):
        blocks = (_row("1", 160, "100Max", "9,000,000", 600)
                  + _row("2", 160, "Agent47", "8,000,000", 800)
                  + _row("3", 160, "Victor9042", "7,000,000", 1000))
        players = self._extract(blocks)
        assert [(p.rank, p.score) for p in players] == [(1, 9_000_000), (2, 8_000_000), (3, 7_000_000)]

    def test_rank_below_the_score_line_is_still_found(self):
        # Strength and Alliance Contribution draw the rank ~0.010 H lower.
        blocks = _row("7", 160, "Pollie", "5,000,000", 600, rank_dy=24)
        assert self._extract(blocks)[0].rank == 7

    def test_duplicate_rank_tokens_take_the_nearest(self):
        blocks = _row("4", 160, "Pollie", "5,000,000", 600) + [make_block("5", 160, 620)]
        assert self._extract(blocks)[0].rank == 4

    def test_token_on_the_column_boundary(self):
        inside = _row("9", int(0.26 * 1080), "Pollie", "5,000,000", 600)
        outside = _row("9", int(0.27 * 1080), "Pollie", "5,000,000", 600)
        assert self._extract(inside)[0].rank == 9
        assert self._extract(outside)[0].rank is None

    def test_merged_rank_is_read_from_the_name(self):
        blocks = [make_block("48KeldaVornic", 200, 600), make_block("5,000,000", 850, 600)]
        players = self._extract(blocks)
        assert players[0].rank == 48
        assert players[0].player_name == "KeldaVornic"

    def test_unread_rank_inferred_and_marked(self):
        blocks = (_row("1", 160, "Pollie", "9,000,000", 600)
                  + _row("B", 160, "Agent47", "8,000,000", 800)
                  + _row("3", 160, "Victor9042", "7,000,000", 1000))
        report = {}
        players = self._extract(blocks, report)
        assert [(p.rank, p.rank_inferred) for p in players] == [(1, None), (2, True), (3, None)]
        assert report["ranks"]["inferred"] == 1


def _snapshots():
    out = {}
    public = Path(__file__).parent / "fixtures" / "extraction_snapshot.json"
    out.update(json.loads(public.read_text(encoding="utf-8")))
    if PRIVATE_FIXTURES:
        private = PRIVATE_FIXTURES / "ocr-service" / "extraction_snapshot.json"
        if private.is_file():
            out.update(json.loads(private.read_text(encoding="utf-8")))
    return out


_SNAPSHOTS = _snapshots()


@pytest.mark.parametrize("fixture_name", discover_fixtures())
def test_names_and_scores_unchanged(fixture_name):
    """Every recording extracts exactly the (name, score, candidates) rows it
    did before ranks were read (snapshot taken at the private-fixtures move)."""
    if fixture_name not in _SNAPSHOTS:
        pytest.skip(f"no snapshot for {fixture_name}")
    expected = _SNAPSHOTS[fixture_name]
    data = load_fixture(fixture_name)
    players = extract_players(data["text_blocks"], screen_type=expected["category"],
                              image_height=data["image_height"], image_width=data["image_width"])
    rows = [[p.player_name, p.score] + ([[[c.player_name, c.score] for c in p.candidates]] if p.candidates else [])
            for p in players]
    assert rows == expected["rows"]


def test_ranks_reach_the_response(client):
    """`rank` on each row and the checksum on the section, end to end."""
    from unittest.mock import MagicMock, patch
    from tests.test_routes import png_file_storage
    import app.routes as routes_module

    blocks = (_row("1", 160, "Pollie", "9,000,000", 600)
              + _row("B", 160, "Agent47", "8,000,000", 800)
              + _row("3", 160, "Victor9042", "7,000,000", 1000))
    blocks.sort(key=lambda b: (b["avg_y"], b["avg_x"]))
    routes_module._result_cache.clear()
    with patch("app.routes.run_ocr", return_value=(MagicMock(), "h")), \
         patch("app.routes.extract_text_blocks", return_value=blocks), \
         patch("app.routes.prepare_stitched_batches") as stitch:
        from PIL import Image
        from app.pipeline.stitcher import ImageRegion
        stitch.return_value = [(Image.new("RGB", (1080, 2400)), [ImageRegion("f.png", 0, 2400)])]
        body = client.post("/process-batch", content_type="multipart/form-data",
                           data={"images": png_file_storage("f.png"), "category": "weekly"}).get_json()
    routes_module._result_cache.clear()
    rows = body["results"]["weekly"]
    assert [(r.get("rank"), r.get("rank_inferred")) for r in rows] == [(1, None), (2, True), (3, None)]
    assert body["diagnostics"]["sections"][0]["ranks"]["inferred"] == 1
