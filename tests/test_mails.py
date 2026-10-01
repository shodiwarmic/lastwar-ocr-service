"""
tests/test_mails.py

The three post-event mails (Alliance Exercise, Zombie Siege, Desert Storm),
read with the column_scoped strategy (app/pipeline/column_scoped.py).

Recordings are named `<category>__<device>__<frame>`. The full set (58 frames,
two devices) is private and needs LASTWAR_FIXTURES; one scrubbed frame per mail
is public. Synthetic tests cover each known trap by name.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.models.schemas import VALID_CATEGORIES
from app.pipeline.column_scoped import parse_suffixed
from app.pipeline.extractor import extract_players
from app.pipeline.screen_definitions import all_categories, get_definition_for_category
from app.pipeline.stitcher import ImageRegion
from tests.conftest import discover_fixtures, fixture_path, load_fixture, make_block
from tests.test_routes import png_file_storage

MAILS = ("alliance_exercise", "zombie_siege", "desert_storm")

# The categories before any mail existed, written out so the derivation is
# checked against a list nobody derived.
RANKING_CATEGORIES = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "weekly", "power", "kills", "donation_daily", "donation_weekly",
    "mutual_assistance_daily", "mutual_assistance_weekly", "mutual_assistance_season",
    "siege_daily", "siege_weekly", "siege_season",
    "rare_soil_war_daily", "rare_soil_war_weekly", "rare_soil_war_season",
    "defeat_daily", "defeat_weekly", "defeat_season",
}


# ---------------------------------------------------------------------------
# Categories come from the definitions
# ---------------------------------------------------------------------------

class TestCategories:

    def test_ranking_screens_derive_exactly_the_old_23(self):
        ranking = {c for c in all_categories() if c not in MAILS}
        assert ranking == RANKING_CATEGORIES
        assert len(RANKING_CATEGORIES) == 23

    def test_every_mail_is_a_category(self):
        assert VALID_CATEGORIES == RANKING_CATEGORIES | set(MAILS)

    def test_alliance_contribution_keys_find_their_definition(self):
        # Joined keys matched no tab item before, and fell back to defaults.
        assert get_definition_for_category("siege_daily").id == "alliance_contribution"
        assert get_definition_for_category("defeat_season").id == "alliance_contribution"

    @pytest.mark.parametrize("category", MAILS)
    def test_mail_definitions(self, category):
        defn = get_definition_for_category(category)
        assert defn.id == f"mail_{category}"
        assert defn.row_clustering.strategy == "column_scoped"
        assert defn.row_clustering.min_score == 0

    def test_health_lists_the_mails(self, client):
        assert set(MAILS) <= set(client.get("/health").get_json()["categories"])


# ---------------------------------------------------------------------------
# Score formats
# ---------------------------------------------------------------------------

class TestSuffixedScores:

    @pytest.mark.parametrize("text,expected", [
        ("23.36G", 23_360_000_000),
        ("35.51G", 35_510_000_000),
        ("9.33M", 9_330_000),
        ("60.0K", 60_000),
        ("1.5B", 1_500_000_000),
        ("4.466", 4_460_000_000),   # Cloud Vision reads G as 6
        ("60.OK", 60_000),          # and 0 as O
        ("0", 0),
    ])
    def test_parses(self, text, expected):
        assert parse_suffixed(text) == expected

    # A value whose suffix was lost has lost its magnitude: "5.20G" has come
    # back from Cloud Vision as "5206".
    @pytest.mark.parametrize("text", ["23.36", "G", "4.6.6G", "Damage", "1.3946", "5206", "123"])
    def test_rejects(self, text):
        assert parse_suffixed(text) is None


# ---------------------------------------------------------------------------
# Synthetic frames: each trap by name
# ---------------------------------------------------------------------------

W, H = 1080, 2404


def _ae_row(rank, name, damage, y):
    """An Alliance Exercise row as Cloud Vision returns it: rank ~40 px below
    the name, "Total" "Damage" ":" value on a line ~60 px below."""
    blocks = [make_block(name, 500, y)]
    if rank is not None:
        blocks.append(make_block(rank, 170, y + 40))
    if damage is not None:
        blocks += [make_block("Total", 420, y + 60), make_block("Damage", 520, y + 60),
                   make_block(":", 585, y + 60), make_block(damage, 650, y + 60)]
    return blocks


def _ae_frame(rows, *, card_y=None, header_y=1072, extra=()):
    blocks = [make_block("Alliance", 300, 358), make_block("Reward", 800, 358)]
    if card_y is not None:  # the MVP card above the header
        blocks += [make_block("MVP", 540, card_y), make_block("[PoWr]Bang", 700, card_y + 104),
                   make_block("279,211,843", 680, card_y + 167),
                   make_block("Attacks:", 420, card_y + 265), make_block("16", 505, card_y + 265),
                   make_block("Total", 400, card_y + 318), make_block("Damage", 500, card_y + 318),
                   make_block(":", 565, card_y + 318), make_block("35.51G", 650, card_y + 318)]
    blocks += [make_block("Damage", 450, header_y), make_block("Ranking", 560, header_y),
               make_block("2-20", 670, header_y)]
    for r in rows:
        blocks += r
    blocks += [make_block("2026-9-26", 480, 2136), make_block("22:47:11", 620, 2136)]
    blocks += list(extra)
    return sorted(blocks, key=lambda b: (b["avg_y"], b["avg_x"]))


def _extract(category, blocks, report=None, height=H, width=W, y_offset=0):
    return extract_players(blocks, screen_type=category, image_height=height,
                           image_width=width, report=report, y_offset=y_offset)


class TestAllianceExercise:

    def test_suffix_mix_on_consecutive_rows(self):
        frame = _ae_frame([_ae_row("54", "QuillOfAries", "172.23M", 1200),
                           _ae_row("55", "LordMistle", "9.33M", 1390)])
        players = _extract("alliance_exercise", frame)
        assert [(p.rank, p.player_name, p.score) for p in players] == [
            (54, "QuillOfAries", 172_230_000), (55, "LordMistle", 9_330_000)]

    def test_the_mvp_card_is_rank_one_where_it_first_appears(self):
        # Frame 1: the card low on the screen, the header below it.
        frame = _ae_frame([_ae_row(None, "SirCoinsALot", "23.36G", 1880)],
                          card_y=1406, header_y=1806)
        players = _extract("alliance_exercise", frame)
        assert (players[0].rank, players[0].player_name, players[0].score) == (1, "Bang", 35_510_000_000)
        assert players[1].player_name == "SirCoinsALot"

    def test_the_mvp_card_is_rank_one_where_it_pins(self):
        frame = _ae_frame([_ae_row("22", "DimWatson", "4.466", 1166)], card_y=670)
        players = _extract("alliance_exercise", frame)
        assert [(p.rank, p.player_name) for p in players] == [(1, "Bang"), (22, "DimWatson")]
        assert players[1].score == 4_460_000_000

    def test_medal_ranks_unread_and_inferred(self):
        frame = _ae_frame([_ae_row("3", "Alpha", "21.22G", 1185),
                           _ae_row("B", "Beta", "21.21G", 1372),   # the medal "4"
                           _ae_row("5", "Charlie", "19.01G", 1559)])
        report = {}
        players = _extract("alliance_exercise", frame, report)
        assert [(p.rank, p.rank_inferred) for p in players] == [(3, None), (4, True), (5, None)]
        assert report["ranks"]["inferred"] == 1

    def test_the_ticker_above_the_header_is_not_a_row(self):
        frame = _ae_frame([_ae_row("7", "Alpha", "14.38G", 1210)],
                          extra=[make_block("TheGrandRuler", 500, 284), make_block("appointed", 700, 284)])
        assert [p.player_name for p in _extract("alliance_exercise", frame)] == ["Alpha"]

    def test_a_tag_is_removed_and_word_gaps_kept(self):
        row = [make_block("[", 393, 1166), make_block("PoWr", 452, 1166), make_block("]", 508, 1166),
               make_block("Sir", 565, 1166), make_block("Drake", 690, 1166),
               make_block("40", 170, 1206)]
        row += _ae_row(None, "x", "21.22G", 1166)[1:]
        for b, (l, r) in zip(row[:5], [(386, 400), (402, 500), (500, 515), (540, 590), (604, 776)]):
            b["bbox"] = {"vertices": [{"x": l, "y": 1150}, {"x": r, "y": 1150},
                                      {"x": r, "y": 1182}, {"x": l, "y": 1182}]}
        players = _extract("alliance_exercise", _ae_frame([row]))
        assert players[0].player_name == "Sir Drake"

    def test_a_value_without_its_suffix_is_unread_not_a_small_number(self):
        frame = _ae_frame([_ae_row("19", "Storm FF7", "5206", 1200),
                           _ae_row("20", "Rèvo", "5.09G", 1390)])
        players = _extract("alliance_exercise", frame)
        assert [(p.rank, p.player_name, p.score, p.score_unread) for p in players] == [
            (19, "Storm FF7", 0, True), (20, "Rèvo", 5_090_000_000, None)]

    def test_paddleocr_returns_the_whole_line_as_one_word(self):
        row = [make_block("[PoWr]DimWatson", 500, 1166), make_block("22", 170, 1206),
               make_block("Total Damage: 4.46G", 500, 1226)]
        players = _extract("alliance_exercise", _ae_frame([row]))
        assert [(p.rank, p.player_name, p.score) for p in players] == [(22, "DimWatson", 4_460_000_000)]

    def test_timestamp_is_reported_normalised(self):
        report = {}
        _extract("alliance_exercise", _ae_frame([_ae_row("7", "Alpha", "14.38G", 1210)]), report)
        assert report["mail_timestamp"] == "2026-09-26 22:47:11"


def _zs_frame(rows):
    """Zombie Siege: header pinned at ~0.44; per row the name, the wave count
    ~34 px lower at the far right, the rank ~40 px lower, the power ~58 px."""
    blocks = [make_block("Ranking", 190, 1062), make_block("Commander", 480, 1062),
              make_block("Wave", 845, 1062)]
    for rank, name, waves, y in rows:
        blocks += [make_block(name, 430, y), make_block(rank, 125, y + 40),
                   make_block("251771713", 520, y + 58)]
        if waves is not None:
            blocks.append(make_block(waves, 920, y + 34))
    blocks += [make_block("2026-9-23", 470, 2136), make_block("23:03:00", 615, 2136)]
    return sorted(blocks, key=lambda b: (b["avg_y"], b["avg_x"]))


class TestZombieSiege:

    def test_zero_wave_rows_survive_with_ties(self):
        report = {}
        players = _extract("zombie_siege", _zs_frame([
            ("97", "Thorbooster", "3", 1242), ("98", "Stocky", "0", 1436), ("99", "Octoberine", "0", 1628)]),
            report)
        assert [(p.rank, p.score, p.score_unread) for p in players] == [(97, 3, None), (98, 0, None), (99, 0, None)]
        assert "order_violations" not in report  # 0 then 0 is a tie, not a violation

    def test_an_unread_wave_count_keeps_the_row_and_says_so(self):
        # Cloud Vision drops the lone "0" on the real frame 33.
        players = _extract("zombie_siege", _zs_frame([("98", "Stocky", None, 1436)]))
        assert [(p.rank, p.player_name, p.score, p.score_unread) for p in players] == [(98, "Stocky", 0, True)]

    def test_the_power_line_is_neither_a_name_nor_a_score(self):
        players = _extract("zombie_siege", _zs_frame([("13", "Storm FF7", "20", 1127)]))
        assert [(p.player_name, p.score) for p in players] == [("Storm FF7", 20)]

    def test_artwork_text_without_rank_or_score_is_not_a_row(self):
        frame = _zs_frame([("13", "Alpha", "20", 1127)]) + [make_block("Perfect", 400, 1700)]
        assert [p.player_name for p in _extract("zombie_siege", frame)] == ["Alpha"]


def _ds_frame(rows, header=True):
    blocks = [make_block("Battle", 560, 281), make_block("Results", 715, 281),
              make_block("STOP", 805, 460)]
    if header:
        blocks += [make_block("Individual", 495, 600), make_block("Points", 610, 600)]
    for rank, name, points, y in rows:
        blocks += [make_block(name, 600, y), make_block(points, 500, y + 40)]
        if rank is not None:
            blocks.append(make_block(rank, 120, y + 36))
    blocks += [make_block("2026-9-25", 480, 2136), make_block("21:30:12", 615, 2136)]
    return sorted(blocks, key=lambda b: (b["avg_y"], b["avg_x"]))


class TestDesertStorm:

    def test_name_and_points_sharing_x_min(self):
        row = ("10", "JetSki", "2003623", 688)
        blocks = _ds_frame([row], header=False)
        for b in blocks:  # the name box and the points box start at the same x
            if b["text"] in ("JetSki", "2003623"):
                b["bbox"] = {"vertices": [{"x": 432, "y": b["avg_y"] - 15}, {"x": 740, "y": b["avg_y"] - 15},
                                          {"x": 740, "y": b["avg_y"] + 15}, {"x": 432, "y": b["avg_y"] + 15}]}
        players = _extract("desert_storm", blocks)
        assert [(p.rank, p.player_name, p.score) for p in players] == [(10, "JetSki", 2_003_623)]

    def test_no_header_falls_back_to_chrome(self):
        # Once the header scrolls away, the title and "STOP" art are cut by
        # chrome.top_fraction; the rows below are read.
        players = _extract("desert_storm", _ds_frame([("11", "Vengador 31", "1917406", 846)], header=False))
        assert [(p.rank, p.player_name) for p in players] == [(11, "Vengador 31")]

    def test_collapsed_list_is_reported(self):
        report = {}
        assert _extract("desert_storm", _ds_frame([]), report) == []
        assert report["note"] == "no_rows_below_header"

    def test_boundaries_hold_inside_a_stitched_batch(self):
        # Sections of a stitched batch keep the batch's y coordinates.
        offset = 5000
        blocks = _ds_frame([("11", "Vengador 31", "1917406", 846)], header=False)
        shifted = [{**b, "avg_y": b["avg_y"] + offset,
                    "bbox": {"vertices": [{**v, "y": v["y"] + offset} for v in b["bbox"]["vertices"]]}}
                   for b in blocks]
        report = {}
        players = _extract("desert_storm", shifted, report, y_offset=offset)
        assert [(p.rank, p.player_name) for p in players] == [(11, "Vengador 31")]
        assert report["mail_timestamp"] == "2026-09-25 21:30:12"


class TestEndToEnd:

    def test_zero_waves_through_process_batch(self, client):
        import app.routes as routes_module
        blocks = _zs_frame([("98", "Stocky", "0", 1436), ("99", "Octoberine", "0", 1628)])
        routes_module._result_cache.clear()
        with patch("app.routes.run_ocr", return_value=(MagicMock(), "h")), \
             patch("app.routes.extract_text_blocks", return_value=blocks), \
             patch("app.routes.prepare_stitched_batches",
                   return_value=[(Image.new("RGB", (W, H)), [ImageRegion("33.png", 0, H)])]):
            body = client.post("/process-batch", content_type="multipart/form-data",
                               data={"images": png_file_storage("33.png"), "category": "zombie_siege",
                                     "schema_version": "1"}).get_json()
        routes_module._result_cache.clear()
        assert body["results"]["zombie_siege"] == [
            {"player_name": "Stocky", "score": 0, "rank": 98},
            {"player_name": "Octoberine", "score": 0, "rank": 99},
        ]
        section = body["diagnostics"]["sections"][0]
        assert section["mail_timestamp"] == "2026-09-23 23:03:00"
        assert section["method"] == "category_override"

    def test_collapsed_frame_note_through_process_batch(self, client):
        import app.routes as routes_module
        routes_module._result_cache.clear()
        with patch("app.routes.run_ocr", return_value=(MagicMock(), "h")), \
             patch("app.routes.extract_text_blocks", return_value=_ds_frame([])), \
             patch("app.routes.prepare_stitched_batches",
                   return_value=[(Image.new("RGB", (W, H)), [ImageRegion("c.png", 0, H)])]):
            body = client.post("/process-batch", content_type="multipart/form-data",
                               data={"images": png_file_storage("c.png"), "category": "desert_storm"}).get_json()
        routes_module._result_cache.clear()
        assert body["diagnostics"]["sections"][0]["note"] == "no_rows_below_header"

    def test_the_classifier_never_picks_a_mail(self):
        # Mails are override-only: the classifier is a fixed cascade over
        # the ranking families, so a mail frame without a category is
        # unclassified rather than misread as a ranking.
        from app.pipeline.classifier import classify_from_ocr_text
        category, _ = classify_from_ocr_text(_zs_frame([("1", "Alpha", "20", 1127)]), image=None,
                                             filename="zs.png")
        assert category not in MAILS


# ---------------------------------------------------------------------------
# Real recordings
# ---------------------------------------------------------------------------

def _board(prefix: str):
    """Every row read from the recordings whose name starts with prefix."""
    names = [n for n in discover_fixtures(mails=True) if n.startswith(prefix) and "scrubbed" not in n]
    if not names:
        pytest.skip(f"no recordings for {prefix} (set LASTWAR_FIXTURES)")
    category = prefix.split("__")[0]
    out = {}
    for name in names:
        data = load_fixture(name)
        report = {}
        players = extract_players(data["text_blocks"], screen_type=category,
                                  image_height=data["image_height"], image_width=data["image_width"],
                                  report=report)
        out[name] = (players, report)
    return out


def _ranks(board):
    return {p.rank for players, _ in board.values() for p in players if p.rank is not None}


class TestRealBoards:

    def test_alliance_exercise_reads_every_rank(self):
        board = _board("alliance_exercise__1080x2404")
        assert _ranks(board) == set(range(1, 56))
        frame1 = board["alliance_exercise__1080x2404__01"][0]
        assert frame1[0].rank == 1 and frame1[0].score == 35_510_000_000   # card, frame-1 position
        frame6 = board["alliance_exercise__1080x2404__06"][0]
        assert (frame6[0].rank, frame6[0].score) == (1, 35_510_000_000)     # card, pinned
        assert all(r["mail_timestamp"] == "2026-09-26 22:47:11" for _, r in board.values())

    def test_zombie_siege_reads_every_rank_and_keeps_zero_rows(self):
        board = _board("zombie_siege__1080x2404")
        assert _ranks(board) == set(range(1, 100))
        last = {p.rank: p for p in board["zombie_siege__1080x2404__33"][0]}
        assert last[97].score == 3
        assert {98, 99} <= set(last)  # the zero-wave rows are there…
        assert all(last[r].score == 0 for r in (98, 99))  # …with no waves read

    def test_desert_storm_b_reads_every_rank(self):
        assert _ranks(_board("desert_storm__1080x2404__b")) == set(range(1, 23))

    def test_desert_storm_a_reports_a_misread_rank_instead_of_repairing_it(self):
        board = _board("desert_storm__1080x2404__a")
        players, report = board["desert_storm__1080x2404__a02"]
        assert report["ranks"]["duplicates"] == [3]
        assert report["ranks"]["gaps"] == [4, 5]
        # Frame 3 reads rank 14 as 13 as well; the rest of the board is whole.
        assert board["desert_storm__1080x2404__a03"][1]["ranks"]["duplicates"] == [13]
        assert _ranks(board) == {1, 2, 3} | set(range(6, 27)) - {14}

    def test_the_collapsed_frame(self):
        players, report = _board("desert_storm__1080x2404__collapsed")["desert_storm__1080x2404__collapsed"]
        assert players == []
        assert report["note"] == "no_rows_below_header"

    def test_the_second_device(self):
        assert _ranks(_board("zombie_siege__1320x2868")) == set(range(1, 7))
        assert _ranks(_board("desert_storm__1320x2868__a")) == set(range(1, 10))


@pytest.mark.parametrize("fixture_name", discover_fixtures(mails=True))
def test_every_mail_recording_extracts(fixture_name):
    """Smoke: every frame yields rows with names, except the collapsed one."""
    if fixture_path(fixture_name) is None:
        pytest.skip("no mail recordings")
    data = load_fixture(fixture_name)
    category = fixture_name.split("__")[0]
    players = extract_players(data["text_blocks"], screen_type=category,
                              image_height=data["image_height"], image_width=data["image_width"])
    if "collapsed" in fixture_name:
        assert players == []
        return
    assert players and all(p.player_name.strip() for p in players)
    assert all(p.score >= 0 for p in players)
