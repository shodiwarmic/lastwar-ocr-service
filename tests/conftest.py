"""
tests/conftest.py

Shared pytest fixtures available to all test modules.

Fixtures here cover:
    - Flask test client (no Vision API called)
    - Sample OCR text block lists matching the structure returned by
      ocr_client.extract_text_blocks() — built from the 10 sample screenshots
    - PIL Image stubs for testing image utilities without real screenshots

Loading real fixtures:
    After running tools/capture_ocr_fixture.py against your sample screenshots,
    JSON files will appear in tests/fixtures/ocr_responses/. The load_fixture()
    helper loads these and returns the text_blocks list directly, ready to pass
    to classify_from_ocr_text() or extract_players().

If fixture files are not yet present (e.g. in CI before first capture),
tests that depend on them are automatically skipped via the
`require_fixture` marker defined below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import create_app

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ocr_responses"


# ---------------------------------------------------------------------------
# Flask app / test client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def flask_app():
    """
    Creates a Flask app instance configured for testing.

    Session-scoped so the app is created once per test run rather than once
    per test function — mirrors how Cloud Run reuses a single app instance.
    """
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    return app


@pytest.fixture()
def client(flask_app):
    """
    Returns a Flask test client for making HTTP requests in tests.

    Function-scoped (default) so each test gets a fresh request context.
    The Vision API is never called through this client — route tests should
    mock run_ocr() at the call site.
    """
    return flask_app.test_client()


# ---------------------------------------------------------------------------
# Fixture file loading
# ---------------------------------------------------------------------------

def load_fixture(name: str) -> dict:
    """
    Loads a JSON OCR fixture file by name and returns its parsed contents.

    Args:
        name: Fixture filename without extension (e.g. "8851" for 8851.json).

    Returns:
        Dict with keys: source_file, image_hash, image_width, image_height,
        annotation, text_blocks.

    Raises:
        FileNotFoundError: If the fixture file does not exist.
        Use the `skip_if_no_fixture` fixture to gracefully skip tests
        when fixtures have not yet been captured.
    """
    fixture_path = FIXTURE_DIR / f"{name}.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    with open(fixture_path, encoding="utf-8") as f:
        return json.load(f)


def get_text_blocks(fixture_name: str) -> list[dict]:
    """
    Convenience wrapper: loads a fixture and returns just the text_blocks list.

    Args:
        fixture_name: Fixture name without extension.

    Returns:
        List of text block dicts ready to pass to classify_from_ocr_text()
        or extract_players().
    """
    return load_fixture(fixture_name)["text_blocks"]


@pytest.fixture()
def skip_if_no_fixture():
    """
    Returns a helper function that skips the current test if a fixture is missing.

    Usage in a test:
        def test_something(skip_if_no_fixture):
            skip_if_no_fixture("8851")
            blocks = get_text_blocks("8851")
            ...
    """
    def _skip(fixture_name: str):
        fixture_path = FIXTURE_DIR / f"{fixture_name}.json"
        if not fixture_path.exists():
            pytest.skip(
                f"Fixture '{fixture_name}.json' not found. "
                f"Run: python tools/capture_ocr_fixture.py <screenshots_dir>"
            )
    return _skip


# ---------------------------------------------------------------------------
# Synthetic OCR block builders (no real fixtures needed)
# These build minimal text block structures that match the format produced
# by ocr_client.extract_text_blocks(), used for unit tests of the classifier
# and extractor logic without any file I/O.
# ---------------------------------------------------------------------------

def make_block(text: str, avg_x: float, avg_y: float) -> dict:
    """
    Creates a minimal text block dict in the format expected by the pipeline.

    Args:
        text:  The text content of the block.
        avg_x: Horizontal centre position (pixels from left).
        avg_y: Vertical centre position (pixels from top).

    Returns:
        Dict with text, avg_x, avg_y, and a minimal bbox.
    """
    half = 20
    return {
        "text":  text,
        "avg_x": avg_x,
        "avg_y": avg_y,
        "bbox": {
            "vertices": [
                {"x": int(avg_x - half), "y": int(avg_y - half)},
                {"x": int(avg_x + half), "y": int(avg_y - half)},
                {"x": int(avg_x + half), "y": int(avg_y + half)},
                {"x": int(avg_x - half), "y": int(avg_y + half)},
            ]
        },
    }


@pytest.fixture()
def weekly_rank_blocks():
    """
    Synthetic OCR blocks representing a Weekly Rank screen header area.
    Used to test classify_from_ocr_text() without real fixtures.
    """
    return [
        make_block("RANKING",       300, 80),
        make_block("Daily",         150, 200),
        make_block("Rank",          190, 200),
        make_block("Weekly",        350, 200),
        make_block("Rank",          400, 200),
        make_block("Ranking",        80, 290),
        make_block("Commander",     300, 290),
        make_block("Points",        550, 290),
    ]


@pytest.fixture()
def strength_ranking_blocks():
    """
    Synthetic OCR blocks representing a Strength Ranking screen.
    """
    return [
        make_block("STRENGTH",      250,  80),
        make_block("RANKING",       370,  80),
        make_block("Power",         120, 200),
        make_block("Kills",         300, 200),
        make_block("Donation",      490, 200),
        make_block("Ranking",        80, 290),
        make_block("Commander",     300, 290),
        make_block("Power",         550, 290),
    ]


@pytest.fixture()
def friday_daily_blocks():
    """
    Synthetic OCR blocks representing a Friday Daily Rank screen.

    The active tab (Fri) is given WITHOUT a trailing period to simulate the
    signal used by the scoring-based active day detector. Inactive day tabs
    retain their periods (Mon., Tues., etc.). All tabs sit at the same Y
    to match real screenshot behaviour where all tabs are in one horizontal bar.
    """
    return [
        make_block("RANKING",       300,  80),
        make_block("Daily",         150, 180),
        make_block("Rank",          195, 180),
        make_block("Weekly",        350, 180),
        make_block("Rank",          400, 180),
        # Inactive tabs — with trailing period
        make_block("Mon.",           75, 260),
        make_block("Tues.",         185, 260),
        make_block("Wed.",          295, 260),
        make_block("Thur.",         405, 260),
        # Active tab — no trailing period (scores +2 vs +1 for inactive tabs)
        make_block("Fri",           515, 260),
        make_block("Sat.",          620, 260),
        make_block("Ranking",        80, 340),
        make_block("Commander",     300, 340),
        make_block("Points",        550, 340),
    ]


@pytest.fixture()
def player_row_blocks():
    """
    Synthetic OCR blocks representing a single player row.
    Matches: rank=1, name="SirBucksALot", alliance="[PoWr]", score=45,635,206
    """
    return [
        make_block("1",               60, 400),
        make_block("[PoWr]",         160, 400),
        make_block("SirBucksALot",   300, 400),
        make_block("45,635,206",     580, 400),
    ]


@pytest.fixture()
def kills_ranking_blocks():
    """
    Synthetic OCR blocks representing a Strength Ranking screen with the Kills
    tab active.  Player data is taken from the Kills screenshot shared by the user:
        1 Charlie9042 17,886,167 | 2 SirBucksALot 14,511,061 | 3 TheDudeAbides22 14,486,434
        4 pudgey27 6,689,046 | 5 Cloud FF7 6,504,689 | 6 TheMojoDude 5,967,706
        7 SubZero221 5,506,236 | 15 ShodiWarmic 2,698,480

    "Kills" appears at y=200 as a tab label AND at y=290 as a column header —
    the dual-occurrence is preserved as it reflects real screenshot OCR output.
    """
    return [
        # Header
        make_block("STRENGTH",          250,  80),
        make_block("RANKING",           370,  80),
        # Top tabs — Kills is active (orange in real image)
        make_block("Power",             120, 200),
        make_block("Kills",             300, 200),   # tab label
        make_block("Donation",          490, 200),
        # Column headers
        make_block("Ranking",            80, 290),
        make_block("Commander",         300, 290),
        make_block("Kills",             550, 290),   # column header (second occurrence)
        # Player rows (R-badge tokens are stripped by the cleaner)
        make_block("1",                  50, 380),
        make_block("R3",                130, 380),
        make_block("Charlie9042",       300, 380),
        make_block("17,886,167",        600, 380),
        make_block("2",                  50, 460),
        make_block("R4",                130, 460),
        make_block("SirBucksALot",      300, 460),
        make_block("14,511,061",        600, 460),
        make_block("3",                  50, 540),
        make_block("R5",                130, 540),
        make_block("TheDudeAbides22",   300, 540),
        make_block("14,486,434",        600, 540),
        make_block("4",                  50, 620),
        make_block("R4",                130, 620),
        make_block("pudgey27",          300, 620),
        make_block("6,689,046",         600, 620),
        make_block("5",                  50, 700),
        make_block("R3",                130, 700),
        make_block("Cloud",             270, 700),
        make_block("FF7",               340, 700),
        make_block("6,504,689",         600, 700),
        make_block("6",                  50, 780),
        make_block("R3",                130, 780),
        make_block("TheMojoDude",       300, 780),
        make_block("5,967,706",         600, 780),
        make_block("7",                  50, 860),
        make_block("R4",                130, 860),
        make_block("SubZero221",        300, 860),
        make_block("5,506,236",         600, 860),
        make_block("15",                 50, 940),
        make_block("R4",                130, 940),
        make_block("ShodiWarmic",       300, 940),
        make_block("2,698,480",         600, 940),
    ]


@pytest.fixture()
def donation_daily_blocks():
    """
    Synthetic OCR blocks representing a Strength Ranking screen with the Donation
    tab active and the Daily sub-tab selected.  Data from the Daily donation screenshot:
        1 BlackIce2 14,800 | 2 Cloud FF7 11,900 | 3 Hendley1 9,900
        4 Crazy Carol 9,600 | 5 Davilson Pirani 9,400 | 6 JimmyJames56830 9,350
        7 Doc Hollagoon 8,650 | 61 ShodiWarmic 5,900

    "Points" appears as part of the "Donation Points" column header — preserved
    as it reflects real screenshot OCR output.
    """
    return [
        # Header
        make_block("STRENGTH",           250,  80),
        make_block("RANKING",            370,  80),
        # Top tabs — Donation is active (orange in real image)
        make_block("Power",              120, 200),
        make_block("Kills",              300, 200),
        make_block("Donation",           490, 200),
        # Donation sub-tabs — Daily is active (bright/white), Weekly inactive
        make_block("Daily",              150, 250),
        make_block("Weekly",             350, 250),
        # Column headers ("Donation Points" → two tokens)
        make_block("Ranking",             80, 310),
        make_block("Commander",          300, 310),
        make_block("Donation",           480, 310),
        make_block("Points",             560, 310),
        # Player rows
        make_block("1",                   50, 400),
        make_block("R3",                 130, 400),
        make_block("BlackIce2",          300, 400),
        make_block("14,800",             600, 400),
        make_block("2",                   50, 480),
        make_block("R3",                 130, 480),
        make_block("Cloud",              270, 480),
        make_block("FF7",                340, 480),
        make_block("11,900",             600, 480),
        make_block("3",                   50, 560),
        make_block("R3",                 130, 560),
        make_block("Hendley1",           300, 560),
        make_block("9,900",              600, 560),
        make_block("4",                   50, 640),
        make_block("R3",                 130, 640),
        make_block("Crazy",              260, 640),
        make_block("Carol",              320, 640),
        make_block("9,600",              600, 640),
        make_block("5",                   50, 720),
        make_block("R3",                 130, 720),
        make_block("Davilson",           260, 720),
        make_block("Pirani",             340, 720),
        make_block("9,400",              600, 720),
        make_block("6",                   50, 800),
        make_block("R3",                 130, 800),
        make_block("JimmyJames56830",    300, 800),
        make_block("9,350",              600, 800),
        make_block("7",                   50, 880),
        make_block("R3",                 130, 880),
        make_block("Doc",                260, 880),
        make_block("Hollagoon",          340, 880),
        make_block("8,650",              600, 880),
        make_block("61",                  50, 960),
        make_block("R4",                 130, 960),
        make_block("ShodiWarmic",        300, 960),
        make_block("5,900",              600, 960),
    ]


@pytest.fixture()
def donation_weekly_blocks():
    """
    Synthetic OCR blocks representing a Strength Ranking screen with the Donation
    tab active and the Weekly sub-tab selected.  Data from the Weekly donation screenshot:
        1 CaptTrickster727 28,300 | 2 BlackIce2 23,250 | 3 Crazy Carol 20,350
        4 Hendley1 19,450 | 5 JimmyJames56830 18,600 | 6 Cloud FF7 18,450
        7 Davilson Pirani 18,200 | 65 ShodiWarmic 11,900

    Structurally identical to donation_daily_blocks with Weekly as the active sub-tab.
    Text-only classification returns donation_daily for both (sub-tab is ambiguous
    without colour sampling); active sub-tab requires a PIL image for colour detection.
    """
    return [
        # Header
        make_block("STRENGTH",           250,  80),
        make_block("RANKING",            370,  80),
        # Top tabs — Donation is active
        make_block("Power",              120, 200),
        make_block("Kills",              300, 200),
        make_block("Donation",           490, 200),
        # Donation sub-tabs — Weekly is active
        make_block("Daily",              150, 250),
        make_block("Weekly",             350, 250),
        # Column headers
        make_block("Ranking",             80, 310),
        make_block("Commander",          300, 310),
        make_block("Donation",           480, 310),
        make_block("Points",             560, 310),
        # Player rows
        make_block("1",                   50, 400),
        make_block("R3",                 130, 400),
        make_block("CaptTrickster727",   300, 400),
        make_block("28,300",             600, 400),
        make_block("2",                   50, 480),
        make_block("R3",                 130, 480),
        make_block("BlackIce2",          300, 480),
        make_block("23,250",             600, 480),
        make_block("3",                   50, 560),
        make_block("R3",                 130, 560),
        make_block("Crazy",              260, 560),
        make_block("Carol",              320, 560),
        make_block("20,350",             600, 560),
        make_block("4",                   50, 640),
        make_block("R3",                 130, 640),
        make_block("Hendley1",           300, 640),
        make_block("19,450",             600, 640),
        make_block("5",                   50, 720),
        make_block("R3",                 130, 720),
        make_block("JimmyJames56830",    300, 720),
        make_block("18,600",             600, 720),
        make_block("6",                   50, 800),
        make_block("R3",                 130, 800),
        make_block("Cloud",              260, 800),
        make_block("FF7",                330, 800),
        make_block("18,450",             600, 800),
        make_block("7",                   50, 880),
        make_block("R3",                 130, 880),
        make_block("Davilson",           260, 880),
        make_block("Pirani",             340, 880),
        make_block("18,200",             600, 880),
        make_block("65",                  50, 960),
        make_block("R4",                 130, 960),
        make_block("ShodiWarmic",        300, 960),
        make_block("11,900",             600, 960),
    ]
