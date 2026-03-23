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

    The active tab (Fri.) is given a slightly lower Y value (255) than the
    inactive tabs (265) to simulate the active tab being visually elevated
    in the UI — matching the spatial signal the classifier uses to identify
    the active day when multiple day abbreviations are present.
    """
    return [
        make_block("RANKING",       300,  80),
        make_block("Daily",         150, 180),
        make_block("Rank",          195, 180),
        make_block("Weekly",        350, 180),
        make_block("Rank",          400, 180),
        # Inactive day tabs — slightly lower Y (higher pixel value = lower on screen)
        make_block("Mon.",           75, 265),
        make_block("Tues.",         185, 265),
        make_block("Wed.",          295, 265),
        make_block("Thur.",         405, 265),
        # Active tab — lower Y value = higher on screen = visually elevated
        make_block("Fri.",          515, 255),
        make_block("Sat.",          620, 265),
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
