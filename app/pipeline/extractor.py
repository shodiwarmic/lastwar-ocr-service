"""
app/pipeline/extractor.py

Extracts structured player data (name + score) from Vision API text blocks.

The core challenge here is spatial: Vision API returns a flat list of word
blocks with bounding boxes, not a structured table. We need to reconstruct
which words belong to the same row and then within that row determine which
token is the score (rightmost numeric value) and which tokens form the player
name.

Row clustering algorithm:
    Words are grouped into rows by Y-coordinate proximity. Two words are
    considered to be on the same row if their average Y positions are within
    a tolerance band. The tolerance is expressed as a fraction of image height
    (not a fixed pixel value) so it works correctly across all device resolutions.

    After clustering, each row is sorted left-to-right by average X position.
    The rightmost token that is purely numeric (with commas) is the score.
    All remaining tokens that survive the cleaning pipeline form the player name.

Known edge cases handled:
    - Multi-word names: "gabriel garage", "Shhh mute", "Doc Hollagoon"
    - Names with numbers: "Charlie9042", "Ruthless5432"
    - Alliance tag variants: [PoWr], [PoWr]Pantheon of Wrath (no space)
    - R-badge artefacts on Strength Ranking screen: "R4 ShodiWarmic"
    - Self-player highlight row at the bottom (rank 48, rank 16, etc.)
    - Column header rows: "Ranking  Commander  Points"
    - Scrolling announcement banner (appears above the tab bar — filtered by Y)
"""

from __future__ import annotations

from app.models.schemas import PlayerEntry
from app.utils.text_utils import (
    clean_player_name,
    is_numeric_token,
    is_ui_label,
    parse_score,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Y-coordinate clustering tolerance as a fraction of image height.
# Two word blocks within this fraction of each other are treated as one row.
# 0.018 = 1.8% of image height — at 2400px tall that is ~43px, which safely
# covers line height variation while keeping adjacent rows separate.
ROW_CLUSTER_Y_TOLERANCE_FRACTION = 0.018

# Minimum score value to accept — filters out rank numbers (1–100) and
# small OCR noise values that pass the numeric check.
MIN_VALID_SCORE = 1_000


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def extract_players(
    text_blocks: list[dict],
    screen_type: str,
    image_height: int = 2400,
) -> list[PlayerEntry]:
    """
    Converts a flat list of OCR text blocks into a list of PlayerEntry objects.

    This is the primary entry point called by the route handler after OCR.
    It orchestrates the full extraction pipeline:
        1. Filter obvious noise blocks
        2. Cluster blocks into rows by Y coordinate
        3. Parse each row into (name, score)
        4. Clean names and validate entries
        5. Return validated PlayerEntry objects

    Args:
        text_blocks:  List of dicts from ocr_client.extract_text_blocks(),
                      each containing "text", "bbox", "avg_x", "avg_y".
        screen_type:  Category string (e.g. "friday", "power", "weekly").
                      Used to tune filtering — Strength Ranking rows have
                      R-badge tokens that Daily/Weekly rows do not.
        image_height: Height of the source image in pixels, used to compute
                      the absolute Y-tolerance for row clustering.

    Returns:
        List of validated PlayerEntry objects. Empty list if no valid rows found.
    """
    if not text_blocks:
        logger.warning("extract_players called with empty text_blocks", extra={"screen_type": screen_type})
        return []

    # Step 1: Filter blocks that are clearly not player data
    filtered = _filter_noise_blocks(text_blocks)

    # Step 2: Cluster into rows
    rows = build_rows_from_blocks(filtered, image_height)

    # Step 3-5: Parse, clean, validate
    players: list[PlayerEntry] = []
    for row in rows:
        result = parse_player_row(row)
        if result is None:
            continue

        raw_name, raw_score = result
        clean_name = clean_player_name(raw_name)
        score      = parse_score(raw_score)

        if not is_valid_player_row(clean_name, score):
            continue

        try:
            players.append(PlayerEntry(player_name=clean_name, score=score))
        except Exception:
            # Pydantic validation failed — skip this row
            continue

    logger.info(
        "Extraction complete",
        extra={
            "screen_type":    screen_type,
            "blocks_input":   len(text_blocks),
            "rows_clustered": len(rows),
            "players_found":  len(players),
        },
    )

    return players


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def build_rows_from_blocks(
    text_blocks: list[dict],
    image_height: int = 2400,
) -> list[list[dict]]:
    """
    Groups text blocks into logical rows using Y-coordinate proximity clustering.

    Blocks are already sorted top-to-bottom by ocr_client.extract_text_blocks().
    We iterate through them and assign each block to an existing row if its Y
    position is within tolerance of the row's current average Y, or start a
    new row otherwise.

    After clustering, each row is sorted left-to-right by avg_x so the score
    (rightmost token) is always last.

    Args:
        text_blocks:  Sorted list of text block dicts.
        image_height: Image height in pixels for tolerance calculation.

    Returns:
        List of rows, where each row is a list of text block dicts sorted
        left-to-right.
    """
    if not text_blocks:
        return []

    tolerance = image_height * ROW_CLUSTER_Y_TOLERANCE_FRACTION
    rows: list[list[dict]] = []
    row_avg_ys: list[float] = []

    for block in text_blocks:
        block_y = block["avg_y"]
        placed  = False

        for i, row_y in enumerate(row_avg_ys):
            if abs(block_y - row_y) <= tolerance:
                rows[i].append(block)
                # Update running average Y for the row
                row_avg_ys[i] = sum(b["avg_y"] for b in rows[i]) / len(rows[i])
                placed = True
                break

        if not placed:
            rows.append([block])
            row_avg_ys.append(block_y)

    # Sort each row left-to-right
    for row in rows:
        row.sort(key=lambda b: b["avg_x"])

    return rows


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------

def parse_player_row(row_blocks: list[dict]) -> tuple[str, str] | None:
    """
    Extracts a (raw_name_string, raw_score_string) pair from a single row.

    Strategy:
    - Scan the row right-to-left for the first numeric token (with commas).
      That is the score.
    - Everything to the left of the score token is joined as the player name.
    - If no numeric token is found the row is not a player row → return None.

    Args:
        row_blocks: List of text block dicts for one row, sorted left-to-right.

    Returns:
        (raw_name, raw_score) tuple or None if no score token found.
    """
    if not row_blocks:
        return None

    texts = [b["text"] for b in row_blocks]

    # Find the rightmost numeric token
    score_index = None
    for i in range(len(texts) - 1, -1, -1):
        if is_numeric_token(texts[i]):
            score_index = i
            break

    if score_index is None:
        return None

    raw_score = texts[score_index]
    raw_name  = " ".join(texts[:score_index])

    if not raw_name.strip():
        return None

    return raw_name, raw_score


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_valid_player_row(name: str, score) -> bool:
    """
    Returns True if a (name, score) pair represents a real player row.

    Rejects:
    - Empty names
    - None or zero scores
    - Scores below MIN_VALID_SCORE (filters out rank numbers 1–100)
    - Names that match known UI labels (column headers, tab names)

    Args:
        name:  Cleaned player name string.
        score: Parsed integer score or None.

    Returns:
        True if the row should be included in the output.
    """
    if not name or not name.strip():
        return False

    if score is None:
        return False

    if score < MIN_VALID_SCORE:
        return False

    if is_ui_label(name):
        return False

    return True


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------

def _filter_noise_blocks(text_blocks: list[dict]) -> list[dict]:
    """
    Removes text blocks that are clearly not part of player rows.

    Filtered out:
    - Blocks whose text exactly matches known UI labels
    - Very short single-character blocks (icon OCR noise)
    - Blocks containing only punctuation or whitespace

    This is a lightweight pre-filter. The main cleaning happens per-row in
    clean_player_name() and is_valid_player_row().

    Args:
        text_blocks: Raw list of text block dicts from extract_text_blocks().

    Returns:
        Filtered list with obvious noise removed.
    """
    filtered = []
    for block in text_blocks:
        text = block["text"].strip()

        # Skip empty or whitespace-only
        if not text:
            continue

        # Skip single characters that are likely icon OCR artefacts
        # (but keep digits since rank "1" through "9" are single chars)
        if len(text) == 1 and not text.isdigit():
            continue

        # Skip known UI labels
        if is_ui_label(text):
            continue

        filtered.append(block)

    return filtered
