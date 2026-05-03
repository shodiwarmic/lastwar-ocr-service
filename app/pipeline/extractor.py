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

from app.models.schemas import PlayerEntry, ScoreCandidate
from app.pipeline.screen_definitions import RowClusteringConfig, get_definition_for_category
from app.utils.text_utils import (
    all_crash_splits,
    clean_player_name,
    is_numeric_token,
    is_ui_label,
    parse_score,
    split_name_score_crash,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Y-coordinate clustering tolerance as a fraction of image height.
# Two word blocks within this fraction of each other are treated as one row.
# 0.012 = 1.2% of image height — at 2400px tall that is ~29px, which safely
# covers line height variation (~10-15px) while keeping the player name line
# separate from the alliance subtitle line (~60px below).
ROW_CLUSTER_Y_TOLERANCE_FRACTION = 0.012

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
    image_width: int = 1080,
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
        image_width:  Width of the source image in pixels, used to compute
                      the relative gap threshold for word spacing detection.

    Returns:
        List of validated PlayerEntry objects. Empty list if no valid rows found.
    """
    if not text_blocks:
        logger.warning("extract_players called with empty text_blocks", extra={"screen_type": screen_type})
        return []

    defn = get_definition_for_category(screen_type)
    row_config = defn.row_clustering if defn else RowClusteringConfig()

    # Step 1: Filter blocks that are clearly not player data
    filtered = _filter_noise_blocks(text_blocks)

    # Step 2: Cluster into rows
    rows = build_rows_from_blocks(filtered, image_height, row_config)

    # Step 3-4: Parse and clean every row; note which rows involved a crash split
    # so we can validate them against their neighbours in the next pass.
    _pre: list[dict] = []  # {"name": str, "score": int, "crash_text": str | None}
    for row in rows:
        result = parse_player_row(row, image_width=image_width, row_config=row_config)
        if result is None:
            continue

        raw_name, raw_score = result
        clean_name = clean_player_name(raw_name)
        score      = parse_score(raw_score)

        if not is_valid_player_row(clean_name, score, min_score=row_config.min_score):
            continue

        # Identify the crash block in this row, if any, for use in pass 2.
        crash_text = next(
            (b["text"] for b in reversed(row) if split_name_score_crash(b["text"])),
            None,
        )
        all_splits = all_crash_splits(crash_text) if crash_text else []
        _pre.append({"name": clean_name, "score": score, "crash_text": crash_text, "all_splits": all_splits})

    # Step 5: Validate crash rows using adjacent scores as monotonicity bounds.
    #
    # The leaderboard is sorted highest-to-lowest, so every score must satisfy:
    #     score[i-1] >= score[i] >= score[i+1]
    #
    # For a crash row whose heuristic score falls outside that window, we try
    # progressively larger alternative splits (ascending score order) until one
    # fits.  When no neighbour exists (rank 1 has no row above; the last visible
    # row has no row below) only the available single bound is enforced.
    players: list[PlayerEntry] = []
    for i, entry in enumerate(_pre):
        name       = entry["name"]
        score      = entry["score"]
        all_splits = entry["all_splits"]

        # Build candidates list for any crash row that has genuine ambiguity
        # (two or more valid splits).  Ordered smallest score first so
        # candidates[0] always matches the heuristic top-level name/score.
        candidates: list[ScoreCandidate] | None = None
        if len(all_splits) > 1:
            candidates = [
                ScoreCandidate(player_name=clean_player_name(p), score=parse_score(s))
                for p, s in all_splits
                if is_valid_player_row(clean_player_name(p), parse_score(s), min_score=row_config.min_score)
            ]
            if len(candidates) < 2:
                candidates = None  # collapsed to one valid option — unambiguous

        if entry["crash_text"]:
            upper = _pre[i - 1]["score"] if i > 0            else None
            lower = _pre[i + 1]["score"] if i < len(_pre) - 1 else None

            fits = (upper is None or score <= upper) and (lower is None or score >= lower)

            if not fits:
                for alt_prefix, alt_score_str in all_splits:
                    alt_score = parse_score(alt_score_str)
                    alt_name  = clean_player_name(alt_prefix)
                    if not is_valid_player_row(alt_name, alt_score, min_score=row_config.min_score):
                        continue
                    alt_fits = (
                        (upper is None or alt_score <= upper)
                        and (lower is None or alt_score >= lower)
                    )
                    if alt_fits:
                        logger.debug(
                            "Crash split corrected via adjacent-score bounds",
                            extra={
                                "original_name":  name,
                                "original_score": score,
                                "corrected_name":  alt_name,
                                "corrected_score": alt_score,
                                "upper_bound": upper,
                                "lower_bound": lower,
                            },
                        )
                        name, score = alt_name, alt_score
                        break

        try:
            players.append(PlayerEntry(player_name=name, score=score, candidates=candidates))
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
    row_config: RowClusteringConfig = None,
) -> list[list[dict]]:
    """
    Builds player rows using score-anchored upward clustering.

    Instead of general Y-proximity clustering (which merges player name and
    alliance subtitle since they have similar gaps to the score), this approach:

    1. Identifies score tokens (numeric values >= row_config.min_score)
    2. For each score, collects all non-score text blocks within an upward
       band (UP_BAND px above the score, DOWN_BAND px below)
    3. Name is always ABOVE the score in the UI; alliance is below — so
       the asymmetric band captures the name and excludes the alliance

    Band sizes are derived from screen definition fractions scaled to
    image_height so they work correctly across device resolutions.

    Args:
        text_blocks:  List of text block dicts sorted top-to-bottom.
        image_height: Source image height in pixels, used to compute
                      absolute band sizes from the definition fractions.
        row_config:   Row clustering configuration from the screen
                      definition. Defaults to RowClusteringConfig() if None.

    Returns:
        List of rows, each containing score block + associated name blocks,
        sorted left-to-right.
    """
    if not text_blocks:
        return []

    if row_config is None:
        row_config = RowClusteringConfig()

    UP_BAND   = max(1, int(row_config.score_anchored.up_band_fraction   * image_height))
    DOWN_BAND = max(1, int(row_config.score_anchored.down_band_fraction * image_height))
    min_score = row_config.min_score

    # Separate scores from name/other tokens
    score_blocks = [b for b in text_blocks if _is_score_block(b, min_score)]
    other_blocks = [b for b in text_blocks if not _is_score_block(b, min_score)]

    rows = []
    for score_block in score_blocks:
        score_y = score_block["avg_y"]

        # Collect name tokens in the upward band
        row_blocks = [
            b for b in other_blocks
            if (score_y - UP_BAND) <= b["avg_y"] <= (score_y + DOWN_BAND)
        ]
        row_blocks.append(score_block)
        row_blocks.sort(key=lambda b: b["avg_x"])
        rows.append(row_blocks)

    return rows


def _is_score_block(block: dict, min_score: int = MIN_VALID_SCORE) -> bool:
    """
    Returns True if a block contains a player score >= min_score.

    Handles two cases:
    1. Pure numeric token (e.g. "3,045,000") — direct parse_score check.
    2. Crash token (e.g. "Ruthless54323,045,000") — name+score merged into
       one block when OCR sees no gap between a digit-ending name and the
       score.  split_name_score_crash extracts the embedded score.
    """
    text = block["text"]
    val = parse_score(text)
    if val is not None and val >= min_score:
        return True
    split = split_name_score_crash(text)
    if split:
        _, score_str = split
        val = parse_score(score_str)
        if val is not None and val >= min_score:
            return True
    return False


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------

def parse_player_row(
    row_blocks: list[dict],
    image_width: int = 0,
    row_config: RowClusteringConfig = None,
) -> tuple[str, str] | None:
    """
    Extracts a (raw_name_string, raw_score_string) pair from a single row.

    Strategy (ported and improved from original spatial parser):
    1. Find the rightmost numeric token — that is the score.
    2. Only accept name tokens that are spatially LEFT of the score token
       (leftward X constraint). This eliminates alliance display names and
       other text that appears to the right of or at the same X as the score.
    3. Reconstruct the name string using pixel gap detection: insert a space
       between tokens when the gap between them exceeds GAP_THRESHOLD pixels,
       which correctly handles multi-word names like "gabriel garage" and
       avoids over-spacing in compact names.

    Args:
        row_blocks:  List of text block dicts for one row, sorted left-to-right.
                     Each block must have "text", "avg_x" fields. Optionally
                     "bbox" for precise left/right edge calculation.
        image_width: Width of the source image in pixels. Used to compute a
                     relative GAP_THRESHOLD so spacing scales across devices.
        row_config:  Row clustering configuration from the screen definition.
                     Provides word_gap_fraction and min_word_gap_px. Defaults
                     to RowClusteringConfig() if None.

    Returns:
        (raw_name, raw_score) tuple or None if no score token found.
    """
    if not row_blocks:
        return None

    if row_config is None:
        row_config = RowClusteringConfig()

    # Minimum pixel gap between word bounding boxes to insert a space.
    # Fraction and minimum are sourced from the screen definition so they
    # stay consistent with any per-screen tuning in the YAML.
    GAP_THRESHOLD = (
        max(row_config.min_word_gap_px, int(image_width * row_config.word_gap_fraction))
        if image_width else row_config.min_word_gap_px * 2
    )

    # Find the rightmost numeric token and its left edge X position.
    # Pass 1: pure numeric token (the normal case).
    score_index   = None
    raw_score     = None
    score_left_x  = None
    crash_name_prefix: Optional[str] = None

    for i in range(len(row_blocks) - 1, -1, -1):
        if is_numeric_token(row_blocks[i]["text"]):
            score_index  = i
            raw_score    = row_blocks[i]["text"]
            score_left_x = _block_left_x(row_blocks[i])
            break

    # Pass 2: crash-token fallback — name+score merged into one block.
    # e.g. "Ruthless54323,045,000" → name_prefix="Ruthless5432", score="3,045,000"
    if score_index is None:
        for i in range(len(row_blocks) - 1, -1, -1):
            split = split_name_score_crash(row_blocks[i]["text"])
            if split:
                crash_name_prefix, raw_score = split
                score_index  = i
                score_left_x = _block_left_x(row_blocks[i])
                break

    if score_index is None:
        return None

    # Collect name blocks that are spatially LEFT of the score token.
    name_blocks = [
        b for b in row_blocks[:score_index]
        if score_left_x is None or _block_right_x(b) <= score_left_x
    ]

    # For crash tokens, the name prefix extracted from the merged block is
    # appended as a virtual block so it participates in the name string.
    # It carries no bbox, so the gap check falls back to inserting a space.
    if crash_name_prefix is not None:
        name_blocks.append({"text": crash_name_prefix, "avg_x": score_left_x or 0})

    if not name_blocks:
        return None

    # Build name string with gap-based space insertion
    parts = [name_blocks[0]["text"]]
    for i in range(1, len(name_blocks)):
        prev_right = _block_right_x(name_blocks[i - 1])
        curr_left  = _block_left_x(name_blocks[i])
        gap = curr_left - prev_right if (curr_left and prev_right) else GAP_THRESHOLD + 1
        if gap > GAP_THRESHOLD:
            parts.append(" ")
        parts.append(name_blocks[i]["text"])

    raw_name = "".join(parts).strip()

    if not raw_name:
        return None

    return raw_name, raw_score


def _block_left_x(block: dict) -> float | None:
    """Returns the leftmost X coordinate of a text block, or None if unavailable."""
    bbox = block.get("bbox")
    if not bbox:
        return block.get("avg_x")
    try:
        vertices = list(bbox.vertices)
        return min(v.x for v in vertices)
    except AttributeError:
        pass
    try:
        verts = bbox.get("vertices", [])
        if verts:
            return min(v.get("x", 0) for v in verts)
    except (TypeError, AttributeError):
        pass
    return block.get("avg_x")


def _block_right_x(block: dict) -> float | None:
    """Returns the rightmost X coordinate of a text block, or None if unavailable."""
    bbox = block.get("bbox")
    if not bbox:
        return block.get("avg_x")
    try:
        vertices = list(bbox.vertices)
        return max(v.x for v in vertices)
    except AttributeError:
        pass
    try:
        verts = bbox.get("vertices", [])
        if verts:
            return max(v.get("x", 0) for v in verts)
    except (TypeError, AttributeError):
        pass
    return block.get("avg_x")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_valid_player_row(name: str, score, min_score: int = MIN_VALID_SCORE) -> bool:
    """
    Returns True if a (name, score) pair represents a real player row.

    Rejects:
    - Empty names
    - None or zero scores
    - Scores below min_score (filters out rank numbers 1–100)
    - Names that match known UI labels (column headers, tab names)

    Args:
        name:      Cleaned player name string.
        score:     Parsed integer score or None.
        min_score: Minimum score to accept. Sourced from the screen definition's
                   row_clustering.min_score; falls back to MIN_VALID_SCORE.

    Returns:
        True if the row should be included in the output.
    """
    if not name or not name.strip():
        return False

    if score is None:
        return False

    if score < min_score:
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
