"""
app/utils/text_utils.py

Regex patterns and string cleaning utilities used by the extractor and classifier.

Centralising these here means:
- Patterns are compiled once at import time (performance)
- Extractor and classifier share identical cleaning logic (no drift)
- Patterns are easy to update when OCR edge cases are discovered in production
- Unit tests for cleaning live in one place

All functions operate on plain strings and have no dependencies on PIL or
the Vision API — they can be tested in complete isolation.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Alliance tags: [PoWr], [TAG], (Tag), etc.
# Handles optional spaces inside brackets and mixed case
_ALLIANCE_TAG_RE = re.compile(r"[\[\(][^\]\)]{1,10}[\]\)]", re.IGNORECASE)

# Leading rank numbers with optional punctuation: "1", "1.", "48 ", etc.
# Only matches at the very start of the string
_LEADING_RANK_RE = re.compile(r"^\d{1,3}[.\s]*")

# R-badge OCR artefacts: R1, R2, R3, R4, R5 appearing as standalone tokens
# These come from the rank badge icons Vision sometimes reads as text
_RBADGE_RE = re.compile(r"\bR[1-5]\b")

# Score string: digits optionally separated by commas (e.g. "161,528,090")
_SCORE_RE = re.compile(r"^[\d,]+$")

# Stray non-alphanumeric characters that are clearly OCR noise.
# Keeps spaces, hyphens, underscores, apostrophes, and extended Latin/Cyrillic
# so accented names like Pàcha are preserved.
_OCR_NOISE_RE = re.compile(r"[^\w\s\-_\'\u00C0-\u024F\u0400-\u04FF]")

# Thai character range — appears as OCR noise from rank badge icons in
# the Strength Ranking screen. Ported from the original app.py implementation.
_THAI_RE = re.compile(r"[\u0E00-\u0E7F]+")

# Collapse multiple consecutive spaces into one
_WHITESPACE_RE = re.compile(r"\s{2,}")

# Known UI label strings that should never appear as player names
_UI_LABELS = frozenset({
    "ranking", "commander", "points", "power", "kills", "donation",
    "daily rank", "weekly rank", "strength ranking", "mon", "tues",
    "wed", "thur", "fri", "sat", "your alliance",
})

# Alliance name suffixes that survive bracket stripping because they appear
# as plain text tokens in the OCR output rather than inside brackets.
# These are the visible alliance display names shown beneath player names.
# Add new entries here as alliances are encountered in production logs.
_ALLIANCE_NAME_SUFFIXES: list[str] = [
    "Pantheon of Wrath",
]

# Pre-compiled case-insensitive patterns for efficient suffix stripping
_ALLIANCE_SUFFIX_RES = [
    re.compile(re.escape(s), re.IGNORECASE)
    for s in _ALLIANCE_NAME_SUFFIXES
]

# Bare alliance tag abbreviations — the bracketed form [PoWr] is handled by
# _ALLIANCE_TAG_RE, but OCR sometimes drops the brackets leaving the
# abbreviation as a standalone token. We detect alliance abbreviations
# generically by pattern: short tokens (2-6 chars) with internal uppercase
# (e.g. PoWr, CoRe, TaGs) rather than hardcoding specific alliance names.
def _looks_like_tag(token: str) -> bool:
    """Returns True if a token matches the alliance abbreviation pattern."""
    if not 2 <= len(token) <= 6:
        return False
    # Must have at least one uppercase letter after position 0
    return any(c.isupper() for c in token[1:])


# Day tab label to output key mapping
# Keys are lowercase normalised versions of what OCR returns
_DAY_LABEL_MAP = {
    "mon":      "monday",
    "mon.":     "monday",
    "monday":   "monday",
    "tues":     "tuesday",
    "tues.":    "tuesday",
    "tuesday":  "tuesday",
    "wed":      "wednesday",
    "wed.":     "wednesday",
    "wednesday":"wednesday",
    "thur":     "thursday",
    "thur.":    "thursday",
    "thursday": "thursday",
    "fri":      "friday",
    "fri.":     "friday",
    "friday":   "friday",
    "sat":      "saturday",
    "sat.":     "saturday",
    "saturday": "saturday",
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def strip_alliance_tag(name: str) -> str:
    """
    Removes alliance tags in square or round brackets from a player name.

    Handles tags anywhere in the string, not just at the start, because OCR
    occasionally places the tag after the name depending on layout.

    Args:
        name: Raw player name string which may contain an alliance tag.

    Returns:
        Name with all alliance tags removed and surrounding whitespace stripped.

    Examples:
        strip_alliance_tag("[PoWr] SirBucksALot") → "SirBucksALot"
        strip_alliance_tag("ShodiWarmic [PoWr]Pantheon of Wrath") → "ShodiWarmic Pantheon of Wrath"
    """
    return _ALLIANCE_TAG_RE.sub("", name).strip()


def strip_leading_rank(name: str) -> str:
    """
    Removes a leading rank number from a raw OCR string.

    OCR sometimes captures the rank badge number as part of the same text
    block as the player name, particularly for single-digit ranks.

    Args:
        name: Raw string that may begin with a rank number.

    Returns:
        String with the leading rank number removed.

    Examples:
        strip_leading_rank("1 SirBucksALot") → "SirBucksALot"
        strip_leading_rank("48ShodiWarmic")   → "ShodiWarmic"
    """
    return _LEADING_RANK_RE.sub("", name).strip()


def strip_rbadge_artefacts(name: str) -> str:
    """
    Removes R1–R5 badge OCR artefacts from a player name string.

    The Strength Ranking screen displays R3, R4, R5 rank badges next to player
    names. Vision API sometimes includes these as text tokens mixed into the
    name string.

    Args:
        name: Player name string potentially containing Rn badge tokens.

    Returns:
        Name with all Rn tokens removed.

    Examples:
        strip_rbadge_artefacts("R4 ShodiWarmic") → "ShodiWarmic"
        strip_rbadge_artefacts("SirBucksALot R4") → "SirBucksALot"
    """
    return _RBADGE_RE.sub("", name).strip()


def strip_ocr_noise(name: str) -> str:
    """
    Removes stray non-alphanumeric characters that are clearly OCR noise.

    Preserves spaces, hyphens, underscores, and apostrophes which can all
    legitimately appear in player names.

    Args:
        name: Player name string after alliance tag and rank stripping.

    Returns:
        Cleaned name string with noise characters removed.
    """
    return _OCR_NOISE_RE.sub("", name).strip()


def collapse_whitespace(s: str) -> str:
    """
    Collapses multiple consecutive spaces into a single space.

    Applied as the final step of name cleaning after all removal passes,
    since earlier removals can leave double spaces.

    Args:
        s: Any string.

    Returns:
        String with all runs of whitespace collapsed to a single space.
    """
    return _WHITESPACE_RE.sub(" ", s).strip()


def strip_thai_characters(name: str) -> str:
    """
    Removes Thai script characters from a name string.

    Thai characters appear as OCR noise when Vision API misreads rank badge
    icons on the Strength Ranking screen. They do not appear in any player
    name and can be safely stripped unconditionally.

    Args:
        name: Player name string potentially containing Thai characters.

    Returns:
        Name with all Thai characters removed.
    """
    return _THAI_RE.sub("", name).strip()


def strip_bare_tags(name: str) -> str:
    """
    Removes bare alliance tag abbreviations that appear without brackets.

    OCR sometimes returns [PoWr] with the brackets dropped in two ways:
    1. As a standalone spaced token: "PoWr SirBucksALot"
    2. Directly concatenated: "PoWrSirBucksALot" (no space)

    We handle both cases:
    - Token pass: split by spaces, remove tokens matching the tag pattern
    - Prefix pass: strip a leading tag-like prefix even when concatenated

    Tag pattern: 2-6 chars with at least one internal uppercase letter
    (e.g. PoWr, CoRe, TaGs). Length limit prevents matching real names.

    Args:
        name: Player name string potentially containing a bare tag abbreviation.

    Returns:
        Name with detected alliance abbreviation tokens and prefixes removed.

    Examples:
        strip_bare_tags("PoWr SirBucksALot")    → "SirBucksALot"
        strip_bare_tags("PoWrSirBucksALot")     → "SirBucksALot"
        strip_bare_tags("CoRe PlayerName")       → "PlayerName"
        strip_bare_tags("SirBucksALot")          → "SirBucksALot"  (unchanged)
    """
    # Pass 1: remove space-separated tokens that look like tags, but only if
    # there are other tokens remaining — avoids wiping out a player name that
    # happens to match the tag pattern (e.g. "SayTin", "CoNor").
    tokens = name.split()
    non_tags = [t for t in tokens if not _looks_like_tag(t)]
    if non_tags:
        name = " ".join(non_tags).strip()

    return name.strip()


def strip_alliance_suffixes(name: str) -> str:
    """
    Removes known alliance display names that appear as plain text suffixes.

    Some alliances display their full name as a subtitle beneath the player
    name. Vision API reads this as part of the same text block, producing
    strings like "SirBucksALot Pantheon of Wrath". The bracket tag [PoWr]
    is stripped by strip_alliance_tag, but the display name survives.

    Entries in _ALLIANCE_NAME_SUFFIXES are matched case-insensitively and
    removed wherever they appear in the string (not just at the end).

    To add a new alliance name, append it to _ALLIANCE_NAME_SUFFIXES at the
    top of this module — no code changes needed here.

    Args:
        name: Player name string potentially containing an alliance display name.

    Returns:
        Name with all known alliance display suffixes removed.

    Example:
        strip_alliance_suffixes("SirBucksALot Pantheon of Wrath") → "SirBucksALot"
    """
    for pattern in _ALLIANCE_SUFFIX_RES:
        name = pattern.sub("", name)
    return name.strip()


def clean_player_name(raw_name: str) -> str:
    """
    Full name cleaning pipeline in order of operation.

    Applies all cleaning steps in sequence:
    1. Strip alliance tags     (e.g. [PoWr])
    2. Strip bare tag tokens   (e.g. PoWr without brackets)
    3. Strip alliance suffixes (e.g. "Pantheon of Wrath")
    4. Strip leading rank      (e.g. "48 ")
    5. Strip R-badge tokens    (e.g. R4)
    6. Strip Thai characters   (OCR noise from rank badge icons)
    7. Strip OCR noise         (stray symbols)
    8. Collapse whitespace

    Args:
        raw_name: Unprocessed name string direct from OCR output.

    Returns:
        Clean player name suitable for storage and display.

    Examples:
        clean_player_name("48 R4 [PoWr] ShodiWarmic") → "ShodiWarmic"
        clean_player_name("[PoWr] SirBucksALot Pantheon of Wrath") → "SirBucksALot"
    """
    # Truncate at the first open bracket when the name precedes the tag.
    # e.g. "SirBucksALot [PoWr] Pantheon of Wrath" → "SirBucksALot"
    # Only apply if the pre-bracket content is a plausible name (not just
    # rank/badge noise like "48 R4"). Fall through to strip_alliance_tag
    # for cases where the tag comes before the name.
    _before_bracket = raw_name.split('[')[0].strip()
    _after_noise = re.sub(r'^[\d\s\.R1-5]+', '', _before_bracket).strip()
    name = _before_bracket if _after_noise else raw_name
    name = strip_bare_tags(name)
    name = strip_alliance_suffixes(name)
    name = strip_leading_rank(name)
    name = strip_rbadge_artefacts(name)
    name = strip_thai_characters(name)
    name = strip_ocr_noise(name)
    name = collapse_whitespace(name)
    return name


def is_numeric_token(s: str) -> bool:
    """
    Returns True if a string token represents a pure number (with optional commas).

    Used during row parsing to identify which token in a row is the score.

    Args:
        s: A single whitespace-stripped token string.

    Returns:
        True if the token is numeric (digits and commas only).

    Examples:
        is_numeric_token("161,528,090") → True
        is_numeric_token("SirBucksALot") → False
        is_numeric_token("R4") → False
    """
    return bool(_SCORE_RE.match(s.strip()))


def parse_score(raw_score: str) -> Optional[int]:
    """
    Converts a raw score string to an integer.

    Strips commas, handles common OCR digit confusions (O→0, l→1, I→1),
    and returns None if the result cannot be converted to int so that
    malformed rows are skipped cleanly rather than raising exceptions.

    Args:
        raw_score: Score string as read from OCR (e.g. "161,528,090").

    Returns:
        Integer score value, or None if parsing fails.

    Examples:
        parse_score("161,528,090") → 161528090
        parse_score("16l,528,O90") → 161528090  (OCR correction applied)
        parse_score("Points")      → None
    """
    cleaned = raw_score.replace(",", "").replace(" ", "")
    # Correct common OCR digit substitutions
    cleaned = cleaned.replace("O", "0").replace("l", "1").replace("I", "1")
    try:
        return int(cleaned)
    except ValueError:
        return None


def normalize_day_label(raw_label: str) -> Optional[str]:
    """
    Converts an OCR-detected day tab label to a canonical output key.

    Handles abbreviations, trailing punctuation, and case variations that
    Vision API may return depending on font rendering and image quality.

    Args:
        raw_label: Day string as detected from the image (e.g. "Thur.", "FRI").

    Returns:
        Lowercase canonical day string (e.g. "thursday", "friday"),
        or None if the label is not recognised.

    Examples:
        normalize_day_label("Thur.")  → "thursday"
        normalize_day_label("FRI")    → "friday"
        normalize_day_label("Points") → None
    """
    return _DAY_LABEL_MAP.get(raw_label.strip().lower())


def is_ui_label(text: str) -> bool:
    """
    Returns True if a string matches a known UI label rather than a player name.

    Used by the extractor to reject rows that contain only UI chrome text
    (column headers, tab labels) rather than actual player data.

    Args:
        text: Cleaned string to check.

    Returns:
        True if the string is a known UI label.

    Examples:
        is_ui_label("Commander") → True
        is_ui_label("SirBucksALot") → False
    """
    return text.strip().lower() in _UI_LABELS
