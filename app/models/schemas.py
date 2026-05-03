"""
app/models/schemas.py

Pydantic data models defining the structure of all inputs and outputs.

Using Pydantic here gives us:
- Automatic validation (scores are always ints, names are always strings)
- Clean serialisation to JSON for the API response
- A single source of truth for the data contract between pipeline stages

The final JSON response shape is driven by BatchResult.to_response_dict(),
which omits empty categories so the caller only sees days that had data.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

class ScoreCandidate(BaseModel):
    """
    One possible (name, score) interpretation of an ambiguous OCR crash token.

    Used inside PlayerEntry.candidates when a player name ending in digits was
    rendered flush against the score, causing OCR to merge them.  The caller
    should resolve ambiguity by matching player_name against its member roster.

    Attributes:
        player_name: Candidate player name for this split.
        score:       Candidate score for this split.
    """
    player_name: str
    score: int


class PlayerEntry(BaseModel):
    """
    Represents a single player row extracted from a screenshot.

    Attributes:
        player_name: Cleaned player name (heuristic best guess when ambiguous).
        score:       Numeric score (heuristic best guess when ambiguous).
        candidates:  Present only when the row contained an ambiguous name/score
                     crash.  Lists every valid split ordered smallest score first
                     (index 0 matches the heuristic player_name/score above).
                     The caller should pick the candidate whose player_name
                     matches a known member; fall back to candidates[0] otherwise.
                     Absent entirely when the row was unambiguous.
    """
    player_name: str
    score: int
    candidates: Optional[list[ScoreCandidate]] = None

    @field_validator("player_name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("player_name must not be empty")
        return v.strip()

    @field_validator("score")
    @classmethod
    def score_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("score must be a positive integer")
        return v


# ---------------------------------------------------------------------------
# Classification result (internal — not returned to caller)
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    """
    Internal model representing the output of the classifier for one image.

    Attributes:
        category:        Canonical category key (e.g. "friday", "weekly", "power").
                         None if classification failed entirely.
        confidence:      Float 0.0–1.0 confidence from the pre-OCR pass.
                         Values below the threshold trigger the OCR fallback.
        ocr_triggered:   True if the image was sent individually for OCR-assisted
                         classification rather than grouped for batch stitching.
        filename:        Original uploaded filename for logging and traceability.
        resolution:      (width, height) of the image, used as a stitching group key.
    """
    category: Optional[str] = None
    confidence: float = 0.0
    ocr_triggered: bool = False
    filename: str = ""
    resolution: tuple[int, int] = (0, 0)

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v


# ---------------------------------------------------------------------------
# Batch result (returned to caller)
# ---------------------------------------------------------------------------

# All valid output category keys
VALID_CATEGORIES = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "weekly",           # Weekly rank total
    "power",            # Strength ranking — Power tab
    "kills",            # Strength ranking — Kills tab
    "donation_daily",   # Strength ranking — Donation Daily sub-tab
    "donation_weekly",  # Strength ranking — Donation Weekly sub-tab
    # Season Contribution Ranking — 4 tabs × 3 time periods
    "mutual_assistance_daily",   "mutual_assistance_weekly",   "mutual_assistance_season",
    "siege_daily",               "siege_weekly",               "siege_season",
    "rare_soil_war_daily",       "rare_soil_war_weekly",       "rare_soil_war_season",
    "defeat_daily",              "defeat_weekly",              "defeat_season",
})

# Human-readable label for each category (used in logs and documentation)
CATEGORY_LABELS = {
    "monday":           "Daily Rank — Monday",
    "tuesday":          "Daily Rank — Tuesday",
    "wednesday":        "Daily Rank — Wednesday",
    "thursday":         "Daily Rank — Thursday",
    "friday":           "Daily Rank — Friday",
    "saturday":         "Daily Rank — Saturday",
    "weekly":           "Weekly Rank (Total)",
    "power":            "Strength Ranking — Power",
    "kills":            "Strength Ranking — Kills",
    "donation_daily":   "Strength Ranking — Donation (Daily)",
    "donation_weekly":  "Strength Ranking — Donation (Weekly)",
    "mutual_assistance_daily":   "Season Contribution — Mutual Assistance (Daily)",
    "mutual_assistance_weekly":  "Season Contribution — Mutual Assistance (Weekly)",
    "mutual_assistance_season":  "Season Contribution — Mutual Assistance (Season Total)",
    "siege_daily":               "Season Contribution — Siege (Daily)",
    "siege_weekly":              "Season Contribution — Siege (Weekly)",
    "siege_season":              "Season Contribution — Siege (Season Total)",
    "rare_soil_war_daily":       "Season Contribution — Rare Soil War (Daily)",
    "rare_soil_war_weekly":      "Season Contribution — Rare Soil War (Weekly)",
    "rare_soil_war_season":      "Season Contribution — Rare Soil War (Season Total)",
    "defeat_daily":              "Season Contribution — Defeat (Daily)",
    "defeat_weekly":             "Season Contribution — Defeat (Weekly)",
    "defeat_season":             "Season Contribution — Defeat (Season Total)",
}


class BatchResult:
    """
    Accumulates PlayerEntry lists for each category across an entire batch.

    Not a Pydantic model because we build it incrementally during processing
    rather than constructing it from a single validated input.

    Usage:
        result = BatchResult()
        result.add_entries("friday", [PlayerEntry(player_name="SirBucksALot", score=45635206)])
        response = result.to_response_dict()
    """

    def __init__(self) -> None:
        self._data: dict[str, list[PlayerEntry]] = {cat: [] for cat in VALID_CATEGORIES}

    def add_entries(self, category: str, entries: list[PlayerEntry]) -> None:
        """
        Appends a list of PlayerEntry objects to a given category bucket.

        If the category is not recognised it is logged and ignored rather than
        raising an exception, so one bad classification cannot corrupt good data.

        Args:
            category: One of the VALID_CATEGORIES keys.
            entries:  List of validated PlayerEntry objects.
        """
        if category not in VALID_CATEGORIES:
            return
        self._data[category].extend(entries)

    def to_response_dict(self) -> dict:
        """
        Serialises the accumulated results to a JSON-compatible dict.

        Empty categories are omitted from the output so the caller does not
        need to filter out days with no data.

        Returns:
            Dict mapping category keys to lists of player dicts, e.g.:
            {
                "friday": [{"player_name": "SirBucksALot", "score": 45635206}],
                "power":  [{"player_name": "MOJO DUDE",    "score": 218478394}]
            }
        """
        return {
            category: [entry.model_dump(exclude_none=True) for entry in entries]
            for category, entries in self._data.items()
            if entries
        }

    def is_empty(self) -> bool:
        """Returns True if no entries have been added to any category."""
        return all(len(entries) == 0 for entries in self._data.values())

    def category_count(self, category: str) -> int:
        """Returns the number of entries in a given category."""
        return len(self._data.get(category, []))
