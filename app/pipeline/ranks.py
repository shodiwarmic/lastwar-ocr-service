"""
app/pipeline/ranks.py

The rank column as a row-completeness checksum.

Every ranked screen numbers its rows without gaps, so the rank beside each row
says what the extraction should have found: a gap means a row was dropped, a
duplicate means one was read twice, and a frame whose ranks do not run on from
the previous frame's means a frame is missing. The service reads each row's
rank, infers the ones it could not read where their position settles them, and
reports the checksum per section. It never repairs a value: a misread rank is
reported as read, and the caller decides.

A rank comes from one of three places, recorded per row:
    column   the 1–999 integer token in the definition's `rank` column beside
             the row
    name     the leading digits OCR merged into the row's first name token
             (`48KeldaVornic`), when no rank token stands alone
    inferred not read, but the read ranks either side of it agree on exactly
             one value for its position
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

# How far from the score anchor's centre a rank token may sit, as a fraction of
# image height. Measured on the 169 ranking recordings (2026-09-30): rank digits
# sit up to 0.010 H below the score on Strength and Alliance Contribution, and
# within 0.0033 H on the day and weekly screens; the next row is ~0.09 H away.
RANK_Y_TOLERANCE_FRACTION = 0.012

MAX_RANK = 999

_RANK_TOKEN_RE = re.compile(r"\d{1,3}")
_NAME_LEADING_RANK_RE = re.compile(r"^(\d{1,3})(?=[^\d,.\s])")


def rank_column(defn) -> Optional[tuple[float, float]]:
    """The definition's `rank` column as (x_min, x_max) fractions, or None."""
    if defn is None:
        return None
    for col in defn.columns:
        if col.type == "rank":
            return col.x_min, col.x_max
    return None


def _x_centre(block: dict) -> float:
    return float(block.get("avg_x", 0.0))


def _left_x(block: dict) -> float:
    """Left edge from the bbox (a dict from a recording, a proto from live
    Vision), else the centre."""
    bbox = block.get("bbox")
    if isinstance(bbox, dict) and bbox.get("vertices"):
        return float(min(v.get("x", 0) for v in bbox["vertices"]))
    vertices = getattr(bbox, "vertices", None)
    if vertices:
        return float(min(v.x for v in vertices))
    return _x_centre(block)


def as_rank(text: str) -> Optional[int]:
    """The integer a standalone rank token reads as, or None."""
    text = text.strip().rstrip(".")
    if not _RANK_TOKEN_RE.fullmatch(text):
        return None
    value = int(text)
    return value if 1 <= value <= MAX_RANK else None


def find_rank_token(
    blocks: list[dict],
    column: tuple[float, float],
    image_width: int,
    y_from: float,
    y_to: float,
    y_target: float,
) -> Optional[int]:
    """
    The rank token in `column` whose centre lies in [y_from, y_to], nearest to
    y_target. Tokens belong to the column their x-centre falls in.
    """
    x_min, x_max = column[0] * image_width, column[1] * image_width
    best: Optional[tuple[float, int]] = None
    for b in blocks:
        if not (x_min <= _x_centre(b) <= x_max and y_from <= b["avg_y"] <= y_to):
            continue
        value = as_rank(b["text"])
        if value is None:
            continue
        distance = abs(b["avg_y"] - y_target)
        if best is None or distance < best[0]:
            best = (distance, value)
    return best[1] if best else None


def rank_from_name_token(
    block: Optional[dict],
    column: tuple[float, float],
    image_width: int,
) -> Optional[int]:
    """
    The leading digits OCR merged into a name token (`48KeldaVornic` → 48),
    when the token starts inside the rank column — a name that merely begins
    with digits starts in the name column and is left alone.
    """
    if block is None or _left_x(block) > column[1] * image_width:
        return None
    m = _NAME_LEADING_RANK_RE.match(block["text"].strip())
    if not m:
        return None
    value = int(m.group(1))
    return value if 1 <= value <= MAX_RANK else None


def apply_checksum(rows: list[dict], pinned_row: bool = True) -> dict:
    """
    Infers unread ranks where their position is unambiguous, then computes the
    section's checksum.

    `rows` are the section's emitted rows in order, each a dict with `rank`
    (int or None), `rank_source` ("column" | "name" | None) and `score`. Rows
    whose rank is inferred get `rank` and `rank_source = "inferred"`.

    The ranking screens pin the viewer's own row under the list (rank 23 after
    ranks 1–7). A last row whose read rank is not the one its position calls
    for is taken to be that row: it keeps its rank, and is left out of the
    inference, the gaps, duplicates, order and score-order checks, and
    reported as `pinned`. Screens without one (the mails) pass
    pinned_row=False.

    Returns {"ranks": {...}, "order_violations": [...]}; "ranks" is None when
    no rank was read at all.
    """
    read = [(i, r["rank"]) for i, r in enumerate(rows) if r["rank"] is not None]

    expected_start = None
    pinned = None
    if read:
        offsets = Counter(rank - i for i, rank in read)
        top = offsets.most_common()
        # The modal rank − index: the rank the section's first row should
        # carry if nothing is missing. Ties go to the smaller offset.
        best = max(count for _, count in top)
        expected_start = min(off for off, count in top if count == best)

        last = len(rows) - 1
        if (
            pinned_row and len(rows) >= 3 and best >= 2
            and rows[last]["rank"] is not None
            and rows[last]["rank"] != expected_start + last
        ):
            pinned = last
            read = [(i, r) for i, r in read if i != pinned]

        for i, row in enumerate(rows):
            if row["rank"] is not None or i == pinned:
                continue
            before = next(((j, r) for j, r in reversed(read) if j < i), None)
            after = next(((j, r) for j, r in read if j > i), None)
            candidates = set()
            if before:
                candidates.add(before[1] + (i - before[0]))
            if after:
                candidates.add(after[1] - (after[0] - i))
            if len(candidates) == 1:
                value = candidates.pop()
                if 1 <= value <= MAX_RANK:
                    row["rank"] = value
                    row["rank_source"] = "inferred"

    listed = [r for i, r in enumerate(rows) if i != pinned]

    ranks = None
    if read or pinned is not None:
        final = [r["rank"] for r in listed if r["rank"] is not None]
        counts = Counter(final)
        ordered = sorted(counts)
        gaps = [n for a, b in zip(ordered, ordered[1:]) for n in range(a + 1, b)]
        out_of_order = [b for a, b in zip(final, final[1:]) if b < a]
        ranks = {
            "read": sum(1 for r in rows if r["rank_source"] == "column"),
            "from_name": sum(1 for r in rows if r["rank_source"] == "name"),
            "inferred": sum(1 for r in rows if r["rank_source"] == "inferred"),
            "expected_start": expected_start,
            "gaps": gaps[:50],
            "duplicates": sorted(n for n, c in counts.items() if c > 1),
            "out_of_order": out_of_order[:50],
        }
        if pinned is not None:
            ranks["pinned"] = rows[pinned]["rank"]

    # The board is sorted best first; a score above the row before it means a
    # misread value or a mis-joined row. Ties are allowed. A row whose score
    # was not read is not a violation.
    violations = []
    previous = None
    for i, r in enumerate(rows):
        if i == pinned or r.get("score_unread"):
            continue
        if previous is not None and r["score"] > previous:
            violations.append({"index": i, "rank": r["rank"], "score": r["score"],
                               "previous_score": previous})
        previous = r["score"]

    return {"ranks": ranks, "order_violations": violations}
