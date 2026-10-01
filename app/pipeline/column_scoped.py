"""
app/pipeline/column_scoped.py

Row extraction for screens that stack a row's values under its name, in the
same x-range — the three post-event mails. Selected by
`row_clustering.strategy: column_scoped`; the canonical description is the
screen-definitions README (Consumer Contract → column_scoped).

Why a separate strategy: the score_anchored parser assumes the score sits to
the right of the name on the same line. On the mails it does not — Alliance
Exercise prints "Total Damage: 23.36G" ~0.025 H below the name, Desert Storm a
bare integer ~0.020 H below — so its leftward filter, crash-token splitting,
neighbour re-pick and name cleaning would drop or mangle every row. Here the
name and value are told apart by *line* and *format* instead, and names come
back as read (whitespace collapsed, a leading [TAG] removed).

Stages, per section:
    1. boundaries   the header line's bottom (else chrome.top_fraction, else 0)
                    to the timestamp line's top (else chrome.bottom_fraction,
                    else the image bottom)
    2. labels       label_tokens stripped from tokens (whole tokens, or the
                    front of a token for labels ending in ':')
    3. lines        tokens in the name/score columns grouped by y; a line made
                    only of score-format tokens is a value line, any other line
                    in the name column is a name line
    4. rows         each name line, with the lowest value-line token in the
                    score column and the rank token in the rank column inside
                    its band
    5. fixed rows   elements.fixed_rows, read from above the header
    6. timestamp    elements.timestamp, normalised to YYYY-MM-DD HH:MM:SS
"""

from __future__ import annotations

import re
from typing import Optional

from app.pipeline import ranks as rank_checksum
from app.utils.text_utils import parse_score

_LEADING_TAG_RE = re.compile(r"^\s*\[[^\]]{1,10}\]\s*")
_PLAIN_RE = re.compile(r"^(?:\d{1,3}(?:,\d{3})+|\d+)$")
_SUFFIXED_RE = re.compile(r"^(\d+)(?:\.(\d+))?([KMGB])?$")
# What a suffixed value looks like even when it cannot be read: digits, at most
# one separator, an optional suffix. Such a token is a value, never a name.
_SUFFIXED_SHAPE_RE = re.compile(r"^[\dOo]+(?:[.,][\dOo]+)?[KMGBkmgb]?$")
_TIMESTAMP_RE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})\s*(\d{1,2}):(\d{2}):(\d{2})")
_WHITESPACE_RE = re.compile(r"\s+")

_MAGNITUDE = {"": 1, "K": 10**3, "M": 10**6, "G": 10**9, "B": 10**9}


# ---------------------------------------------------------------------------
# Score formats
# ---------------------------------------------------------------------------

def parse_suffixed(text: str) -> Optional[int]:
    """
    `23.36G` → 23_360_000_000. The magnitude is K/M/G/B (B = billion, as G).

    Cloud Vision reads the game's G as 6, so `4.46G` arrives as `4.466`: the
    game always prints two decimals before a suffix, so a third decimal of 6
    is read as G. Anything else without a suffix is not readable — the suffix
    was lost, and the magnitude with it (`5.20G` has come back as `5206`) —
    except a bare 0.
    """
    t = text.strip().replace("O", "0").replace("o", "0")
    m = re.fullmatch(r"(\d+\.\d{2})6", t)
    if m:
        t = m.group(1) + "G"
    m = _SUFFIXED_RE.fullmatch(t.upper())
    if not m:
        return None
    whole, frac, suffix = m.group(1), m.group(2) or "", m.group(3) or ""
    if not suffix and not (whole.strip("0") == "" and not frac):
        return None
    # Integer arithmetic: no float rounding on 35.51 × 10⁹.
    scale = _MAGNITUDE[suffix]
    value = int(whole) * scale
    if frac:
        value += int(frac) * scale // (10 ** len(frac))
    return value


def is_score_format(text: str, score_format: str) -> bool:
    """Whether a token is shaped like a value of this format — readable or not."""
    if score_format == "suffixed":
        return bool(_SUFFIXED_SHAPE_RE.fullmatch(text.strip()))
    return bool(_PLAIN_RE.fullmatch(text.strip()))


def parse_value(text: str, score_format: str) -> Optional[int]:
    if score_format == "suffixed":
        return parse_suffixed(text)
    return parse_score(text) if _PLAIN_RE.fullmatch(text.strip()) else None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _vertices(block: dict) -> list[tuple[float, float]]:
    bbox = block.get("bbox")
    if isinstance(bbox, dict) and bbox.get("vertices"):
        return [(v.get("x", 0), v.get("y", 0)) for v in bbox["vertices"]]
    if getattr(bbox, "vertices", None):
        return [(v.x, v.y) for v in bbox.vertices]
    return [(block["avg_x"], block["avg_y"])]


def _top(block: dict) -> float:
    return min(y for _, y in _vertices(block))


def _bottom(block: dict) -> float:
    return max(y for _, y in _vertices(block))


def _left(block: dict) -> float:
    return min(x for x, _ in _vertices(block))


def _right(block: dict) -> float:
    return max(x for x, _ in _vertices(block))


def _in_column(block: dict, column, width: int) -> bool:
    return column is not None and column.x_min * width <= block["avg_x"] <= column.x_max * width


def _group_lines(blocks: list[dict], tolerance: float) -> list[list[dict]]:
    """Blocks grouped by y: a block joins the current line while its centre is
    within `tolerance` of the line's mean. Lines sorted top-down, tokens by x."""
    lines: list[list[dict]] = []
    for b in sorted(blocks, key=lambda b: b["avg_y"]):
        if lines:
            line = lines[-1]
            mean = sum(t["avg_y"] for t in line) / len(line)
            if abs(b["avg_y"] - mean) <= tolerance:
                line.append(b)
                continue
        lines.append([b])
    return [sorted(line, key=lambda t: t["avg_x"]) for line in lines]


def _line_y(line: list[dict]) -> float:
    return sum(t["avg_y"] for t in line) / len(line)


def _norm(text: str) -> str:
    """Lower case, spaces collapsed, no space before a colon."""
    return _WHITESPACE_RE.sub(" ", text.lower()).replace(" :", ":").strip()


def _line_text(line: list[dict]) -> str:
    return _norm(" ".join(t["text"] for t in line))


def _join_name(tokens: list[dict], gap_px: float) -> str:
    """Tokens joined left to right, a space where the gap between boxes
    exceeds gap_px; then whitespace collapsed and a leading [TAG] removed."""
    tokens = sorted(tokens, key=_left)
    parts = [tokens[0]["text"]]
    for prev, cur in zip(tokens, tokens[1:]):
        if _left(cur) - _right(prev) > gap_px:
            parts.append(" ")
        parts.append(cur["text"])
    name = _WHITESPACE_RE.sub(" ", "".join(parts)).strip()
    stripped = _LEADING_TAG_RE.sub("", name).strip()
    return stripped or name


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _find_header(lines, signals, region, y_offset, height) -> Optional[float]:
    """The bottom edge of the first line carrying every word of a signal."""
    for line in lines:
        yf = (_line_y(line) - y_offset) / height
        if region and not (region.y_min <= yf <= region.y_max):
            continue
        words = {_norm(t["text"]) for t in line}
        for signal in signals:
            if all(w in words for w in _norm(signal).split()):
                return max(_bottom(t) for t in line)
    return None


def _find_timestamp(lines, region, y_offset, height) -> tuple[Optional[float], Optional[str]]:
    """(top edge, normalised text) of the timestamp line, or (None, None)."""
    for line in reversed(lines):
        yf = (_line_y(line) - y_offset) / height
        if region and not (region.y_min <= yf <= region.y_max):
            continue
        m = _TIMESTAMP_RE.search(" ".join(t["text"] for t in line))
        if m:
            y, mo, d, h, mi, s = (int(g) for g in m.groups())
            return min(_top(t) for t in line), f"{y:04d}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}:{s:02d}"
    return None, None


def _strip_labels(blocks: list[dict], labels: list[str]) -> list[dict]:
    """Drops tokens that are a label; strips a ':'-ended label from the front
    of a token (an engine that returns the whole line as one word)."""
    ordered = sorted(labels, key=len, reverse=True)
    out = []
    for b in blocks:
        text = b["text"].strip()
        low = _norm(text)
        if any(low == _norm(label) for label in ordered):
            continue
        for label in ordered:
            nl = _norm(label)
            if nl.endswith(":") and low.startswith(nl) and len(low) > len(nl):
                # Remove the label's characters from the original, keeping case.
                text = re.sub(r"^\s*" + r"\s*".join(map(re.escape, label.strip().split())), "",
                              text, flags=re.IGNORECASE).strip()
                break
        if text:
            out.append({**b, "text": text} if text != b["text"] else b)
    return out


def _fixed_rows(defn, blocks, header_bottom, height, gap_px, tolerance) -> list[dict]:
    """elements.fixed_rows, read from the lines above the header."""
    if header_bottom is None or not defn.elements.fixed_rows:
        return []
    above = _group_lines([b for b in blocks if b["avg_y"] < header_bottom], tolerance)
    fmt = defn.row_clustering.score_format
    rows = []
    for fr in defn.elements.fixed_rows:
        label = _norm(fr.value_label)
        value_i = next((i for i in range(len(above) - 1, -1, -1) if label in _line_text(above[i])), None)
        if value_i is None:
            continue
        value_line = _strip_labels(above[value_i], [fr.value_label] + fr.value_label.split())
        value = next((parse_value(t["text"], fmt) for t in value_line
                      if parse_value(t["text"], fmt) is not None), None)
        if value is None:
            continue
        skips = [_norm(s) for s in fr.skip_labels]
        name = None
        for line in reversed(above[:value_i]):
            if _line_y(above[value_i]) - _line_y(line) > 0.10 * height:
                break
            text = _line_text(line)
            if any(s in text for s in skips):
                continue
            if all(_PLAIN_RE.fullmatch(t["text"].strip()) for t in line):
                continue
            name = _join_name(line, gap_px)
            break
        if name:
            rows.append({"name": name, "score": value, "rank": fr.rank, "rank_source": "fixed"})
    return rows


def extract(
    text_blocks: list[dict],
    defn,
    image_height: int,
    image_width: int,
    y_offset: float = 0.0,
) -> tuple[list[dict], dict]:
    """
    Returns (rows, report). Each row: name, score, rank, rank_source
    ("column" | "fixed" | "inferred" | None), score_unread. The report may
    carry ranks, order_violations, mail_timestamp and note.
    """
    rc = defn.row_clustering
    H, W = image_height, image_width
    tolerance = max(rc.y_proximity.min_tolerance_px, rc.y_proximity.tolerance_fraction * H)
    gap_px = max(rc.min_word_gap_px, rc.word_gap_fraction * W)
    columns = {c.type: c for c in defn.columns}
    name_col, score_col, rank_col = columns.get("name"), columns.get("score"), columns.get("rank")
    report: dict = {}

    all_lines = _group_lines(text_blocks, tolerance)

    # 1. Boundaries
    header_bottom = _find_header(all_lines, defn.boundaries_header.signals,
                                 defn.boundaries_header.search_region, y_offset, H)
    ts_top, timestamp = (None, None)
    if defn.elements.timestamp is not None:
        ts_top, timestamp = _find_timestamp(all_lines, defn.elements.timestamp, y_offset, H)
    if timestamp:
        report["mail_timestamp"] = timestamp
    top = header_bottom if header_bottom is not None else y_offset + defn.chrome.top_fraction * H
    bottom = ts_top if ts_top is not None else y_offset + (1 - defn.chrome.bottom_fraction) * H

    fixed = _fixed_rows(defn, text_blocks, header_bottom, H, gap_px, tolerance)

    # 2–3. Labels, then lines within the list
    listed = [b for b in text_blocks if top < b["avg_y"] < bottom]
    listed = _strip_labels(listed, rc.label_tokens)
    in_cols = [b for b in listed if _in_column(b, name_col, W) or _in_column(b, score_col, W)]
    lines = _group_lines(in_cols, tolerance)
    value_lines, name_lines = [], []
    for line in lines:
        if all(is_score_format(t["text"], rc.score_format) for t in line):
            value_lines.append(line)
        elif any(_in_column(t, name_col, W) for t in line):
            name_lines.append(line)

    values = [t for line in value_lines for t in line if _in_column(t, score_col, W)]
    up, down = rc.column_scoped.up_band_fraction * H, rc.column_scoped.down_band_fraction * H

    # 4. Rows
    rows, used = [], set()
    for line in name_lines:
        y = _line_y(line)
        candidates = [t for t in values if y - down <= t["avg_y"] <= y + up and id(t) not in used]
        score_token = max(candidates, key=lambda t: t["avg_y"]) if candidates else None
        rank = (rank_checksum.find_rank_token(listed, (rank_col.x_min, rank_col.x_max), W,
                                              y - down, y + up, y)
                if rank_col is not None else None)
        if score_token is None and rank is None:
            continue  # artwork text, a badge: not a row
        name = _join_name([t for t in line if _in_column(t, name_col, W)], gap_px)
        if not name:
            continue
        score = parse_value(score_token["text"], rc.score_format) if score_token else None
        if score_token is not None:
            used.add(id(score_token))
        rows.append({
            "name": name,
            "score": score if score is not None else 0,
            "score_unread": score is None,
            "rank": rank,
            "rank_source": "column" if rank is not None else None,
        })

    if header_bottom is not None and not rows:
        report["note"] = "no_rows_below_header"

    checksum = rank_checksum.apply_checksum(rows, pinned_row=False)
    if checksum["ranks"] is not None:
        report["ranks"] = checksum["ranks"]
    if checksum["order_violations"]:
        report["order_violations"] = checksum["order_violations"]

    for row in fixed:
        row["score_unread"] = False
    return fixed + rows, report
