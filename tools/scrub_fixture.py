"""
tools/scrub_fixture.py

Makes a recorded OCR response safe to publish: every word that is not part of
the game's own interface is replaced by random letters, so no player or
alliance member name survives, while everything the pipeline reads stays where
it was — positions, digits, punctuation, word lengths and letter case. A name
that ends in digits keeps them, so crash tokens (`Name54323,045,000`) still
crash; a CamelCase tag-shaped word stays tag-shaped.

Every word is replaced unless it is on the allow-list below or appears in a
screen definition's signals, tab labels or label tokens. That is deliberately
stronger than replacing a list of known names: a name nobody listed (a former
member, a ticker) cannot slip through.

The same word maps to the same replacement across every file scrubbed in one
run, so a name that recurs across frames still recurs. The mapping is salted
per run (or by --seed) and never written down.

The annotation is rebuilt from the scrubbed words rather than edited, so no
symbol-level text survives either. `source_file` is kept: it is a capture
timestamp, and it lets the scrubbed copy be checked against the original
screenshot wherever the private fixtures are available.

Usage:
    python tools/scrub_fixture.py <recording.json>... --out tests/fixtures/ocr_responses
    # writes <stem>-scrubbed.json beside nothing private; review the output by eye
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Interface words that are safe and that the classifier or extractor reads.
ALLOWED = {
    "ranking", "rank", "daily", "weekly", "commander", "points", "point",
    "strength", "power", "kills", "donation", "alliance", "contribution",
    "mutual", "assistance", "siege", "rare", "soil", "war", "defeat", "season",
    "total", "your", "mon", "tues", "wed", "thur", "fri", "sat",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "mvp", "damage", "attacks", "report", "zombie", "wave", "perfect",
    "battle", "results", "individual", "desert", "storm", "exercise", "reward", "rewards",
    "mail", "r1", "r2", "r3", "r4", "r5",
    # The project's own alliance, already named in the code (text_utils).
    "powr", "pantheon", "of", "wrath",
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
# Case-boundary parts, for interface words OCR ran together ("RankingWeekly").
_PART_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")


def _definition_words() -> set[str]:
    """Every word of every screen definition's signals and labels."""
    from app.pipeline.screen_definitions import load_all
    words: set[str] = set()
    for defn in load_all():
        texts = list(defn.page_signals) + list(defn.negative_signals)
        texts += defn.boundaries_header.signals + defn.boundaries_footer.signals
        if defn.tabs:
            for item in defn.tabs.items:
                texts += item.signals
        texts += getattr(defn.row_clustering, "label_tokens", []) or []
        for t in texts:
            words.update(w.lower() for w in _WORD_RE.findall(t))
    return words


class Scrubber:
    def __init__(self, seed: str, allowed: set[str]):
        self.seed = seed
        self.allowed = allowed
        self.mapping: dict[str, str] = {}

    def _letters(self, run: str) -> str:
        rng = random.Random(hashlib.sha256((self.seed + run).encode()).digest())
        out = []
        for ch in run:
            if ch.isupper():
                out.append(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            elif ch.islower():
                out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))
            else:  # letters without case (Korean, Thai, …)
                out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))
        return "".join(out)

    def word(self, text: str) -> str:
        """Replaces each run of letters not on the allow-list; keeps the rest."""
        if text in self.mapping:
            return self.mapping[text]

        def repl(m: re.Match) -> str:
            run = m.group(0)
            parts = _PART_RE.findall(run)
            if run.lower() in self.allowed or (
                parts and "".join(parts) == run and all(p.lower() in self.allowed for p in parts)
            ):
                return run
            return self._letters(run)

        out = _WORD_RE.sub(repl, text)
        self.mapping[text] = out
        return out


def _annotation_from_blocks(blocks: list[dict], width: int, height: int) -> dict:
    """One page, one block and paragraph per word, symbols split evenly."""
    page_blocks = []
    for b in blocks:
        verts = b["bbox"]["vertices"]
        xs = [v.get("x", 0) for v in verts]
        ys = [v.get("y", 0) for v in verts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        n = max(1, len(b["text"]))
        symbols = []
        for i, ch in enumerate(b["text"]):
            sx0 = x0 + (x1 - x0) * i // n
            sx1 = x0 + (x1 - x0) * (i + 1) // n
            symbols.append({"text": ch, "bounding_box": {"vertices": [
                {"x": sx0, "y": y0}, {"x": sx1, "y": y0},
                {"x": sx1, "y": y1}, {"x": sx0, "y": y1}]}})
        box = {"vertices": verts}
        page_blocks.append({"paragraphs": [{"words": [{"symbols": symbols, "boundingBox": box}],
                                            "boundingBox": box}],
                            "boundingBox": box})
    return {"text": " ".join(b["text"] for b in blocks),
            "pages": [{"blocks": page_blocks, "width": width, "height": height}]}


def scrub(data: dict, scrubber: Scrubber) -> dict:
    blocks = [{**b, "text": scrubber.word(b["text"])} for b in data["text_blocks"]]
    return {
        "source_file": data["source_file"],
        "image_hash": data.get("image_hash", ""),
        "image_width": data["image_width"],
        "image_height": data["image_height"],
        "annotation": _annotation_from_blocks(blocks, data["image_width"], data["image_height"]),
        "text_blocks": blocks,
        "scrubbed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", default=None, help="Salt for the replacement letters (random by default).")
    args = parser.parse_args()

    scrubber = Scrubber(args.seed or secrets.token_hex(16), ALLOWED | _definition_words())
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.recordings:
        data = json.loads(path.read_text(encoding="utf-8"))
        stem = f"{path.stem}-scrubbed"
        (args.out / f"{stem}.json").write_text(
            json.dumps(scrub(data, scrubber), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{path.name} -> {stem}.json")
    kept = sorted({w for w, r in scrubber.mapping.items() if w == r and _WORD_RE.search(w)})
    print("Words kept as they were (check none is a name):", ", ".join(kept))


if __name__ == "__main__":
    main()
