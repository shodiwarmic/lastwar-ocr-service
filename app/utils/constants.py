"""
app/utils/constants.py

Loads cross-consumer constants from the bundled
``app/screen_definitions/constants.yaml`` file. See that file (and the
"Shared fallback constants" section of the screen-definitions README)
for the canonical values; this module exists only to give Python code
typed accessors so consumers don't read the YAML directly.

Edit values in ``constants.yaml`` to change them — both the OCR service
and the Android scanner pick up the new values when their submodule SHA
is bumped. Don't hardcode values here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import yaml


_DEFS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "screen_definitions",
)
_CONSTANTS_FILE = os.path.join(_DEFS_DIR, "constants.yaml")


@dataclass(frozen=True)
class OrangeRgbThresholds:
    r_min: int
    g_min: int
    g_max: int
    b_max: int

    def matches(self, r: int, g: int, b: int) -> bool:
        return r > self.r_min and self.g_min <= g <= self.g_max and b < self.b_max


@dataclass(frozen=True)
class WhiteRgbThresholds:
    r_min: int
    g_min: int
    b_min: int

    def matches(self, r: int, g: int, b: int) -> bool:
        return r > self.r_min and g > self.g_min and b > self.b_min


@dataclass(frozen=True)
class HsvThresholds:
    h_min: float = 0.0
    h_max: float = 1.0
    s_min: float = 0.0
    s_max: float = 1.0
    v_min: float = 0.0
    v_max: float = 1.0


@dataclass(frozen=True)
class WindowDetectionConstants:
    near_black_max_channel: int
    border_coverage_threshold: float
    min_window_fraction: float
    sample_count: int
    bbox_padding_fraction: float


@dataclass(frozen=True)
class CrashTokenConstants:
    score_suffix_pattern: str


@lru_cache(maxsize=1)
def load_constants() -> dict:
    """Loads constants.yaml once and caches the result."""
    with open(_CONSTANTS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def orange_rgb() -> OrangeRgbThresholds:
    rgb = load_constants()["fallback_colors"]["orange"]["rgb"]
    return OrangeRgbThresholds(
        r_min=rgb["r_min"],
        g_min=rgb["g_min"],
        g_max=rgb["g_max"],
        b_max=rgb["b_max"],
    )


@lru_cache(maxsize=1)
def orange_hsv() -> HsvThresholds:
    hsv = load_constants()["fallback_colors"]["orange"]["hsv"]
    return HsvThresholds(
        h_min=hsv.get("h_min", 0.0),
        h_max=hsv.get("h_max", 1.0),
        s_min=hsv.get("s_min", 0.0),
        v_min=hsv.get("v_min", 0.0),
    )


@lru_cache(maxsize=1)
def white_rgb() -> WhiteRgbThresholds:
    rgb = load_constants()["fallback_colors"]["white"]["rgb"]
    return WhiteRgbThresholds(
        r_min=rgb["r_min"],
        g_min=rgb["g_min"],
        b_min=rgb["b_min"],
    )


@lru_cache(maxsize=1)
def window_detection() -> WindowDetectionConstants:
    wd = load_constants()["window_detection"]
    return WindowDetectionConstants(
        near_black_max_channel=int(wd["near_black_max_channel"]),
        border_coverage_threshold=float(wd["border_coverage_threshold"]),
        min_window_fraction=float(wd["min_window_fraction"]),
        sample_count=int(wd["sample_count"]),
        bbox_padding_fraction=float(wd["bbox_padding_fraction"]),
    )


@lru_cache(maxsize=1)
def crash_tokens() -> CrashTokenConstants:
    ct = load_constants()["crash_tokens"]
    return CrashTokenConstants(
        score_suffix_pattern=str(ct["score_suffix_pattern"]),
    )
