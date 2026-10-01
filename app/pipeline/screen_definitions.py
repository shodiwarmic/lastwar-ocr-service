"""
app/pipeline/screen_definitions.py

Loads and provides access to screen definitions from bundled YAML files.

These definitions drive the classifier, stitcher, and extractor rather than
using hardcoded constants, so tuning a tab position, colour threshold, or
crop fraction only requires editing the YAML files in app/screen_definitions/.

Bundled definition layout (relative to this file's parent package):
    app/screen_definitions/catalog.yaml      the screens, in priority order
    app/screen_definitions/meta-schema.json  what every screen file must satisfy
    app/screen_definitions/screens/*.yaml    one file per screen

The categories this service reads are derived from the definitions
(`all_categories`), so a new screen's category needs no code change here.

All definitions are parsed once and cached via @lru_cache.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import jsonschema
import yaml

# Path to the bundled YAML files (app/screen_definitions/)
_DEFS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "screen_definitions")


# ---------------------------------------------------------------------------
# Typed dataclasses mirroring the meta-schema
# ---------------------------------------------------------------------------

@dataclass
class HsvOverride:
    h_min: float = 0.014
    h_max: float = 0.153
    s_min: float = 0.40
    v_min: float = 0.55


@dataclass
class ColorDef:
    named: str = "orange"
    hsv_override: Optional[HsvOverride] = None


@dataclass
class PreOcrHint:
    x_hint: float = 0.5
    y_hint: float = 0.2
    color: ColorDef = field(default_factory=ColorDef)
    confidence: float = 0.88


@dataclass
class NormalizedRegion:
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0


@dataclass
class BoundaryAnchor:
    signals: list[str] = field(default_factory=list)
    search_region: Optional[NormalizedRegion] = None


@dataclass
class ChromeDef:
    top_fraction: float = 0.22
    bottom_fraction: float = 0.12


@dataclass
class TabActiveIndicator:
    strategy: str = "brightest"  # "brightest", "color_fraction", or "least_saturated"
    min_gap: float = 0.04
    # `least_saturated` strategy thresholds (white-pill tab bars). The active
    # tab is the lowest-saturation tab, provided its mean S is ≤ max_saturation
    # and it leads the next-lowest by ≥ min_saturation_gap.
    max_saturation: float = 0.06
    min_saturation_gap: float = 0.03
    min_fraction: float = 0.10
    color: Optional[ColorDef] = None
    bbox_padding_fraction: float = 0.007


@dataclass
class TabItem:
    id: str = ""
    category: str = ""
    signals: list[str] = field(default_factory=list)
    x_hint: float = 0.5
    group: str = ""


@dataclass
class TabGroupConfig:
    strategy: str = "color_fraction"  # or "brightest"
    min_fraction: float = 0.10


@dataclass
class TabsDef:
    active_indicator: TabActiveIndicator = field(default_factory=TabActiveIndicator)
    items: list[TabItem] = field(default_factory=list)
    search_region: Optional[NormalizedRegion] = None
    y_hint: float = 0.20
    # Maps group id (e.g. "category", "period") to its detection config.
    # Populated only on layouts where any tab_item has a non-empty `group`
    # (e.g. alliance_contribution). When non-empty, the classifier picks
    # one winner per group and joins them with `_`.
    groups: dict = field(default_factory=dict)  # str -> TabGroupConfig


@dataclass
class ColumnDef:
    id: str = ""
    type: str = "ignore"  # "name", "score", or "ignore"
    x_min: float = 0.0
    x_max: float = 1.0


@dataclass
class ScoreAnchoredConfig:
    up_band_fraction: float = 0.021
    down_band_fraction: float = 0.002


@dataclass
class YProximityConfig:
    tolerance_fraction: float = 0.02
    min_tolerance_px: int = 20


@dataclass
class ColumnScopedConfig:
    up_band_fraction: float = 0.03
    down_band_fraction: float = 0.004


@dataclass
class RowClusteringConfig:
    strategy: str = "score_anchored"
    score_anchored: ScoreAnchoredConfig = field(default_factory=ScoreAnchoredConfig)
    y_proximity: YProximityConfig = field(default_factory=YProximityConfig)
    column_scoped: ColumnScopedConfig = field(default_factory=ColumnScopedConfig)
    min_score: int = 1000
    score_format: str = "plain"  # "plain" or "suffixed"
    label_tokens: list[str] = field(default_factory=list)
    word_gap_fraction: float = 0.015
    min_word_gap_px: int = 8


@dataclass
class FixedRowDef:
    rank: int = 1
    value_label: str = ""
    skip_labels: list[str] = field(default_factory=list)
    region: str = "above_header"


@dataclass
class ElementsDef:
    fixed_rows: list[FixedRowDef] = field(default_factory=list)
    # The timestamp line's search region, or None when the screen has none.
    timestamp: Optional[NormalizedRegion] = None


@dataclass
class ScreenDefinition:
    id: str
    version: int
    name: str
    description: str
    page_signals: list[str]
    negative_signals: list[str]
    pre_ocr_hint: Optional[PreOcrHint]
    boundaries_header: BoundaryAnchor
    boundaries_footer: BoundaryAnchor
    chrome: ChromeDef
    tabs: Optional[TabsDef]
    columns: list[ColumnDef]
    row_clustering: RowClusteringConfig
    # The wire category of a screen with no tab bar ("" when it has one).
    category: str = ""
    elements: ElementsDef = field(default_factory=ElementsDef)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_hsv_override(d: Optional[dict]) -> Optional[HsvOverride]:
    if not d:
        return None
    return HsvOverride(
        h_min=float(d.get("h_min", 0.014)),
        h_max=float(d.get("h_max", 0.153)),
        s_min=float(d.get("s_min", 0.40)),
        v_min=float(d.get("v_min", 0.55)),
    )


def _parse_color(d: Optional[dict]) -> ColorDef:
    if not d:
        return ColorDef()
    return ColorDef(
        named=d.get("named", "orange"),
        hsv_override=_parse_hsv_override(d.get("hsv_override")),
    )


def _parse_pre_ocr_hint(d: Optional[dict]) -> Optional[PreOcrHint]:
    if not d:
        return None
    return PreOcrHint(
        x_hint=float(d.get("x_hint", 0.5)),
        y_hint=float(d.get("y_hint", 0.2)),
        color=_parse_color(d.get("color")),
        confidence=float(d.get("confidence", 0.88)),
    )


def _parse_normalized_region(d: Optional[dict]) -> Optional[NormalizedRegion]:
    if not d:
        return None
    return NormalizedRegion(
        x_min=float(d.get("x_min", 0.0)),
        x_max=float(d.get("x_max", 1.0)),
        y_min=float(d.get("y_min", 0.0)),
        y_max=float(d.get("y_max", 1.0)),
    )


def _parse_boundary_anchor(d: Optional[dict]) -> BoundaryAnchor:
    if not d:
        return BoundaryAnchor()
    return BoundaryAnchor(
        signals=d.get("signals", []),
        search_region=_parse_normalized_region(d.get("search_region")),
    )


def _parse_chrome(d: Optional[dict]) -> ChromeDef:
    if not d:
        return ChromeDef()
    return ChromeDef(
        top_fraction=float(d.get("top_fraction", 0.22)),
        bottom_fraction=float(d.get("bottom_fraction", 0.12)),
    )


def _parse_tab_active_indicator(d: Optional[dict]) -> TabActiveIndicator:
    if not d:
        return TabActiveIndicator()
    return TabActiveIndicator(
        strategy=d.get("strategy", "brightest"),
        min_gap=float(d.get("min_gap", 0.04)),
        max_saturation=float(d.get("max_saturation", 0.06)),
        min_saturation_gap=float(d.get("min_saturation_gap", 0.03)),
        min_fraction=float(d.get("min_fraction", 0.10)),
        color=_parse_color(d.get("color")) if d.get("color") else None,
        bbox_padding_fraction=float(d.get("bbox_padding_fraction", 0.007)),
    )


def _parse_tabs(d: Optional[dict]) -> Optional[TabsDef]:
    if not d:
        return None
    items = [
        TabItem(
            id=item.get("id", ""),
            category=item.get("category", ""),
            signals=item.get("signals", []),
            x_hint=float(item.get("x_hint", 0.5)),
            group=item.get("group", ""),
        )
        for item in d.get("items", [])
    ]
    groups: dict = {}
    for gid, gcfg in (d.get("groups") or {}).items():
        groups[gid] = TabGroupConfig(
            strategy=gcfg.get("strategy", "color_fraction"),
            min_fraction=float(gcfg.get("min_fraction", 0.10)),
        )
    return TabsDef(
        active_indicator=_parse_tab_active_indicator(d.get("active_indicator")),
        items=items,
        search_region=_parse_normalized_region(d.get("search_region")),
        y_hint=float(d.get("y_hint", 0.20)),
        groups=groups,
    )


def _parse_columns(lst: list) -> list[ColumnDef]:
    return [
        ColumnDef(
            id=c.get("id", ""),
            type=c.get("type", "ignore"),
            x_min=float(c.get("x_min", 0.0)),
            x_max=float(c.get("x_max", 1.0)),
        )
        for c in lst
    ]


def _parse_row_clustering(d: Optional[dict]) -> RowClusteringConfig:
    if not d:
        return RowClusteringConfig()
    sa_raw = d.get("score_anchored") or {}
    yp_raw = d.get("y_proximity") or {}
    cs_raw = d.get("column_scoped") or {}
    return RowClusteringConfig(
        strategy=d.get("strategy", "score_anchored"),
        score_anchored=ScoreAnchoredConfig(
            up_band_fraction=float(sa_raw.get("up_band_fraction", 0.021)),
            down_band_fraction=float(sa_raw.get("down_band_fraction", 0.002)),
        ),
        y_proximity=YProximityConfig(
            tolerance_fraction=float(yp_raw.get("tolerance_fraction", 0.02)),
            min_tolerance_px=int(yp_raw.get("min_tolerance_px", 20)),
        ),
        column_scoped=ColumnScopedConfig(
            up_band_fraction=float(cs_raw.get("up_band_fraction", 0.03)),
            down_band_fraction=float(cs_raw.get("down_band_fraction", 0.004)),
        ),
        min_score=int(d.get("min_score", 1000)),
        score_format=d.get("score_format", "plain"),
        label_tokens=list(d.get("label_tokens") or []),
        word_gap_fraction=float(d.get("word_gap_fraction", 0.015)),
        min_word_gap_px=int(d.get("min_word_gap_px", 8)),
    )


def _parse_elements(d: Optional[dict]) -> ElementsDef:
    if not d:
        return ElementsDef()
    timestamp = d.get("timestamp")
    return ElementsDef(
        fixed_rows=[
            FixedRowDef(
                rank=int(fr.get("rank", 1)),
                value_label=fr.get("value_label", ""),
                skip_labels=list(fr.get("skip_labels") or []),
                region=fr.get("region", "above_header"),
            )
            for fr in d.get("fixed_rows") or []
        ],
        timestamp=(
            (_parse_normalized_region(timestamp.get("search_region")) or NormalizedRegion())
            if timestamp is not None else None
        ),
    )


def _parse_definition(raw: dict) -> ScreenDefinition:
    ident  = raw.get("identification") or {}
    bounds = raw.get("boundaries") or {}
    return ScreenDefinition(
        id=raw["id"],
        version=int(raw.get("version", 1)),
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        page_signals=ident.get("page_signals", []),
        negative_signals=ident.get("negative_signals", []),
        pre_ocr_hint=_parse_pre_ocr_hint(ident.get("pre_ocr_hint")),
        boundaries_header=_parse_boundary_anchor(bounds.get("header")),
        boundaries_footer=_parse_boundary_anchor(bounds.get("footer")),
        chrome=_parse_chrome(raw.get("chrome")),
        tabs=_parse_tabs(raw.get("tabs")),
        columns=_parse_columns(raw.get("columns", [])),
        row_clustering=_parse_row_clustering(raw.get("row_clustering")),
        category=raw.get("category", ""),
        elements=_parse_elements(raw.get("elements")),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_meta_schema() -> dict:
    schema_path = os.path.join(_DEFS_DIR, "meta-schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_all() -> list[ScreenDefinition]:
    """
    Loads all screen definitions in catalog priority order.

    Reads the bundled catalog.yaml to discover which YAML files to load and
    in what order, then validates and parses each one. Results are cached
    after the first call so the files are read only once per process lifetime.

    Each YAML is validated against meta-schema.json — a validation failure
    raises and aborts startup rather than silently falling back to defaults.

    Returns:
        List of ScreenDefinition objects ordered by priority (lowest first).
    """
    catalog_path = os.path.join(_DEFS_DIR, "catalog.yaml")
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    entries = sorted(
        catalog.get("screens", []),
        key=lambda e: e.get("priority", 99),
    )

    schema = _load_meta_schema()

    definitions = []
    for entry in entries:
        rel_path = entry["file"]  # e.g. "screens/strength_metrics.yaml"
        abs_path = os.path.join(_DEFS_DIR, rel_path)
        with open(abs_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        try:
            jsonschema.validate(raw, schema)
        except jsonschema.ValidationError as e:
            raise RuntimeError(
                f"Screen definition {rel_path} failed schema validation: "
                f"{e.message} (at {'/'.join(str(p) for p in e.absolute_path)})"
            ) from e
        definitions.append(_parse_definition(raw))

    return definitions


@lru_cache(maxsize=16)
def get_definition(screen_id: str) -> Optional[ScreenDefinition]:
    """Returns the definition for a specific screen ID, or None if not found."""
    for defn in load_all():
        if defn.id == screen_id:
            return defn
    return None


def categories_of(defn: ScreenDefinition) -> list[str]:
    """
    The wire categories a definition produces (screen-definitions README,
    Consumer Contract → Wire categories):

    - with `tabs.groups`: one per combination of one item from each group,
      joined with `_` in group declaration order (`siege_daily`);
    - otherwise with `tabs`: each item's `category`, falling back to its `id`;
    - with no tab bar: the top-level `category`.
    """
    if defn.tabs and defn.tabs.groups:
        combos = [""]
        for group in defn.tabs.groups:
            items = [i.category or i.id for i in defn.tabs.items if i.group == group]
            combos = [f"{c}_{item}" if c else item for c in combos for item in items]
        return combos
    if defn.tabs:
        return [item.category or item.id for item in defn.tabs.items]
    return [defn.category] if defn.category else []


@lru_cache(maxsize=1)
def _category_index() -> dict[str, ScreenDefinition]:
    index: dict[str, ScreenDefinition] = {}
    for defn in load_all():
        for category in categories_of(defn):
            index.setdefault(category, defn)
    return index


def all_categories() -> frozenset[str]:
    """Every category the loaded definitions produce."""
    return frozenset(_category_index())


def get_definition_for_category(category: str) -> Optional[ScreenDefinition]:
    """
    Returns the screen definition that owns the given output category.

    Resolved through the same derivation as all_categories(), so an Alliance
    Contribution key (`siege_daily`) finds its definition too — before, only
    tab items' own categories matched, and those keys fell back to defaults.

    Args:
        category: Category string e.g. "power", "monday", "siege_daily".

    Returns:
        The owning ScreenDefinition, or None if not found.
    """
    return _category_index().get(category)
