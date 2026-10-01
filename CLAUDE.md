# Last War OCR Service — Claude Context

## What this service does

Flask microservice that ingests batches of *Last War: Survival* ranking screenshots (multipart POST) and returns structured JSON of player names and scores. Two deployment modes:

- **Cloud (default)** — Google Cloud Vision via OIDC on Cloud Run. Auto-detects the active screen + tab from OCR text. Built from `Dockerfile`.
- **Local sidecar** — PaddleOCR running in a self-hosted Docker container, no Cloud Vision dependency. Built from `Dockerfile.local`. Caller must supply `category` to skip auto-classification because PaddleOCR's English model can't reliably read Last War's stylised header text. The calibration findings (`LOCAL_OCR_POC.md`) are in the private fixtures repository, under `ocr-service/screenshots/ranking/`.

Engine is selected at runtime by the `OCR_ENGINE` environment variable (`cloud_vision` default, `paddleocr` for the local image). `app/pipeline/ocr_client.py` is the dispatch layer; `ocr_client_paddle.py` is the PaddleOCR implementation.

**Contract:** wire contract v1, whose canonical text is the screen-definitions README (Consumer Contract → Wire contract v1). `SCHEMA_VERSIONS = (1,)` in `schemas.py` is this side's constant; the app's is `ocrContractVersion`. Everything added since is optional within v1. A change that removes a field or changes its meaning is v2 and needs both versions served for a release.

**Main endpoint:** `POST /process-batch` — accepts `images[]` (up to 100 files), optional `category` and `schema_version` (absent → 1; anything unsupported → `400 {code: "schema_not_supported"}`), returns a `{schema_version, results, diagnostics}` envelope:
```json
{
  "schema_version": 1,
  "results": {
    "friday":          [{"player_name": "KeldaVornic",      "score": 161528090}],
    "power":           [{"player_name": "SirCoinsALot",     "score": 218478394}],
    "donation_weekly": [{"player_name": "CaptJuggler727", "score": 28300}]
  },
  "diagnostics": { "schema_version": 1, "engine": "cloud_vision", "...": "see below" }
}
```
Only categories with data appear under `results`. Rows may carry `rank`, `rank_inferred` and (mails only) `score_unread`. The empty case is still `200` with `{"results": {}, "diagnostics": {...}, "warning": "..."}`; validation failures are `4xx {"error": ...}`, with a `code` for an unknown category or an unsupported contract version. `GET /health` returns `{status, version, commit, schema_versions, categories}`; the version and commit come from `app/version.py` (baked in at build time).

### Diagnostics block

A lightweight, structured per-batch classification trace, returned alongside every `200` so a misclassification can be triaged from the archive without re-running the pipeline. The Go backend persists it verbatim as `diagnostics.json`. Built in `routes.py` from `app/models/schemas.py` models (`BatchDiagnostics` → `BatchDiagnostic` / `SectionDiagnostic`), serialised with `model_dump(exclude_none=True)` (so `null` fields are omitted):

- Top level: `schema_version`, `service_version`, `service_commit`, `engine` (`cloud_vision`|`paddleocr`, from `ocr_client.active_engine()`), `image_count`, `batch_count`, `category_override`.
- `batches[]`: `batch_index`, `stitched_size`, `source_images`, `cache_hit`.
- `sections[]` (one per source image): `image`, `batch_index`, `y_range`, `category`, `confidence`, `method`, `players_found`, `cache_hit`, `note`, and when they apply `ranks` (the rank checksum from `app/pipeline/ranks.py`), `order_violations`, `mail_timestamp`.

`method` (derived from `(category, confidence)` by `schemas.classification_method`, no classifier signature change) is one of `category_override`, `day_color_saturation`, `day_text_fallback`, `weekly_marker`, `strength_tab`, `alliance_contribution_tab`, `unclassified` — e.g. a section reading `thursday @ 0.75 / day_text_fallback` is the smoking-gun signal that colour sampling fell through to the fallback. `note` flags sections that yielded nothing: `no_ocr_blocks`, `classification_failed`, `no_players`, `ocr_failed`, `no_rows_below_header` (a mail whose list is collapsed). Heavy artifacts (stitched images, raw OCR) are intentionally **not** here — those go to the Go backend's ephemeral bucket.

---

## Pipeline (in order)

1. **Stitch** (`app/pipeline/stitcher.py`) — group by resolution `(width, height)`, concatenate vertically with 10 px black separators, recording each source image's Y-range as an `ImageRegion`. Recursively bisects if the stitched image exceeds 20 MB or 75 MP.
2. **OCR** (`app/pipeline/ocr_client.py`) — one `document_text_detection` call per stitched group (not `text_detection`); returns word-level blocks with bounding boxes in the stitched image's coordinate space.
3. **Classify** (`app/pipeline/classifier.py`) — per `ImageRegion`, filter OCR blocks to that Y-range, then two-pass classification:
   - Pass 1: colour-sample the tab bar using positions from the screen definition (`pre_ocr_hint`, `tabs.items[].x_hint`)
   - Pass 2 (fallback): OCR text markers from `page_signals` / `negative_signals` + bounding-box colour sampling of the active tab crop
4. **Extract** (`app/pipeline/extractor.py`) — by the definition's `row_clustering.strategy`:
   - `score_anchored` (the ranking screens): find numeric tokens ≥ `min_score`, collect name tokens within `up_band_fraction` × `image_height` above the score, reconstruct name with gap-based space insertion, then **clean** it (`app/utils/text_utils.py`: alliance tags `[PoWr]`, bare tag tokens, alliance display names, leading rank numbers, R-badges, Thai OCR noise, stray symbols).
   - `column_scoped` (the post-event mails, `app/pipeline/column_scoped.py`): the value sits *under* the name in the same columns, so rows are built from name lines and value lines told apart by format; names come back as read (a leading `[TAG]` removed, nothing else).
5. **Ranks** (`app/pipeline/ranks.py`) — each row's rank from the definition's `rank` column (or digits merged into the name), unread ones inferred where the neighbours settle them, and a per-section checksum. Values are never repaired.

---

## Screen definitions

`app/screen_definitions/` is a git submodule (repo: `shodiwarmic/lastwar-screen-definitions`). YAML files in `screens/` drive classification thresholds, tab positions, and row-clustering parameters. No Python code changes are needed to tune them.

`app/pipeline/screen_definitions.py` loads and caches all definitions via `@lru_cache`. Key public API:
- `load_all()` — returns definitions in catalog priority order
- `get_definition(screen_id)` — look up by screen ID
- `get_definition_for_category(category)` — find the definition that owns a category (e.g. `"kills"` → `strength_metrics`, `"siege_daily"` → `alliance_contribution`)
- `all_categories()` — every category the definitions produce. `VALID_CATEGORIES` is this: tab-group combinations for Alliance Contribution, tab items elsewhere, a top-level `category` for a screen with no tab bar.

The classifier reads colour thresholds and tab groupings entirely from the definition. The extractor reads `row_clustering.*` parameters from the definition for the given `screen_type`.

---

## Key design decisions

- **Stitch-first, classify-per-section**: stitching happens before classification; each section of the stitched image is classified independently. This means no category-based grouping before OCR, which simplifies the pipeline and halves round-trips.
- **Active day tab by saturation, not brightness** (`least_saturated` strategy): the active day tab is a desaturated near-white pill (S ≈ 0.02) against warm-grey inactive tabs (S ≈ 0.09). Brightness was unreliable — the pill's bold dark glyphs pull its sampled V to within ~0.02 of an inactive tab, a margin under OCR-bbox jitter that once misrouted a Monday screen to Thursday via the text fallback. The text fallback (`_ocr_detect_active_day_by_text`) only resolves an unambiguous single-tab crop and returns None on a full tab bar rather than guessing.
- **Score-anchored clustering** (not Y-proximity): prevents alliance subtitle lines (~28 px below score) from merging into the player row. Band fractions come from the screen definition (`up_band_fraction`, `down_band_fraction`).
- **`min_score` from definition**: filters rank numbers and OCR noise. Set to 1,000 for all current screens (all real scores exceed this, including Donation point totals).
- **`_looks_like_tag` heuristic** (3–4 chars — the game's tag length — internal uppercase, at least one lowercase): detects bare alliance abbreviations like `PoWr`, `CoRe`. Requires a lowercase letter to avoid stripping all-caps name components like `FF7`. Only strips a token when other tokens remain.
- **Vision client cached at module level**: avoids ~300 ms auth overhead per request on warm Cloud Run instances.
- **In-memory result cache** (`routes.py`): keyed on (SHA-256 of the stitched batch's bytes, category override, schema version), per-instance, non-persistent. The bytes alone once replayed one category's results for another. Intentionally simple — upgrade to Cloud Memorystore if cross-instance caching is needed.
- **Stitching reduces Vision API costs**: ~10 screenshots per alliance → ~3–4 API calls.

---

## Categories

| Key | Screen |
|---|---|
| `monday`–`saturday` | Daily Rank (VS points per day) |
| `weekly` | Weekly Rank (7-day total) |
| `power` | Strength Ranking — Power tab |
| `kills` | Strength Ranking — Kills tab |
| `donation_daily` | Strength Ranking — Donation tab, Daily sub-tab |
| `donation_weekly` | Strength Ranking — Donation tab, Weekly sub-tab |
| `mutual_assistance_daily/weekly/season` | Season Contribution — Mutual Assistance tab |
| `siege_daily/weekly/season` | Season Contribution — Siege tab |
| `rare_soil_war_daily/weekly/season` | Season Contribution — Rare Soil War tab |
| `defeat_daily/weekly/season` | Season Contribution — Defeat tab |
| `alliance_exercise` | "[Alliance Exercise] Alliance Reward" mail (Marshal's Guard and Large Sandworm) |
| `zombie_siege` | "Zombie Siege Report (Alliance)" mail |
| `desert_storm` | "[Desert Storm] Battle Results!" mail |

The mails are **override-only**: the classifier is a fixed cascade over the five ranking families (`classifier.py`), not a loop over the catalog, so it never picks a mail; the caller sends the category.

---

## Project structure

```
main.py                          Gunicorn entrypoint (app = create_app())
app/
  routes.py                      POST /process-batch, GET /health
  version.py                     Release and commit (OCR_SERVICE_VERSION / _COMMIT)
  screen_definitions/            Git submodule — YAML screen definitions
    catalog.yaml                 Priority-ordered list of screens
    meta-schema.json             JSON Schema for definition files
    screens/                     daily_ranking, weekly_ranking, strength_metrics,
                                 strength_donation, season_contribution,
                                 mail_alliance_exercise, mail_zombie_siege,
                                 mail_desert_storm (.yaml)
  pipeline/
    screen_definitions.py        Loads and caches YAML definitions; derives categories
    classifier.py                Two-pass colour + OCR classification (ranking screens)
    stitcher.py                  Window crop, resolution grouping, vertical stitching
    ocr_client.py                Vision API wrapper, word-block extraction
    ocr_client_paddle.py         PaddleOCR backend (local image)
    extractor.py                 Row clustering and player parsing (score_anchored)
    column_scoped.py             Row extraction for the mails
    ranks.py                     Rank reading, inference and the checksum
  utils/
    text_utils.py                Name cleaning, token detection, regex patterns
    image_utils.py               PIL helpers (crop, sample colour, convert)
    window_detect.py             Game-window detection (letterboxed captures)
    logger.py                    Structured JSON logger (LOG_LEVEL, default INFO)
  models/
    schemas.py                   Pydantic models, SCHEMA_VERSIONS, VALID_CATEGORIES
tools/
  capture_ocr_fixture.py         Record Cloud Vision responses
  scrub_fixture.py               Make a recording safe to publish
docs/
  GCP_PERMISSIONS.md             Identities, grants, WIF setup, key rotation, budget
  RELEASING.md                   Versions, the release procedure, rollback
```

---

## Tests and fixtures

Recorded Cloud Vision responses ("recordings") of real screens carry real member names, so the
full set and its source screenshots live in the **private** repository
`shodiwarmic/lastwar-test-fixtures` (`ocr-service/ocr_responses/`, `ocr-service/screenshots/`).
Point `LASTWAR_FIXTURES` at a clone of it; `tests/conftest.py` then searches it beside the
public `tests/fixtures/ocr_responses/`, which holds only a scrubbed set (`*-scrubbed.json`, made
by `tools/scrub_fixture.py`). Never commit an unscrubbed recording or a screenshot here.

Run pytest in a `python:3.12` container with the submodule initialised. A test that needs a
source image calls `skip_missing_image()` (reason `source image missing`);
`REQUIRE_FIXTURE_IMAGES=1`, set by CI on main and tags, turns those skips into failures.

## Releases

`vX.Y.Z` tags publish, deploy production and then move `:latest` / `:local`; a merge to `main` publishes only `:edge`. See `docs/RELEASING.md`; the identities and their grants are in `docs/GCP_PERMISSIONS.md`. `LOG_LEVEL` (default `INFO`) raises the logs on one revision: `gcloud run services update lastwar-ocr-service --region us-east1 --update-env-vars LOG_LEVEL=DEBUG`.

## Running locally

```bash
git submodule update --init --recursive   # initialise screen definitions submodule
gcloud auth application-default login     # or set GOOGLE_APPLICATION_CREDENTIALS
python main.py                            # runs on :8080
```

Production uses Gunicorn via `Dockerfile CMD`. `PORT` env var is set automatically by Cloud Run.

---

## Extending / tuning

- **Colour thresholds, tab positions, clustering bands:** edit the relevant YAML in `app/screen_definitions/screens/` — no code changes needed. See `app/screen_definitions/README.md` for the full field reference.
- **New screen type:** add a YAML definition, register it in `catalog.yaml`, capture a fixture into the private fixtures repository, run `pytest`. Its category is derived from the definition, and an existing extraction strategy needs no Python change — but auto-detecting it does, since the classifier is a fixed cascade. A screen read only by override (like the mails) needs none.
- **New alliance display name to strip:** add to `_ALLIANCE_NAME_SUFFIXES` in `text_utils.py`.
- **New UI label to ignore:** add to `_UI_LABELS` in `text_utils.py`.
- **Max batch size:** `MAX_IMAGES_PER_BATCH` in `routes.py`.
