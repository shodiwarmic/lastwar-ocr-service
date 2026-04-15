# Last War OCR Service — Claude Context

## What this service does

Flask microservice deployed on Google Cloud Run. Accepts batches of *Last War: Survival* ranking screenshots (multipart POST), runs Google Cloud Vision OCR, and returns structured JSON of player names and scores.

**Single endpoint:** `POST /process-batch` — accepts `images[]` (up to 100 files), returns:
```json
{
  "friday":          [{"player_name": "ShodiWarmic",      "score": 161528090}],
  "power":           [{"player_name": "SirBucksALot",     "score": 218478394}],
  "kills":           [{"player_name": "Charlie9042",      "score": 17886167}],
  "donation_daily":  [{"player_name": "BlackIce2",        "score": 14800}],
  "donation_weekly": [{"player_name": "CaptTrickster727", "score": 28300}]
}
```
Only categories with data are included. Health check: `GET /health`.

---

## Pipeline (in order)

1. **Stitch** (`app/pipeline/stitcher.py`) — group by resolution `(width, height)`, concatenate vertically with 10 px black separators, recording each source image's Y-range as an `ImageRegion`. Recursively bisects if the stitched image exceeds 20 MB or 75 MP.
2. **OCR** (`app/pipeline/ocr_client.py`) — one `document_text_detection` call per stitched group (not `text_detection`); returns word-level blocks with bounding boxes in the stitched image's coordinate space.
3. **Classify** (`app/pipeline/classifier.py`) — per `ImageRegion`, filter OCR blocks to that Y-range, then two-pass classification:
   - Pass 1: colour-sample the tab bar using positions from the screen definition (`pre_ocr_hint`, `tabs.items[].x_hint`)
   - Pass 2 (fallback): OCR text markers from `page_signals` / `negative_signals` + bounding-box colour sampling of the active tab crop
4. **Extract** (`app/pipeline/extractor.py`) — score-anchored row clustering driven by the screen definition's `row_clustering` config: find numeric tokens ≥ `min_score`, collect name tokens within `up_band_fraction` × `image_height` above the score, reconstruct name with gap-based space insertion
5. **Clean** (`app/utils/text_utils.py`) — strip alliance tags `[PoWr]`, bare tag tokens (CamelCase, 2–6 chars with a lowercase letter), alliance display names, leading rank numbers, R-badge artefacts (R1–R5), Thai OCR noise, stray symbols

---

## Screen definitions

`app/screen_definitions/` is a git submodule (repo: `shodiwarmic/lastwar-screen-definitions`). YAML files in `screens/` drive classification thresholds, tab positions, and row-clustering parameters. No Python code changes are needed to tune them.

`app/pipeline/screen_definitions.py` loads and caches all definitions via `@lru_cache`. Key public API:
- `load_all()` — returns definitions in catalog priority order
- `get_definition(screen_id)` — look up by screen ID
- `get_definition_for_category(category)` — find the definition that owns a category (e.g. `"kills"` → `strength_ranking`)

The classifier reads colour thresholds and tab groupings entirely from the definition. The extractor reads `row_clustering.*` parameters from the definition for the given `screen_type`.

---

## Key design decisions

- **Stitch-first, classify-per-section**: stitching happens before classification; each section of the stitched image is classified independently. This means no category-based grouping before OCR, which simplifies the pipeline and halves round-trips.
- **Score-anchored clustering** (not Y-proximity): prevents alliance subtitle lines (~28 px below score) from merging into the player row. Band fractions come from the screen definition (`up_band_fraction`, `down_band_fraction`).
- **`min_score` from definition**: filters rank numbers and OCR noise. Set to 1,000 for all current screens (all real scores exceed this, including Donation point totals).
- **`_looks_like_tag` heuristic** (2–6 chars, internal uppercase, at least one lowercase): detects bare alliance abbreviations like `PoWr`, `CoRe`. Requires a lowercase letter to avoid stripping all-caps name components like `FF7`. Only strips a token when other tokens remain.
- **Vision client cached at module level**: avoids ~300 ms auth overhead per request on warm Cloud Run instances.
- **In-memory result cache** (`routes.py`): keyed on SHA-256 of each uploaded image's bytes, per-instance, non-persistent. Intentionally simple — upgrade to Cloud Memorystore if cross-instance caching is needed.
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

---

## Project structure

```
main.py                          Gunicorn entrypoint (app = create_app())
app/
  routes.py                      POST /process-batch, GET /health
  screen_definitions/            Git submodule — YAML screen definitions
    catalog.yaml                 Priority-ordered list of screens
    meta-schema.json             JSON Schema for definition files
    screens/
      daily_ranking.yaml
      weekly_ranking.yaml
      strength_ranking.yaml
  pipeline/
    screen_definitions.py        Loads and caches YAML definitions
    classifier.py                Two-pass colour + OCR classification
    stitcher.py                  Resolution grouping and vertical stitching
    ocr_client.py                Vision API wrapper, word-block extraction
    extractor.py                 Row clustering and player parsing
  utils/
    text_utils.py                Name cleaning, token detection, regex patterns
    image_utils.py               PIL helpers (crop, sample colour, convert)
    logger.py                    Structured JSON logger
  models/
    schemas.py                   Pydantic models: PlayerEntry, BatchResult
```

---

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
- **New screen type:** add a YAML definition, register it in `catalog.yaml`, capture a fixture, run `pytest`. No Python changes needed for screens using existing classifier strategies.
- **New alliance display name to strip:** add to `_ALLIANCE_NAME_SUFFIXES` in `text_utils.py`.
- **New UI label to ignore:** add to `_UI_LABELS` in `text_utils.py`.
- **Max batch size:** `MAX_IMAGES_PER_BATCH` in `routes.py`.
