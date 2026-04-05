# Last War OCR Service — Claude Context

## What this service does

Flask microservice deployed on Google Cloud Run. Accepts batches of *Last War: Survival* ranking screenshots (multipart POST), runs Google Cloud Vision OCR, and returns structured JSON of player names and scores.

**Single endpoint:** `POST /process-batch` — accepts `images[]` (up to 100 files), returns:
```json
{
  "friday":  [{"player_name": "ShodiWarmic", "score": 161528090}],
  "power":   [{"player_name": "SirBucksALot", "score": 218478394}]
}
```
Only categories with data are included. Health check: `GET /health`.

---

## Pipeline (in order)

1. **Classify** (`app/pipeline/classifier.py`) — two-pass:
   - Pass 1: colour-sample the tab bar at fixed fractions to detect orange active tab (fast, no API call)
   - Pass 2 (fallback): run OCR individually, then use bounding-box colour sampling of day tab crops
2. **Stitch** (`app/pipeline/stitcher.py`) — group by `(category, width, height)`, crop UI chrome from interior boundaries, concatenate vertically → 1 Vision API call per group instead of per image
3. **OCR** (`app/pipeline/ocr_client.py`) — `document_text_detection` (not `text_detection`); returns word-level blocks with bounding boxes
4. **Extract** (`app/pipeline/extractor.py`) — score-anchored row clustering: find numeric tokens ≥ 1,000, collect name tokens in a 50px-upward / 5px-downward band, reconstruct name with gap-based space insertion
5. **Clean** (`app/utils/text_utils.py`) — strip alliance tags `[PoWr]`, bare tag tokens, alliance display names, leading rank numbers, R-badge artefacts (R1–R5), Thai OCR noise, stray symbols

---

## Key design decisions

- **Score-anchored clustering** (not Y-proximity): prevents alliance subtitle lines (~28px below score) from merging into the player row. `UP_BAND=50`, `DOWN_BAND=5` px.
- **`MIN_VALID_SCORE = 1_000`**: filters rank numbers (1–100) and small OCR noise; all real game scores are well above this.
- **`_looks_like_tag` heuristic** (2–6 chars, internal uppercase): detects bare alliance abbreviations like `PoWr`, `CoRe`. Only strips a token when other tokens remain, preventing false removal of player names like `SayTin`.
- **Vision client cached at module level**: avoids ~300ms auth overhead per request on warm Cloud Run instances.
- **In-memory result cache** (`routes.py`): keyed on SHA-256 of image bytes, per-instance, non-persistent. Intentionally simple — upgrade to Cloud Memorystore if cross-instance caching is needed.
- **Stitching reduces Vision API costs**: ~10 screenshots per alliance → ~3–4 API calls.

---

## Categories

| Key | Screen |
|---|---|
| `monday`–`saturday` | Daily Rank (VS points per day) |
| `weekly` | Weekly Rank (7-day total) |
| `power` | Strength Ranking (player power) |

---

## Project structure

```
main.py                        Gunicorn entrypoint (app = create_app())
app/
  routes.py                    POST /process-batch, GET /health
  pipeline/
    classifier.py              Two-pass colour + OCR classification
    stitcher.py                Image grouping and vertical stitching
    ocr_client.py              Vision API wrapper, word-block extraction
    extractor.py               Row clustering and player parsing
  utils/
    text_utils.py              Name cleaning, token detection, regex patterns
    image_utils.py             PIL helpers (crop, sample colour, convert)
    logger.py                  Structured JSON logger
  models/
    schemas.py                 Pydantic models: PlayerEntry, BatchResult
```

---

## Running locally

```bash
gcloud auth application-default login   # or set GOOGLE_APPLICATION_CREDENTIALS
python main.py                          # runs on :8080
```

Production uses Gunicorn via `Dockerfile CMD`. `PORT` env var is set automatically by Cloud Run.

---

## Extending / tuning

- **New alliance display name to strip:** add to `_ALLIANCE_NAME_SUFFIXES` in `text_utils.py` — no other changes needed.
- **New UI label to ignore:** add to `_UI_LABELS` in `text_utils.py`.
- **Row clustering band:** `UP_BAND` / `DOWN_BAND` constants in `extractor.py`.
- **Classification colour thresholds:** `ORANGE_H_MIN/MAX`, `ORANGE_S_MIN`, `ORANGE_V_MIN` in `classifier.py`.
- **Max batch size:** `MAX_IMAGES_PER_BATCH` in `routes.py`.
