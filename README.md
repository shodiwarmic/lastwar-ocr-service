# Last War OCR Service

A Python Flask microservice that accepts *Last War: Survival* ranking screenshots and returns structured player leaderboard data via a JSON API.

Designed for deployment on **Google Cloud Run** with **Google Cloud Vision** for OCR. Engineered to stay within or very close to the Vision API free tier (1,000 images/month) via image stitching and result caching.

---

## Features

- **Seven output categories:** Daily Rank (Mon–Sat), Weekly Rank, Strength Ranking (Power, Kills, Donation Daily, Donation Weekly)
- **YAML-driven screen definitions:** classification thresholds, tab positions, and row-clustering parameters live in `app/screen_definitions/` — no code changes needed to tune them
- **Stitch-first pipeline:** groups screenshots by resolution, stitches into a single tall image, runs one Vision API call per group, then splits the result back into per-image sections
- **Two-pass classification:** fast colour-sampling pre-filter (Pass 1) + OCR-assisted fallback (Pass 2) for ambiguous images
- **Structured logging:** JSON logs to stdout, automatically captured by Cloud Logging on Cloud Run
- **Offline test suite:** classification and extraction logic tested against captured OCR fixtures — Vision API never called during tests

---

## How It Works

```
POST /process-batch  (up to 100 images)
        │
        ▼
  Load & validate images
        │
        ▼
  prepare_stitched_batches()
    Group by resolution → stitch vertically with 10 px separators
    Record Y-range of each source image (ImageRegion)
        │
        ▼
  run_ocr()  ← one Vision API call per stitched group
        │
        ▼
  For each ImageRegion:
    Filter OCR blocks to region Y-range
    classify_from_ocr_text(blocks, image=stitched)
      Pass 1: colour-sample the tab bar using screen definitions
      Pass 2: OCR text markers + bounding-box colour sampling
    extract_players(blocks, screen_type, image_height)
      Score-anchored row clustering (from screen definition)
      Name cleaning & validation
        │
        ▼
  Merge results by category → return JSON
```

---

## Project Structure

```
lastwar-ocr-service/
├── app/
│   ├── __init__.py              Flask app factory
│   ├── routes.py                /process-batch and /health endpoints
│   ├── screen_definitions/      Git submodule — YAML screen definitions
│   │   ├── catalog.yaml         Ordered list of screens
│   │   ├── meta-schema.json     JSON Schema for definition files
│   │   ├── README.md            Schema reference and authoring guide
│   │   └── screens/
│   │       ├── daily_ranking.yaml
│   │       ├── weekly_ranking.yaml
│   │       └── strength_ranking.yaml
│   ├── pipeline/
│   │   ├── screen_definitions.py  Loads and caches YAML definitions
│   │   ├── classifier.py          Two-pass screenshot classification
│   │   ├── stitcher.py            Resolution grouping and vertical stitching
│   │   ├── ocr_client.py          Google Cloud Vision wrapper
│   │   └── extractor.py           OCR text → structured player data
│   ├── models/
│   │   └── schemas.py             Pydantic models: PlayerEntry, BatchResult
│   └── utils/
│       ├── logger.py              Structured JSON logger for Cloud Run
│       ├── image_utils.py         PIL helpers
│       └── text_utils.py          Regex patterns and string cleaning
├── tests/
│   ├── conftest.py                Shared fixtures and synthetic block builders
│   ├── fixtures/
│   │   └── ocr_responses/         Captured Vision API JSON fixtures (git-ignored)
│   ├── test_classifier.py
│   ├── test_extractor.py
│   └── test_routes.py
├── tools/
│   └── capture_ocr_fixture.py    CLI: capture real OCR responses as test fixtures
├── Dockerfile
├── requirements.txt
└── main.py                        Gunicorn entrypoint
```

---

## API Reference

### `POST /process-batch`

Accepts a batch of screenshots and returns extracted player data grouped by category.

The request and response are **wire contract v1**, whose canonical text is the
screen-definitions README (Consumer Contract → Wire contract v1).

**Request**
```
Content-Type:   multipart/form-data
images          1–100 image files
category        optional — skip classification and read every frame as this category
schema_version  optional — the contract version the caller will parse; absent means 1
```

**Response 200** — a `{schema_version, results, diagnostics}` envelope:
```json
{
  "schema_version": 1,
  "results": {
    "monday":           [{"player_name": "Charlie9042",      "score": 38686463}],
    "friday":           [{"player_name": "SirBucksALot",     "score": 45635206}],
    "power":            [{"player_name": "MOJO DUDE",        "score": 218478394}],
    "donation_weekly":  [{"player_name": "CaptTrickster727", "score": 28300}]
  },
  "diagnostics": {
    "schema_version": 1,
    "engine": "cloud_vision",
    "image_count": 1, "batch_count": 1, "category_override": null,
    "batches":  [{"batch_index": 0, "stitched_size": [1080, 2400],
                  "source_images": ["IMG_4358.png"], "cache_hit": false}],
    "sections": [{"image": "IMG_4358.png", "batch_index": 0, "y_range": [483, 2900],
                  "category": "friday", "confidence": 0.95, "method": "day_color_saturation",
                  "players_found": 8, "cache_hit": false}]
  }
}
```

Player data lives under `results` (only categories with data appear). `diagnostics` is a
lightweight, structured classification trace — per source image: which `category` it became,
the `confidence`, and the `method` (`day_color_saturation`, `day_text_fallback`,
`category_override`, `weekly_marker`, `strength_tab`, `alliance_contribution_tab`,
`unclassified`); `note` (`no_ocr_blocks` / `classification_failed` / `no_players` /
`ocr_failed`) flags sections that produced no players. `null` fields are omitted. The empty
case is still `200` with `{"results": {}, "diagnostics": {...}, "warning": "..."}`; input
validation failures return `4xx {"error": "..."}`.

| Category | Screen |
|---|---|
| `monday`–`saturday` | Daily Rank — VS points for that day |
| `weekly` | Weekly Rank — 7-day cumulative total |
| `power` | Strength Ranking — Power tab |
| `kills` | Strength Ranking — Kills tab |
| `donation_daily` | Strength Ranking — Donation tab, Daily sub-tab |
| `donation_weekly` | Strength Ranking — Donation tab, Weekly sub-tab |

**Response 400** — missing or invalid input (no images, too many images, non-image files), an
unknown `category` (`code: "category_not_supported"`), or an unsupported `schema_version`
(`code: "schema_not_supported"`)  
**Response 500** — internal processing error

---

### `GET /health`

Liveness and capabilities: `{"status": "ok", "version": "v1.0.0", "commit": "<sha>",
"schema_versions": [1], "categories": [...]}`. A caller checks it before relying on a category
or a contract version. (Cloud Run's startup probe is a TCP check, not this route.)

---

## Local Development

### Prerequisites
- Python 3.12+
- Google Cloud SDK (`gcloud`) installed and authenticated
- A Google Cloud project with the Vision API enabled

### Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/shodiwarmic/lastwar-ocr-service.git
cd lastwar-ocr-service

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Cloud (for Vision API access)
gcloud auth application-default login

# Run locally
python main.py
```

The service will be available at `http://localhost:8080`.

> **Submodule note:** If you cloned without `--recurse-submodules`, initialise it with:
> ```bash
> git submodule update --init --recursive
> ```

---

## Capturing Test Fixtures

Before running tests against real screenshot data, capture OCR responses from your sample images. This calls the Vision API **once per image** and saves the results as JSON fixtures for all future test runs.

```bash
# Capture all images in a directory
python tools/capture_ocr_fixture.py /path/to/your/screenshots/

# Capture a single image
python tools/capture_ocr_fixture.py /path/to/screenshot.png

# Preview what would be captured (no API calls)
python tools/capture_ocr_fixture.py /path/to/screenshots/ --dry-run
```

Fixtures are saved to `tests/fixtures/ocr_responses/`. The filename prefix is used to infer the expected output category — include the category in the screenshot filename before capturing:

| Filename includes | Expected category |
|---|---|
| `Monday`, `Tuesday`, … `Saturday` | `monday` … `saturday` |
| `Weekly` | `weekly` |
| `Power` or `Strength` | `power` |
| `Kills` | `kills` |
| `Donation_Daily` | `donation_daily` |
| `Donation_Weekly` | `donation_weekly` |

The `.gitignore` excludes fixture files by default — remove that exclusion if you want them committed.

---

## Running Tests

```bash
# Run all tests (fixture-dependent tests auto-skip if fixtures not captured yet)
pytest

# Run with verbose output
pytest -v

# Run only unit tests (no fixtures required)
pytest tests/test_classifier.py tests/test_extractor.py -k "not RealFixture"

# Run only fixture-based tests
pytest -k "RealFixture"
```

---

## Deployment to Cloud Run

**The service is never public.** Deploy it with `--no-allow-unauthenticated` and grant
`roles/run.invoker` to the app's service account alone: every request spends Vision units
billed to you, and `/health` names the exact release and commit it runs. The identities,
their grants and the full one-time setup are in [`docs/GCP_PERMISSIONS.md`](docs/GCP_PERMISSIONS.md).

### Self-hosting the cloud image

Run a **released** image (`ghcr.io/shodiwarmic/lastwar-ocr-service:vX.Y.Z`; see
[`docs/RELEASING.md`](docs/RELEASING.md) for what each tag means). Cloud Run cannot pull from
GHCR directly, so put the image in an Artifact Registry repository in your service's region
(pulls within a region are free). Either let a **remote repository** proxy GHCR:

```bash
gcloud artifacts repositories create ghcr-proxy --location us-east1 \
  --repository-format docker --mode remote-repository --remote-docker-repo https://ghcr.io
# image: us-east1-docker.pkg.dev/$PROJECT/ghcr-proxy/shodiwarmic/lastwar-ocr-service:vX.Y.Z
```

or copy the release into a standard repository:

```bash
docker pull ghcr.io/shodiwarmic/lastwar-ocr-service:vX.Y.Z
docker tag  ghcr.io/shodiwarmic/lastwar-ocr-service:vX.Y.Z \
            us-east1-docker.pkg.dev/$PROJECT/lastwar-ocr/lastwar-ocr-service:vX.Y.Z
docker push us-east1-docker.pkg.dev/$PROJECT/lastwar-ocr/lastwar-ocr-service:vX.Y.Z
```

Then deploy it, as a dedicated runtime account that can call Vision and nothing else:

```bash
export PROJECT=your-gcp-project-id REGION=us-east1
gcloud run deploy lastwar-ocr-service --region $REGION \
  --image <the image above> \
  --service-account lastwar-ocr-runtime@$PROJECT.iam.gserviceaccount.com \
  --no-allow-unauthenticated \
  --concurrency 1 --max-instances 3 --memory 1Gi --cpu 1 --timeout 120
gcloud run services add-iam-policy-binding lastwar-ocr-service --region $REGION \
  --member serviceAccount:<the app's account> --role roles/run.invoker
```

`--concurrency 1`: gunicorn runs one sync worker, so a second request on an instance only
waits, against the 120 s timeout. `--max-instances 3` bounds both parallelism and spend.

**Keep a few versions, not all.** A cleanup policy on your repository stops old images piling
up. This project keeps the three most recent ([`deploy/artifact-registry-cleanup.json`](deploy/artifact-registry-cleanup.json)):

```bash
gcloud artifacts repositories set-cleanup-policies lastwar-ocr --location $REGION \
  --policy deploy/artifact-registry-cleanup.json --no-dry-run
```

Three is our default, not a requirement: a deleted image breaks rolling back to the revision
that used it, so keep as many as you might roll back across. GHCR keeps every release.

**Set a billing budget alert — required.** Vision, Translation and the archive bucket all bill
the same account; [`docs/GCP_PERMISSIONS.md`](docs/GCP_PERMISSIONS.md#billing-a-budget-alert-is-required)
has the steps.

### This project's production

Production is deployed by the release workflow, never by hand: a `vX.Y.Z` tag publishes the
image to GHCR and to a private Artifact Registry repository with the same digest, deploys it
as a no-traffic revision, smoke-tests it, and then moves traffic. See
[`docs/RELEASING.md`](docs/RELEASING.md).

---

## Cost Considerations

| Component | Free Tier | Expected Usage |
|---|---|---|
| Cloud Run | 2M requests/month, 180K vCPU-seconds | Effectively free for alliance-scale usage |
| Vision API | 1,000 units/month | ~3–4 units per full batch (with stitching) |

With stitching enabled, a 10-image batch uses ~3–4 Vision API units. The free tier supports roughly **250–330 full batches per month** before any charges apply.

---

## Troubleshooting

**Classification failures** — Check Cloud Logging for `"Pass 2"` log entries, which indicate images where Pass 1 colour-sampling was inconclusive and OCR fallback was used. If both passes fail the entry will have `"OCR classification failed"`.

**OCR quality issues** — Ensure screenshots are taken at full device resolution. PNG is preferred over JPEG to avoid compression artefacts on small text.

**Cold start latency** — The Vision API client initialises on the first request (~300 ms). If cold start latency is unacceptable, set `--min-instances 1` in your Cloud Run deployment (incurs cost).

---

## Extending

### Tuning an existing screen (colour thresholds, tab positions, clustering bands)
Edit the relevant YAML file in `app/screen_definitions/screens/` and increment its `version`. See [`app/screen_definitions/README.md`](app/screen_definitions/README.md) for the full field reference.

### Adding a new screen type
1. Add a YAML definition in `app/screen_definitions/screens/` following the schema
2. Register it in `app/screen_definitions/catalog.yaml` with an appropriate priority
3. Capture an OCR fixture with `tools/capture_ocr_fixture.py` and name it to include the new category keyword
4. Run `pytest` — the new fixture is auto-discovered by all three test files

Tuning needs no Python change, and neither does a screen that reuses an existing extraction strategy (`score_anchored`, `column_scoped`) **when the caller sends its category**. Auto-detection is different: the classifier is a fixed cascade over the five ranking screen families, not a loop over the catalog, so a screen the service should recognise unprompted needs a classifier branch. The post-event mails are override-only for that reason.

### Adding a new alliance display name to strip
Add it to `_ALLIANCE_NAME_SUFFIXES` in [`app/utils/text_utils.py`](app/utils/text_utils.py).
