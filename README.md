# Last War OCR Service

A Python Flask microservice that accepts **Last War: Survival** ranking screenshots and returns structured player leaderboard data via a JSON API.

Designed for deployment on **Google Cloud Run** with **Google Cloud Vision** for OCR. Engineered to stay within or very close to the Vision API free tier (1,000 images/month) via image stitching and result caching.

---

## Features

- **Three screen types supported:** Daily Rank (Mon–Sat), Weekly Rank, Strength Ranking (Power)
- **Two-pass classification:** fast colour-sampling pre-filter + OCR-assisted fallback for ambiguous images
- **Resolution-aware stitching:** groups screenshots by category *and* resolution before OCR to minimise API calls
- **Structured logging:** JSON logs to stdout, automatically captured by Cloud Logging on Cloud Run
- **Offline test suite:** all classification and extraction logic tested against captured OCR fixtures — Vision API never called during tests

---

## Project Structure

```
lastwar-ocr-service/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── routes.py                # /process-batch and /health endpoints
│   ├── pipeline/
│   │   ├── classifier.py        # Two-pass screenshot classification
│   │   ├── stitcher.py          # Image grouping and vertical concatenation
│   │   ├── ocr_client.py        # Google Cloud Vision wrapper
│   │   └── extractor.py         # OCR text → structured player data
│   ├── models/
│   │   └── schemas.py           # Pydantic models: PlayerEntry, BatchResult
│   └── utils/
│       ├── logger.py            # Structured JSON logger for Cloud Run
│       ├── image_utils.py       # PIL helpers
│       └── text_utils.py        # Regex patterns and string cleaning
├── tests/
│   ├── conftest.py              # Shared fixtures and synthetic block builders
│   ├── fixtures/
│   │   └── ocr_responses/       # Captured Vision API JSON fixtures (git-ignored by default)
│   ├── test_classifier.py
│   ├── test_extractor.py
│   └── test_routes.py
├── tools/
│   └── capture_ocr_fixture.py  # CLI: capture real OCR responses as test fixtures
├── Dockerfile
├── requirements.txt
└── main.py                      # Gunicorn entrypoint
```

---

## API Reference

### `POST /process-batch`

Accepts a batch of screenshots and returns extracted player data.

**Request**
```
Content-Type: multipart/form-data
Body key: images  (one or more image files, max 20)
```

**Response 200**
```json
{
  "monday":    [{"player_name": "Charlie9042",   "score": 38686463}],
  "friday":    [{"player_name": "SirBucksALot", "score": 45635206}],
  "weekly":    [{"player_name": "SirBucksALot", "score": 161528090}],
  "power":     [{"player_name": "MOJO DUDE",    "score": 218478394}]
}
```
Only categories with extracted data are included in the response.

**Response 400** — invalid input  
**Response 500** — internal processing error

---

### `GET /health`

Cloud Run health check. Returns `{"status": "ok"}` with HTTP 200.

---

## Local Development

### Prerequisites
- Python 3.12+
- Google Cloud SDK (`gcloud`) installed and authenticated
- A Google Cloud project with the Vision API enabled

### Setup

```bash
# Clone and enter the project
git clone https://github.com/shodiwarmic/lastwar-ocr-service.git
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

Fixtures are saved to `tests/fixtures/ocr_responses/`. The `.gitignore` excludes them by default — remove that exclusion if you want fixtures committed to the repo.

---

## Running Tests

```bash
# Run all tests (fixture-dependent tests auto-skip if fixtures not yet captured)
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

### 1. Enable required APIs
```bash
gcloud services enable run.googleapis.com vision.googleapis.com cloudbuild.googleapis.com
```

### 2. Build and deploy
```bash
# Set your project ID
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1
export SERVICE_NAME=lastwar-ocr-service

# Build container image
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 512Mi \
  --timeout 120
```

### 3. Grant Vision API access
The Cloud Run service account needs the Vision API User role:
```bash
# Get the service account email
SA_EMAIL=$(gcloud run services describe $SERVICE_NAME --region $REGION \
  --format="value(spec.template.spec.serviceAccountName)")

# Grant Vision API access
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/cloudvision.user"
```

### 4. Set a billing budget alert
To protect against unexpected Vision API costs:
```bash
# Via Cloud Console: Billing → Budgets & alerts → Create budget
# Recommended: Alert at $1 spend
```

---

## Cost Considerations

| Component | Free Tier | Expected Usage |
|---|---|---|
| Cloud Run | 2M requests/month, 180K vCPU-seconds | Effectively free for alliance-scale usage |
| Vision API | 1,000 units/month | ~3–4 units per full batch (with stitching) |

With stitching enabled, a 10-image batch uses ~3–4 Vision API units. The free tier of 1,000 units supports roughly **250–330 full batches per month** before any charges apply.

---

## Troubleshooting

### Classification failures
Check Cloud Logging for entries with `jsonPayload.ocr_triggered = true`. These indicate images where colour-sampling failed and OCR fallback was used. If the fallback also fails, `jsonPayload.error` will be set.

### OCR quality issues
- Ensure screenshots are taken at full device resolution (do not resize before uploading)
- PNG format is preferred over JPEG to avoid compression artefacts on small text

### Cold start latency
Cloud Run containers start cold after periods of inactivity. The Vision API client initialises on the first request (~300ms). Subsequent requests in a warm instance are faster. If cold start latency is unacceptable, consider setting `--min-instances 1` in your Cloud Run deployment (incurs cost).

---

## Contributing / Extending

### Adding a new screen type
1. Add the new category key to `VALID_CATEGORIES` in `app/models/schemas.py`
2. Add detection logic to `classifier.py` (both `_detect_*` and `_ocr_detect_*`)
3. Add crop fractions to `stitcher.py` if the new screen has a different header height
4. Add test cases to `test_classifier.py` and capture a fixture

### Adjusting classification thresholds
- `CONFIDENCE_THRESHOLD` in `classifier.py` — lower to send more images through OCR fallback
- `ROW_CLUSTER_Y_TOLERANCE_FRACTION` in `extractor.py` — adjust if rows are being merged or split incorrectly
- Orange colour thresholds in `image_utils.is_orange()` — adjust if new device screenshots have different UI colours
