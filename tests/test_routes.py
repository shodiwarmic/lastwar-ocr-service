"""
tests/test_routes.py

Integration tests for the Flask HTTP endpoints.

The Vision API is mocked at the ocr_client level so no real API calls are
made. Tests verify:
    - /health returns 200
    - /process-batch rejects invalid input with 400
    - /process-batch routes images through the pipeline and returns correct JSON
    - Response structure matches the documented contract
    - Edge cases: empty batch result, all-fallback batch, mixed batch

Mocking strategy:
    unittest.mock.patch is used to replace run_ocr() with a function that
    returns a pre-built TextAnnotation-like dict (same structure as real
    fixtures). This lets us test the full pipeline end-to-end without
    touching the Vision API.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from werkzeug.datastructures import FileStorage

from tests.conftest import FIXTURE_DIR, load_fixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_png_bytes(width: int = 1080, height: int = 2400) -> bytes:
    """Creates a minimal solid-colour PNG as bytes for multipart upload tests."""
    img = Image.new("RGB", (width, height), color=(30, 40, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def png_file_storage(filename: str = "test.png", **kwargs) -> FileStorage:
    """
    Returns a Werkzeug FileStorage object for Flask test client multipart uploads.

    Flask's test client accepts FileStorage objects directly in the data dict,
    which avoids Werkzeug trying to interpret a filename string as a file path
    to open on disk.
    """
    return FileStorage(
        stream=io.BytesIO(make_png_bytes(**kwargs)),
        filename=filename,
        content_type="image/png",
    )


def fixture_annotation(fixture_name: str):
    """
    Loads a real OCR fixture and returns a mock annotation object whose
    extract_text_blocks() output matches the fixture's text_blocks.
    """
    try:
        data = load_fixture(fixture_name)
        return data["annotation"], data["text_blocks"]
    except FileNotFoundError:
        return None, None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        response = client.get("/health")
        data = response.get_json()
        assert data == {"status": "ok"}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestProcessBatchValidation:

    def test_missing_images_returns_400(self, client):
        response = client.post("/process-batch")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_empty_images_list_returns_400(self, client):
        response = client.post(
            "/process-batch",
            content_type="multipart/form-data",
            data={},
        )
        assert response.status_code == 400

    def test_non_image_file_is_skipped_gracefully(self, client):
        """A text file uploaded as 'images' should be skipped, returning 400
        since no valid images remain after filtering."""
        bad_file = FileStorage(
            stream=io.BytesIO(b"not an image"),
            filename="bad.txt",
            content_type="text/plain",
        )
        response = client.post(
            "/process-batch",
            content_type="multipart/form-data",
            data={"images": bad_file},
        )
        assert response.status_code == 400

    def test_too_many_images_returns_400(self, client):
        files = [png_file_storage(f"img{i}.png") for i in range(101)]
        response = client.post(
            "/process-batch",
            content_type="multipart/form-data",
            data={"images": files},
        )
        assert response.status_code == 400
        assert "Too many images" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# Successful processing (mocked OCR)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_route_cache():
    """Clear the in-memory result cache before and after each test."""
    import app.routes as routes_module
    routes_module._result_cache.clear()
    yield
    routes_module._result_cache.clear()


class TestProcessBatchMocked:

    @patch("app.routes.classify_from_ocr_text")
    @patch("app.routes.run_ocr")
    @patch("app.routes.extract_text_blocks")
    @patch("app.routes.extract_players")
    def test_returns_correct_category_key(
        self,
        mock_extract,
        mock_text_blocks,
        mock_ocr,
        mock_classify,
        client,
    ):
        from app.models.schemas import PlayerEntry

        mock_classify.return_value = ("friday", 0.95)
        mock_ocr.return_value = (MagicMock(), "hash_abc")
        mock_text_blocks.return_value = [{"text": "x", "bbox": {}, "avg_x": 100.0, "avg_y": 1200.0}]
        mock_extract.return_value = [
            PlayerEntry(player_name="SirBucksALot", score=45_635_206)
        ]

        response = client.post(
            "/process-batch",
            content_type="multipart/form-data",
            data={"images": png_file_storage("friday.png")},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "friday" in data
        assert data["friday"][0]["player_name"] == "SirBucksALot"
        assert data["friday"][0]["score"] == 45_635_206

    @patch("app.routes.classify_from_ocr_text")
    @patch("app.routes.run_ocr")
    @patch("app.routes.extract_text_blocks")
    @patch("app.routes.extract_players")
    def test_empty_result_returns_warning(
        self,
        mock_extract,
        mock_text_blocks,
        mock_ocr,
        mock_classify,
        client,
    ):
        mock_classify.return_value = ("friday", 0.95)
        mock_ocr.return_value = (MagicMock(), "hash_xyz")
        mock_text_blocks.return_value = [{"text": "x", "bbox": {}, "avg_x": 100.0, "avg_y": 1200.0}]
        mock_extract.return_value = []

        response = client.post(
            "/process-batch",
            content_type="multipart/form-data",
            data={"images": png_file_storage("friday.png")},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "warning" in data

    @patch("app.routes.classify_from_ocr_text")
    @patch("app.routes.run_ocr")
    @patch("app.routes.extract_text_blocks")
    @patch("app.routes.extract_players")
    def test_multiple_categories_in_one_batch(
        self,
        mock_extract,
        mock_text_blocks,
        mock_ocr,
        mock_classify,
        client,
    ):
        from app.models.schemas import PlayerEntry

        call_count = 0

        def classify_side_effect(text_blocks, image=None, filename=""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("friday", 0.95)
            return ("power", 0.92)

        def extract_side_effect(blocks, screen_type, image_height=2400, image_width=1080):
            if screen_type == "friday":
                return [PlayerEntry(player_name="SirBucksALot", score=45_635_206)]
            return [PlayerEntry(player_name="MOJO DUDE", score=218_478_394)]

        mock_classify.side_effect = classify_side_effect
        mock_ocr.return_value = (MagicMock(), "hash_multi")
        mock_text_blocks.return_value = [
            {"text": "d1", "bbox": {}, "avg_x": 100.0, "avg_y": 1200.0},
            {"text": "d2", "bbox": {}, "avg_x": 100.0, "avg_y": 3610.0},
        ]
        mock_extract.side_effect = extract_side_effect

        response = client.post(
            "/process-batch",
            content_type="multipart/form-data",
            data={"images": [
                png_file_storage("friday.png"),
                png_file_storage("power.png"),
            ]},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "friday" in data
        assert "power" in data


# ---------------------------------------------------------------------------
# Real fixture end-to-end tests — auto-discovered
# ---------------------------------------------------------------------------

_FILENAME_TO_CATEGORY = {
    "monday":    "monday",
    "tuesday":   "tuesday",
    "wednesday": "wednesday",
    "thursday":  "thursday",
    "friday":    "friday",
    "saturday":  "saturday",
    "weekly":    "weekly",
    "power":     "power",
    "strength":  "power",
}


def _infer_category(fixture_name: str):
    lower = fixture_name.lower()
    for keyword, category in _FILENAME_TO_CATEGORY.items():
        if keyword in lower:
            return category
    return None


def _discovered_fixtures():
    stems = sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))
    return stems if stems else ["__no_fixtures__"]


class TestProcessBatchRealFixtures:
    """
    Full HTTP pipeline tests using real Vision API fixture data.
    Auto-discovers all fixtures from tests/fixtures/ocr_responses/.
    Expected category is inferred from the fixture filename.
    """

    @pytest.mark.parametrize("fixture_name", _discovered_fixtures())
    def test_full_pipeline_with_real_fixture(
        self, fixture_name, client, skip_if_no_fixture
    ):
        skip_if_no_fixture(fixture_name)

        expected_category = _infer_category(fixture_name)
        if expected_category is None:
            pytest.skip(
                f"Cannot infer expected category from fixture name '{fixture_name}'. "
                f"Rename the screenshot to include a day or screen type."
            )

        fixture_data = load_fixture(fixture_name)
        text_blocks  = fixture_data["text_blocks"]
        mock_ann     = MagicMock()
        mock_ann.text = "fixture"
        mock_ann.pages = []

        # Use the real screenshot if available so colour-based classification
        # works correctly. Fall back to a synthetic PNG if not found — in that
        # case Pass 1 will be ambiguous and Pass 2 text scoring takes over.
        image_file = _real_image_or_synthetic(fixture_data.get("source_file", ""), fixture_name)

        with patch("app.routes.run_ocr", return_value=(mock_ann, fixture_data["image_hash"])):
            with patch("app.routes.extract_text_blocks", return_value=text_blocks):
                response = client.post(
                    "/process-batch",
                    content_type="multipart/form-data",
                    data={"images": image_file},
                )

        assert response.status_code == 200
        data = response.get_json()
        assert expected_category in data, (
            f"Fixture '{fixture_name}': expected category '{expected_category}' "
            f"in response keys {list(data.keys())}"
        )
        assert len(data[expected_category]) > 0


def _real_image_or_synthetic(source_file: str, fixture_name: str) -> FileStorage:
    """
    Returns a FileStorage wrapping the real screenshot if it can be found,
    otherwise returns a synthetic PNG. The real image is needed so that
    colour-based day tab classification works correctly in route tests.
    """
    from pathlib import Path

    search_dirs = [
        Path("tests/fixtures/screenshots"),
        Path.home() / "lastwar-screenshots",
        Path.home() / "Pictures",
        Path.home() / "Downloads",
    ]

    for directory in search_dirs:
        for name in [source_file, f"{fixture_name}.png"]:
            if not name:
                continue
            candidate = directory / name
            if candidate.exists():
                return FileStorage(
                    stream=io.BytesIO(candidate.read_bytes()),
                    filename=name,
                    content_type="image/png",
                )

    # Real screenshot not found — fall back to synthetic
    return png_file_storage(f"{fixture_name}.png")
