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

from tests.conftest import load_fixture


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
        files = [png_file_storage(f"img{i}.png") for i in range(25)]
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

class TestProcessBatchMocked:

    @patch("app.routes.classify_screenshot")
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
        mock_text_blocks.return_value = []
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

    @patch("app.routes.classify_screenshot")
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
        mock_text_blocks.return_value = []
        mock_extract.return_value = []

        response = client.post(
            "/process-batch",
            content_type="multipart/form-data",
            data={"images": png_file_storage("friday.png")},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "warning" in data

    @patch("app.routes.classify_screenshot")
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

        def classify_side_effect(image, filename=""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("friday", 0.95)
            return ("power", 0.92)

        def extract_side_effect(blocks, screen_type, image_height=2400):
            if screen_type == "friday":
                return [PlayerEntry(player_name="SirBucksALot", score=45_635_206)]
            return [PlayerEntry(player_name="MOJO DUDE", score=218_478_394)]

        mock_classify.side_effect = classify_side_effect
        mock_ocr.return_value = (MagicMock(), "hash_multi")
        mock_text_blocks.return_value = []
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
# Real fixture end-to-end tests (skipped if fixtures absent)
# ---------------------------------------------------------------------------

class TestProcessBatchRealFixtures:
    """
    Full pipeline tests using real Vision API fixture data.
    Run tools/capture_ocr_fixture.py first to generate fixtures.
    """

    @pytest.mark.parametrize("fixture_name,expected_category", [
        ("8851", "weekly"),
        ("8836", "friday"),
        ("8725", "power"),
    ])
    def test_full_pipeline_with_real_fixture(
        self, fixture_name, expected_category, client, skip_if_no_fixture
    ):
        skip_if_no_fixture(fixture_name)

        fixture_data = load_fixture(fixture_name)
        text_blocks  = fixture_data["text_blocks"]
        mock_ann     = MagicMock()
        mock_ann.text = "fixture"
        mock_ann.pages = []

        with patch("app.routes.run_ocr", return_value=(mock_ann, fixture_data["image_hash"])):
            with patch("app.routes.extract_text_blocks", return_value=text_blocks):
                response = client.post(
                    "/process-batch",
                    content_type="multipart/form-data",
                    data={"images": png_file_storage(f"{fixture_name}.png")},
                )

        assert response.status_code == 200
        data = response.get_json()
        assert expected_category in data, (
            f"Expected category '{expected_category}' in response keys {list(data.keys())}"
        )
        assert len(data[expected_category]) > 0
