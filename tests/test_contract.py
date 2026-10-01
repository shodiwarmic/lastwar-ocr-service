"""
tests/test_contract.py

Wire contract v1 (screen-definitions README → Consumer Contract → Wire
contract v1): /health's shape, schema_version negotiation, the coded 4xx
refusals, and the result cache key.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import SCHEMA_VERSIONS, VALID_CATEGORIES, PlayerEntry
from tests.test_routes import png_file_storage

BLOCKS = [{"text": "x", "bbox": {}, "avg_x": 100.0, "avg_y": 1200.0}]


@pytest.fixture(autouse=True)
def clear_route_cache():
    import app.routes as routes_module
    routes_module._result_cache.clear()
    yield
    routes_module._result_cache.clear()


def _post(client, **form):
    data = {"images": png_file_storage("frame.png"), **form}
    return client.post("/process-batch", content_type="multipart/form-data", data=data)


class TestHealth:

    def test_shape(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "ok"
        assert body["schema_versions"] == list(SCHEMA_VERSIONS) == [1]
        assert body["categories"] == sorted(VALID_CATEGORIES)
        assert isinstance(body["version"], str) and body["version"]
        assert isinstance(body["commit"], str) and body["commit"]

    def test_version_and_commit_default_outside_an_image(self, client):
        body = client.get("/health").get_json()
        assert (body["version"], body["commit"]) == ("dev", "unknown")


@pytest.fixture()
def mocked_pipeline():
    """OCR, classification and extraction mocked; yields the extract mock."""
    with patch("app.routes.classify_from_ocr_text", return_value=("friday", 0.95)), \
         patch("app.routes.run_ocr", return_value=(MagicMock(), "hash")), \
         patch("app.routes.extract_text_blocks", return_value=BLOCKS), \
         patch("app.routes.extract_players") as mock_extract:
        mock_extract.return_value = [PlayerEntry(player_name="A", score=5_000)]
        yield mock_extract


@pytest.mark.usefixtures("mocked_pipeline")
class TestSchemaVersion:

    def test_absent_means_v1(self, client):
        response = _post(client)
        assert response.status_code == 200
        assert response.get_json()["schema_version"] == 1

    def test_explicit_v1(self, client):
        response = _post(client, schema_version="1")
        assert response.status_code == 200
        assert response.get_json()["schema_version"] == 1

    @pytest.mark.parametrize("requested", ["2", "0", "abc", "1.0"])
    def test_unsupported_is_refused_not_downgraded(self, client, requested):
        response = _post(client, schema_version=requested)
        assert response.status_code == 400
        body = response.get_json()
        assert body["code"] == "schema_not_supported"
        assert body["supported_versions"] == [1]
        assert "error" in body

    def test_empty_result_carries_the_version_too(self, client, mocked_pipeline):
        mocked_pipeline.return_value = []
        body = _post(client).get_json()
        assert body["schema_version"] == 1
        assert body["results"] == {}

    def test_diagnostics_name_the_release(self, client):
        diagnostics = _post(client).get_json()["diagnostics"]
        assert diagnostics["service_version"] == "dev"
        assert diagnostics["service_commit"] == "unknown"


class TestCategoryRefusal:

    def test_unknown_category_is_coded(self, client):
        response = _post(client, category="not_a_category")
        assert response.status_code == 400
        body = response.get_json()
        assert body["code"] == "category_not_supported"
        assert body["category"] == "not_a_category"
        assert body["supported_categories"] == sorted(VALID_CATEGORIES)
        assert "Unknown category" in body["error"]  # the v1 string is unchanged


class TestCacheKey:

    def test_same_frames_under_a_second_category_are_read_again(self, client, mocked_pipeline):
        first = _post(client, category="siege_daily").get_json()
        second = _post(client, category="defeat_daily").get_json()
        assert list(first["results"]) == ["siege_daily"]
        assert list(second["results"]) == ["defeat_daily"]
        assert mocked_pipeline.call_count == 2
        assert not second["diagnostics"]["batches"][0]["cache_hit"]

    def test_same_frames_same_category_hit_the_cache(self, client, mocked_pipeline):
        _post(client, category="siege_daily")
        again = _post(client, category="siege_daily").get_json()
        assert mocked_pipeline.call_count == 1
        assert again["diagnostics"]["batches"][0]["cache_hit"]
