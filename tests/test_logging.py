"""
tests/test_logging.py

LOG_LEVEL (default INFO; every logger used to be hard-wired to DEBUG), and the
WARNING a classified section with no players now raises.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from app.utils import logger as logger_module


@pytest.mark.parametrize("value,expected", [
    (None, logging.INFO), ("DEBUG", logging.DEBUG), ("warning", logging.WARNING),
    ("nonsense", logging.INFO),
])
def test_log_level(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("LOG_LEVEL", value)
    assert logger_module._level() == expected


def test_a_section_without_players_warns(client):
    import app.routes as routes_module
    from tests.test_routes import png_file_storage
    routes_module._result_cache.clear()
    blocks = [{"text": "x", "bbox": {}, "avg_x": 100.0, "avg_y": 1200.0}]
    with patch("app.routes.run_ocr", return_value=(MagicMock(), "h")), \
         patch("app.routes.extract_text_blocks", return_value=blocks), \
         patch("app.routes.extract_players", return_value=[]), \
         patch.object(routes_module.logger, "warning") as warn:
        client.post("/process-batch", content_type="multipart/form-data",
                    data={"images": png_file_storage("f.png"), "category": "weekly"})
    routes_module._result_cache.clear()
    messages = [c.args[0] for c in warn.call_args_list]
    assert "Section yielded no players" in messages
