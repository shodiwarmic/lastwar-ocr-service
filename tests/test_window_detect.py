"""
tests/test_window_detect.py

Unit tests for app/utils/window_detect.

Covers:
  - Black-border detection on synthetic letterboxed images.
  - Sanity-check rejection of implausibly-narrow detections.
  - OCR-bbox fallback returning the union rect.
  - Composite detect_game_window choice between strategies.
  - Detection on real Pixel Fold inside-landscape fixture (smoke test).
"""

from pathlib import Path
import pytest
from PIL import Image

from app.utils.window_detect import (
    WindowRect,
    detect_window_by_black_borders,
    detect_window_by_ocr_bboxes,
    detect_game_window,
    crop_to_window,
)

SCREENSHOTS = Path.home() / "lastwar-screenshots"


def _solid(width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def _letterboxed(canvas_size, window_rect, bg=(0, 0, 0), fg=(200, 100, 50)) -> Image.Image:
    """Build a synthetic letterboxed test image: a `bg`-coloured canvas with
    a `fg`-coloured rectangle at `window_rect` (left, top, right, bottom)."""
    img = _solid(*canvas_size, bg)
    win = _solid(window_rect[2] - window_rect[0], window_rect[3] - window_rect[1], fg)
    img.paste(win, (window_rect[0], window_rect[1]))
    return img


# ---------------------------------------------------------------------------
# Black-border detection
# ---------------------------------------------------------------------------

class TestBlackBorderDetection:

    def test_detects_centred_window(self):
        # Game window in centre of landscape canvas (mimics Pixel Fold center)
        img = _letterboxed((2208, 1840), (559, 0, 1650, 1840))
        rect = detect_window_by_black_borders(img)
        assert rect is not None
        assert abs(rect.left - 559) <= 1
        assert abs(rect.right - 1650) <= 1
        assert rect.top == 0
        assert rect.bottom == 1840

    def test_detects_left_aligned_window(self):
        img = _letterboxed((2208, 1840), (0, 0, 1091, 1840))
        rect = detect_window_by_black_borders(img)
        assert rect is not None
        assert rect.left == 0
        assert abs(rect.right - 1091) <= 1

    def test_detects_right_aligned_window(self):
        img = _letterboxed((2208, 1840), (1117, 0, 2208, 1840))
        rect = detect_window_by_black_borders(img)
        assert rect is not None
        assert abs(rect.left - 1117) <= 1
        assert rect.right == 2208

    def test_returns_none_when_no_letterbox(self):
        # Game fills the entire canvas — no borders to find
        img = _solid(1080, 2404, (200, 100, 50))
        assert detect_window_by_black_borders(img) is None

    def test_rejects_implausibly_narrow_detection(self):
        # Tiny content blob with huge borders should NOT be returned —
        # likely a near-black loading frame, not a real game window
        img = _letterboxed((2208, 1840), (1080, 880, 1130, 960))  # ~50x80
        assert detect_window_by_black_borders(img) is None

    def test_handles_rgba_input(self):
        img = Image.new("RGBA", (2208, 1840), (0, 0, 0, 255))
        win = Image.new("RGBA", (1091, 1840), (200, 100, 50, 255))
        img.paste(win, (559, 0))
        rect = detect_window_by_black_borders(img)
        assert rect is not None
        assert rect.width == 1091


# ---------------------------------------------------------------------------
# OCR-bbox fallback
# ---------------------------------------------------------------------------

class TestOcrBboxDetection:

    def _block(self, x0, y0, x1, y1, text="x"):
        return {
            "text": text,
            "bbox": {"vertices": [
                {"x": x0, "y": y0}, {"x": x1, "y": y0},
                {"x": x1, "y": y1}, {"x": x0, "y": y1},
            ]},
        }

    def test_unions_block_bboxes(self):
        blocks = [
            self._block(100, 100, 200, 150),
            self._block(700, 800, 900, 850),
        ]
        rect = detect_window_by_ocr_bboxes(blocks, (1000, 1000), padding_fraction=0.0)
        assert rect is not None
        assert rect.left == 100
        assert rect.top == 100
        assert rect.right == 900
        assert rect.bottom == 850

    def test_pads_by_fraction(self):
        blocks = [self._block(500, 500, 600, 600)]
        rect = detect_window_by_ocr_bboxes(blocks, (1000, 1000), padding_fraction=0.05)
        assert rect is not None
        assert rect.left == 500 - 50  # 5% of 1000 = 50
        assert rect.right == 600 + 50

    def test_returns_none_for_empty_blocks(self):
        assert detect_window_by_ocr_bboxes([], (1000, 1000)) is None

    def test_returns_none_when_blocks_span_full_image(self):
        blocks = [self._block(0, 0, 1000, 1000)]
        assert detect_window_by_ocr_bboxes(blocks, (1000, 1000), padding_fraction=0.0) is None

    def test_clamps_padding_to_image_bounds(self):
        # Padding pushes the union beyond the image — the function clamps,
        # but clamped rect == full image, so it returns None (no crop needed).
        blocks = [self._block(10, 10, 990, 990)]
        assert detect_window_by_ocr_bboxes(blocks, (1000, 1000), padding_fraction=0.10) is None
        # With smaller padding that doesn't fully consume the gap, the rect
        # is returned as the clamped union.
        blocks = [self._block(50, 50, 800, 800)]
        rect = detect_window_by_ocr_bboxes(blocks, (1000, 1000), padding_fraction=0.05)
        assert rect is not None
        assert rect.left == 0           # clamped (50 - 50 = 0)
        assert rect.right == 850        # 800 + 50


# ---------------------------------------------------------------------------
# Composite + crop helper
# ---------------------------------------------------------------------------

class TestComposite:

    def test_prefers_black_borders_when_available(self):
        img = _letterboxed((2208, 1840), (559, 0, 1650, 1840))
        rect = detect_game_window(img, text_blocks=[])
        assert rect is not None
        assert rect.width == 1091

    def test_falls_back_to_ocr_bboxes(self):
        # No clear black borders — but OCR bboxes show content is in the centre
        img = _solid(1000, 1000, (90, 90, 90))  # uniform grey, no borders
        blocks = [{
            "text": "x",
            "bbox": {"vertices": [
                {"x": 300, "y": 300}, {"x": 700, "y": 300},
                {"x": 700, "y": 700}, {"x": 300, "y": 700},
            ]},
        }]
        rect = detect_game_window(img, text_blocks=blocks)
        assert rect is not None
        assert 250 <= rect.left <= 350

    def test_returns_none_when_both_strategies_inconclusive(self):
        # Uniform image, no text blocks
        img = _solid(1000, 1000, (200, 100, 50))
        assert detect_game_window(img, text_blocks=[]) is None
        assert detect_game_window(img) is None


class TestCropToWindow:

    def test_basic_crop(self):
        img = _solid(100, 100, (255, 0, 0))
        rect = WindowRect(10, 20, 90, 80)
        cropped = crop_to_window(img, rect)
        assert cropped.size == (80, 60)

    def test_raises_on_out_of_bounds(self):
        img = _solid(100, 100, (255, 0, 0))
        with pytest.raises(ValueError):
            crop_to_window(img, WindowRect(0, 0, 200, 100))


# ---------------------------------------------------------------------------
# Real-fixture smoke test — only runs if the screenshots are present
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (SCREENSHOTS / "pixel_fold_inside_landscape").is_dir(),
    reason="lastwar-screenshots/pixel_fold_inside_landscape not present",
)
class TestRealLandscapeFixtures:

    def test_landscape_sample_yields_portrait_window(self):
        landscape_dir = SCREENSHOTS / "pixel_fold_inside_landscape"
        sample = next(iter(landscape_dir.glob("*.png")), None)
        assert sample is not None, "no landscape PNGs found"
        img = Image.open(sample)
        rect = detect_window_by_black_borders(img)
        assert rect is not None, f"expected to detect game window in {sample.name}"
        # The detected window should be portrait-oriented (h > w) — that's
        # the whole point of cropping inside-landscape captures.
        assert rect.height > rect.width, (
            f"detected window {rect.as_tuple()} is not portrait-oriented"
        )
        # And should be substantially narrower than the original landscape canvas.
        assert rect.width < img.width * 0.7
