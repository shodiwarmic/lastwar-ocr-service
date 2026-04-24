"""
app/pipeline/ocr_client_paddle.py

PaddleOCR backend for the local-deployment story (`OCR_ENGINE=paddleocr`).
Produces a dict-shape annotation that ``extract_text_blocks`` already
knows how to walk — see the dict-form branches in ocr_client.py.

Why not Cloud Vision?
    For users who want to run alliance-manager + this service entirely
    on their own hardware (no Google Cloud account, no per-call cost,
    no images leaving the network). Trade-off: PaddleOCR struggles with
    Last War's stylised title text (`STRENGTH RANKING`, `ALLIANCE
    CONTRIBUTION RANKING`) so the page-identification stage isn't
    reliable on its own — local mode therefore expects the caller to
    supply ``category=…`` to the route, skipping classification. Body
    OCR (player names, scores, tab labels) is comparable to Cloud
    Vision; that's enough for extraction.

This module is imported lazily by ocr_client.py only when
``OCR_ENGINE=paddleocr``. PaddleOCR + paddlepaddle are not in the
default requirements.txt — install with ``pip install -r
requirements-local.txt`` for the local-mode build.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Optional

from PIL import Image

from app.utils.image_utils import pil_to_bytes
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lazily-instantiated PaddleOCR pipeline. Loading the model takes ~10s
# and downloads weights on first run — defer until first request.
_pipeline = None
_pipeline_lock = threading.Lock()


def _get_pipeline():
    """Returns the PaddleOCR pipeline, instantiating it on first call."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        from paddleocr import PaddleOCR  # noqa: WPS433
        # `enable_mkldnn=False` works around a PaddlePaddle 3.x OneDNN bug
        # ("ConvertPirAttribute2RuntimeAttribute not support") that trips
        # on common server CPUs. See LOCAL_OCR_POC.md in lastwar-screenshots.
        lang = os.environ.get("PADDLEOCR_LANG", "en")
        _pipeline = PaddleOCR(
            lang=lang,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
        logger.info("PaddleOCR pipeline initialised", extra={"lang": lang})
        return _pipeline


def run_ocr(
    pil_image: Image.Image,
    client=None,            # accepted for signature compatibility — unused
    fmt: str = "JPEG",
) -> tuple[Optional[dict], str]:
    """
    PaddleOCR equivalent of ocr_client.run_ocr.

    Returns a (annotation_dict, image_hash) tuple. The annotation dict
    matches the structure produced by capture_ocr_fixture.py and is
    consumed by ``extract_text_blocks`` via its dict-walking branches —
    no changes needed downstream.
    """
    img_bytes = pil_to_bytes(pil_image, fmt=fmt)
    img_hash = hashlib.sha256(img_bytes).hexdigest()

    pipeline = _get_pipeline()

    import numpy as np  # noqa: WPS433 — local import keeps numpy off the
                        # critical-path for cloud-only deployments.
    img_array = np.array(pil_image.convert("RGB"))
    try:
        raw = pipeline.predict(img_array)
    except Exception as exc:
        logger.error(
            "PaddleOCR predict failed",
            extra={"image_hash": img_hash, "error": str(exc)},
        )
        return None, img_hash

    if not raw:
        return None, img_hash

    page_result = raw[0] if isinstance(raw, list) else raw
    if isinstance(page_result, dict):
        polys = page_result.get("rec_polys") or page_result.get("dt_polys") or []
        texts = page_result.get("rec_texts", [])
    else:
        # Older PaddleOCR releases used a per-line list of [poly, (text, conf)].
        polys = [item[0] for item in page_result]
        texts = [item[1][0] for item in page_result]

    # Build a dict-form "annotation" structurally similar to
    # capture_ocr_fixture.py's output — pages → blocks → paragraphs → words
    # — so extract_text_blocks walks it the same way it walks Cloud Vision
    # protos. Each PaddleOCR detection becomes one word in one paragraph in
    # one block.
    words = []
    text_pieces = []
    for poly, text in zip(polys, texts):
        if not text or not str(text).strip():
            continue
        text_clean = str(text).strip()
        text_pieces.append(text_clean)
        verts = [{"x": int(p[0]), "y": int(p[1])} for p in poly]
        words.append({
            "symbols": [{"text": text_clean, "bounding_box": {"vertices": verts}}],
            "boundingBox": {"vertices": verts},
        })

    if not words:
        logger.warning("PaddleOCR returned no usable text", extra={"image_hash": img_hash})
        return None, img_hash

    annotation = {
        "text": "\n".join(text_pieces),
        "pages": [{
            "width": pil_image.width,
            "height": pil_image.height,
            "blocks": [{
                "paragraphs": [{
                    "words": words,
                    "boundingBox": {"vertices": [
                        {"x": 0, "y": 0},
                        {"x": pil_image.width, "y": 0},
                        {"x": pil_image.width, "y": pil_image.height},
                        {"x": 0, "y": pil_image.height},
                    ]},
                }],
                "boundingBox": {"vertices": [
                    {"x": 0, "y": 0},
                    {"x": pil_image.width, "y": 0},
                    {"x": pil_image.width, "y": pil_image.height},
                    {"x": 0, "y": pil_image.height},
                ]},
            }],
        }],
    }

    logger.info(
        "PaddleOCR completed successfully",
        extra={"image_hash": img_hash, "text_length": len(annotation["text"]), "word_count": len(words)},
    )
    return annotation, img_hash
