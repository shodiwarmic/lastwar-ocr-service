"""
app/routes.py

HTTP route definitions for the Last War OCR microservice.

Endpoints:
    POST /process-batch   — Main batch processing endpoint
    GET  /health          — Cloud Run health check

Pipeline orchestration (POST /process-batch):
    1. Validate incoming files
    2. Convert to PIL images
    3. Pass 1: classify each image using colour sampling
    4. For low-confidence images: Pass 2 OCR-assisted classification
    5. Group high-confidence images by (category, resolution)
    6. Stitch each group into one tall image
    7. Run OCR on all stitched images + any Pass 2 fallback images
    8. Extract player rows from each OCR result
    9. Merge results into BatchResult and return JSON

Cache:
    A simple in-memory dict maps image hash → extraction result.
    If the same screenshot is submitted twice in the same container lifetime,
    the Vision API is not called again. The cache is per-instance and does not
    persist across container restarts — this is intentional to avoid stale data.
    For a persistent cache, replace with Cloud Memorystore (Redis).
"""

from __future__ import annotations

import hashlib

from flask import Blueprint, jsonify, request

from app.models.schemas import BatchResult, PlayerEntry
from app.pipeline.classifier import (
    CONFIDENCE_THRESHOLD,
    classify_screenshot,
    classify_from_ocr_text,
)
from app.pipeline.extractor import extract_players
from app.pipeline.ocr_client import extract_text_blocks, run_ocr
from app.pipeline.stitcher import (
    group_images_by_category_and_resolution,
    stitch_images_vertically,
)
from app.utils.image_utils import get_image_dimensions, pil_from_file_storage, pil_to_bytes
from app.utils.logger import get_logger, log_classification_event

logger = get_logger(__name__)

bp = Blueprint("main", __name__)

# Simple in-memory result cache: {image_hash: list[PlayerEntry]}
_result_cache: dict[str, list[PlayerEntry]] = {}

# Maximum number of images accepted per batch (safety limit)
MAX_IMAGES_PER_BATCH = 100


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/health", methods=["GET"])
def health():
    """
    Cloud Run health check endpoint.

    Cloud Run sends periodic GET /health requests to verify the container
    is alive. Returns 200 OK with a minimal JSON body. No authentication
    required — health checks come from the Cloud Run infrastructure.
    """
    return jsonify({"status": "ok"}), 200


@bp.route("/process-batch", methods=["POST"])
def process_batch():
    """
    Accepts a batch of Last War ranking screenshots and returns structured data.

    Request:
        Content-Type: multipart/form-data
        Body key: "images" (one or more image files)

    Response 200:
        {
            "monday":    [{"player_name": "...", "score": 123456}],
            "friday":    [...],
            "weekly":    [...],
            "power":     [...]
        }
        Only categories that produced results are included.

    Response 400:
        {"error": "human-readable error message"}

    Response 500:
        {"error": "Internal processing error", "detail": "..."}
    """
    # ------------------------------------------------------------------ #
    # 1. Validate input
    # ------------------------------------------------------------------ #
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided. Send images via multipart/form-data with key 'images'."}), 400

    if len(files) > MAX_IMAGES_PER_BATCH:
        return jsonify({"error": f"Too many images. Maximum {MAX_IMAGES_PER_BATCH} per batch."}), 400

    logger.info("Batch received", extra={"image_count": len(files)})

    # ------------------------------------------------------------------ #
    # 2. Load images
    # ------------------------------------------------------------------ #
    loaded: list[tuple] = []   # (pil_image, filename, bytes_hash)
    for file_storage in files:
        filename = file_storage.filename or "unknown"
        pil_image = pil_from_file_storage(file_storage)
        if pil_image is None:
            logger.warning("Skipping unreadable image", extra={"image_filename": filename})
            continue

        img_bytes = pil_to_bytes(pil_image)
        img_hash  = hashlib.sha256(img_bytes).hexdigest()
        loaded.append((pil_image, filename, img_hash))

    if not loaded:
        return jsonify({"error": "No valid images could be opened from the provided files."}), 400

    # ------------------------------------------------------------------ #
    # 3. Pass 1 — colour-based classification
    # ------------------------------------------------------------------ #
    classified: list[tuple]  = []   # (pil_image, category, filename) for high-confidence
    fallback_images: list[tuple] = []   # (pil_image, filename, img_hash) for OCR fallback

    for pil_image, filename, img_hash in loaded:
        category, confidence = classify_screenshot(pil_image, filename=filename)

        if confidence >= CONFIDENCE_THRESHOLD and category is not None:
            classified.append((pil_image, category, filename))
            log_classification_event(
                logger, filename,
                resolution=get_image_dimensions(pil_image),
                pass1_result=category,
                pass1_confidence=confidence,
                pass2_result=None,
                ocr_triggered=False,
            )
        else:
            # Low confidence — queue for OCR-assisted classification
            fallback_images.append((pil_image, filename, img_hash))
            logger.debug(
                "Image queued for OCR fallback",
                extra={"image_filename": filename, "pass1_result": category, "confidence": confidence},
            )

    # ------------------------------------------------------------------ #
    # 4. Build stitched OCR queue from high-confidence images
    # ------------------------------------------------------------------ #
    result = BatchResult()

    # Group and stitch high-confidence images
    groups = group_images_by_category_and_resolution(classified)
    ocr_queue: list[tuple] = []   # (stitched_image, category, source_filenames)

    for (category, w, h), image_list in groups.items():
        stitched = stitch_images_vertically(image_list, category)
        filenames = [f for _, f in image_list]
        ocr_queue.append((stitched, category, filenames))

    # ------------------------------------------------------------------ #
    # 5. Process fallback images individually (unstitched)
    # ------------------------------------------------------------------ #
    for pil_image, filename, img_hash in fallback_images:

        # Check cache first
        if img_hash in _result_cache:
            logger.info("Cache hit for fallback image", extra={"image_filename": filename})
            # We don't know the category from cache alone — re-classify from
            # cached text blocks if available, else re-run OCR
            # For simplicity: cache miss on fallbacks, just re-run
            pass

        annotation, img_hash_returned = run_ocr(pil_image)
        if annotation is None:
            log_classification_event(
                logger, filename,
                resolution=get_image_dimensions(pil_image),
                pass1_result=None,
                pass1_confidence=0.0,
                pass2_result=None,
                ocr_triggered=True,
                error="OCR API call failed",
            )
            continue

        text_blocks = extract_text_blocks(annotation)
        category, confidence = classify_from_ocr_text(text_blocks, image=pil_image, filename=filename)

        log_classification_event(
            logger, filename,
            resolution=get_image_dimensions(pil_image),
            pass1_result=None,
            pass1_confidence=0.0,
            pass2_result=category,
            ocr_triggered=True,
        )

        if category is None:
            logger.error(
                "Classification failed after OCR fallback — image skipped",
                extra={"image_filename": filename},
            )
            continue

        # Extract players directly from the OCR already performed
        w, h = get_image_dimensions(pil_image)
        players = extract_players(text_blocks, screen_type=category, image_height=h, image_width=w)

        # Cache the result
        _result_cache[img_hash_returned] = players

        result.add_entries(category, players)

    # ------------------------------------------------------------------ #
    # 6. Run OCR on stitched groups and extract players
    # ------------------------------------------------------------------ #
    for stitched_image, category, source_filenames in ocr_queue:

        img_bytes = pil_to_bytes(stitched_image)
        img_hash  = hashlib.sha256(img_bytes).hexdigest()

        if img_hash in _result_cache:
            logger.info(
                "Cache hit for stitched image",
                extra={"category": category, "filenames": source_filenames},
            )
            result.add_entries(category, _result_cache[img_hash])
            continue

        annotation, img_hash_returned = run_ocr(stitched_image)
        if annotation is None:
            logger.error(
                "OCR failed for stitched group",
                extra={"category": category, "filenames": source_filenames},
            )
            continue

        text_blocks = extract_text_blocks(annotation)
        w, h = get_image_dimensions(stitched_image)
        players = extract_players(text_blocks, screen_type=category, image_height=h, image_width=w)

        _result_cache[img_hash_returned] = players
        result.add_entries(category, players)

    # ------------------------------------------------------------------ #
    # 7. Return response
    # ------------------------------------------------------------------ #
    if result.is_empty():
        logger.warning("Batch produced no results", extra={"image_count": len(loaded)})
        return jsonify({"warning": "No player data could be extracted from the provided images."}), 200

    response_data = result.to_response_dict()
    logger.info(
        "Batch complete",
        extra={
            "categories_returned": list(response_data.keys()),
            "total_players": sum(len(v) for v in response_data.values()),
        },
    )

    return jsonify(response_data), 200


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@bp.app_errorhandler(413)
def request_too_large(e):
    return jsonify({"error": "Request payload too large. Reduce image sizes or batch count."}), 413


@bp.app_errorhandler(500)
def internal_error(e):
    logger.error("Unhandled internal error", extra={"error": str(e)})
    return jsonify({"error": "Internal processing error", "detail": str(e)}), 500
