"""
app/routes.py

HTTP route definitions for the Last War OCR microservice.

Endpoints:
    POST /process-batch   — Main batch processing endpoint
    GET  /health          — Cloud Run health check

Pipeline (POST /process-batch):
    1. Validate and load uploaded images.
    2. Group by resolution (width × height); stitch each group into one tall
       image with black separator bands between images.  Record the Y range of
       each source image in the stitched output.
    3. If any stitched image would exceed Vision API limits (20 MB / 75 MP),
       recursively bisect that group until every sub-batch is within bounds.
    4. Submit each stitched image to the Vision API (one call per sub-batch).
    5. For each source image section: filter OCR blocks to its Y range, then
       classify the section from its text content.  Day/tab detection uses
       colour sampling on the OCR-returned bounding boxes.
    6. Extract player rows from each classified section.
    7. Merge results and return JSON.

Cache:
    An in-memory dict maps JPEG-hash → per-category extraction results.
    Per-instance, non-persistent.
"""

from __future__ import annotations

import hashlib

from flask import Blueprint, jsonify, request

from app.models.schemas import BatchResult, PlayerEntry, VALID_CATEGORIES
from app.pipeline.classifier import classify_from_ocr_text
from app.pipeline.extractor import extract_players
from app.pipeline.ocr_client import extract_text_blocks, run_ocr
from app.pipeline.stitcher import prepare_stitched_batches
from app.utils.image_utils import pil_from_file_storage, pil_to_bytes
from app.utils.logger import get_logger

logger = get_logger(__name__)

bp = Blueprint("main", __name__)

# In-memory result cache: {jpeg_hash: {category: [PlayerEntry, ...]}}
_result_cache: dict[str, dict[str, list[PlayerEntry]]] = {}

MAX_IMAGES_PER_BATCH = 100


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@bp.route("/process-batch", methods=["POST"])
def process_batch():
    # ------------------------------------------------------------------ #
    # 1. Validate input
    # ------------------------------------------------------------------ #
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided. Send images via multipart/form-data with key 'images'."}), 400

    if len(files) > MAX_IMAGES_PER_BATCH:
        return jsonify({"error": f"Too many images. Maximum {MAX_IMAGES_PER_BATCH} per batch."}), 400

    # Optional caller-supplied category — bypasses classification entirely.
    # Required for screens where auto-detection is not used (e.g. alliance
    # contribution tabs: mutual_assistance, siege, rare_soil_war, defeat).
    category_override: str | None = request.form.get("category", "").strip() or None
    if category_override is not None and category_override not in VALID_CATEGORIES:
        return jsonify({"error": f"Unknown category '{category_override}'. Valid values: {sorted(VALID_CATEGORIES)}"}), 400

    logger.info("Batch received", extra={"image_count": len(files), "category_override": category_override})

    # ------------------------------------------------------------------ #
    # 2. Load images
    # ------------------------------------------------------------------ #
    loaded: list[tuple] = []
    for file_storage in files:
        filename  = file_storage.filename or "unknown"
        pil_image = pil_from_file_storage(file_storage)
        if pil_image is None:
            logger.warning("Skipping unreadable image", extra={"image_filename": filename})
            continue
        loaded.append((pil_image, filename))

    if not loaded:
        return jsonify({"error": "No valid images could be opened from the provided files."}), 400

    # ------------------------------------------------------------------ #
    # 3. Group by resolution, stitch with separators, split if needed
    # ------------------------------------------------------------------ #
    batches = prepare_stitched_batches(loaded)

    # ------------------------------------------------------------------ #
    # 4. OCR each stitched batch; classify and extract per section
    # ------------------------------------------------------------------ #
    result = BatchResult()

    for stitched_image, regions in batches:
        img_bytes = pil_to_bytes(stitched_image, fmt="JPEG")
        img_hash  = hashlib.sha256(img_bytes).hexdigest()

        if img_hash in _result_cache:
            logger.info(
                "Cache hit for stitched batch",
                extra={"image_hash": img_hash[:12], "region_count": len(regions)},
            )
            for category, players in _result_cache[img_hash].items():
                result.add_entries(category, players)
            continue

        annotation, _ = run_ocr(stitched_image)
        if annotation is None:
            logger.error(
                "OCR failed for stitched batch",
                extra={"image_filenames": [r.filename for r in regions]},
            )
            continue

        all_blocks = extract_text_blocks(annotation)
        batch_cache_entry: dict[str, list[PlayerEntry]] = {}

        for region in regions:
            section_blocks = [
                b for b in all_blocks
                if region.y_start <= b["avg_y"] < region.y_end
            ]

            if not section_blocks:
                logger.warning(
                    "No OCR blocks in section — image may be blank or unreadable",
                    extra={"image_filename": region.filename},
                )
                continue

            if category_override is not None:
                category   = category_override
                confidence = 1.0
            else:
                category, confidence = classify_from_ocr_text(
                    section_blocks,
                    image=stitched_image,
                    filename=region.filename,
                )

            if category is None:
                logger.warning(
                    "Classification failed for section",
                    extra={"image_filename": region.filename},
                )
                continue

            section_height = region.y_end - region.y_start
            players = extract_players(
                section_blocks,
                screen_type=category,
                image_height=section_height,
                image_width=stitched_image.width,
            )

            result.add_entries(category, players)
            batch_cache_entry.setdefault(category, []).extend(players)

            logger.info(
                "Section processed",
                extra={
                    "image_filename": region.filename,
                    "category":       category,
                    "confidence":     round(confidence, 2),
                    "players_found":  len(players),
                },
            )

        _result_cache[img_hash] = batch_cache_entry

    # ------------------------------------------------------------------ #
    # 5. Return response
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
