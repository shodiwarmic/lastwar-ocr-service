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

from app.models.schemas import (
    BatchDiagnostic,
    BatchDiagnostics,
    BatchResult,
    PlayerEntry,
    SectionDiagnostic,
    VALID_CATEGORIES,
    classification_method,
)
from app.pipeline.classifier import classify_from_ocr_text
from app.pipeline.extractor import extract_players
from app.pipeline.ocr_client import active_engine, extract_text_blocks, run_ocr
from app.pipeline.stitcher import prepare_stitched_batches
from app.utils.image_utils import pil_from_file_storage, pil_to_bytes
from app.utils.logger import get_logger

logger = get_logger(__name__)

bp = Blueprint("main", __name__)

# In-memory result cache, keyed on the SHA-256 of the stitched-image bytes.
# Stores per-batch results AND the per-section diagnostics derived from the
# image *content* (category, confidence, method, players_found, y_range, note).
# Filenames are request-specific, not content-derived, so on a cache hit the
# replayed sections' `image`/`batch_index`/`cache_hit` are overwritten with the
# current request's values (see process_batch). Per-instance, non-persistent.
#   {jpeg_hash: {"results": {category: [PlayerEntry, ...]},
#                "sections": [SectionDiagnostic, ...]}}
_result_cache: dict[str, dict] = {}

MAX_IMAGES_PER_BATCH = 100


def _make_section(
    region,
    batch_index: int,
    *,
    category,
    confidence: float,
    method: str,
    players_found: int,
    note,
    cache_hit: bool = False,
) -> SectionDiagnostic:
    """Builds a SectionDiagnostic from an ImageRegion plus classification outcome."""
    return SectionDiagnostic(
        image=region.filename,
        batch_index=batch_index,
        y_range=(region.y_start, region.y_end),
        category=category,
        confidence=confidence,
        method=method,
        players_found=players_found,
        cache_hit=cache_hit,
        note=note,
    )


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
    batch_diags: list[BatchDiagnostic] = []
    section_diags: list[SectionDiagnostic] = []
    is_override = category_override is not None

    for batch_index, (stitched_image, regions) in enumerate(batches):
        img_bytes = pil_to_bytes(stitched_image, fmt="JPEG")
        img_hash  = hashlib.sha256(img_bytes).hexdigest()
        source_images = [r.filename for r in regions]

        if img_hash in _result_cache:
            logger.info(
                "Cache hit for stitched batch",
                extra={"image_hash": img_hash[:12], "region_count": len(regions)},
            )
            cached = _result_cache[img_hash]
            for category, players in cached["results"].items():
                result.add_entries(category, players)
            # Replay cached sections, but re-stamp the request-specific fields:
            # the cache key is the stitched *bytes*, so the same content can be
            # re-uploaded under different filenames / at a different batch index.
            for region, sec in zip(regions, cached["sections"]):
                section_diags.append(sec.model_copy(update={
                    "image": region.filename,
                    "batch_index": batch_index,
                    "cache_hit": True,
                }))
            batch_diags.append(BatchDiagnostic(
                batch_index=batch_index, stitched_size=stitched_image.size,
                source_images=source_images, cache_hit=True,
            ))
            continue

        annotation, _ = run_ocr(stitched_image)
        if annotation is None:
            logger.error(
                "OCR failed for stitched batch",
                extra={"image_filenames": source_images},
            )
            for region in regions:
                section_diags.append(_make_section(
                    region, batch_index, category=None, confidence=0.0,
                    method="unclassified", players_found=0, note="ocr_failed",
                ))
            batch_diags.append(BatchDiagnostic(
                batch_index=batch_index, stitched_size=stitched_image.size,
                source_images=source_images, cache_hit=False,
            ))
            continue

        all_blocks = extract_text_blocks(annotation)
        batch_results: dict[str, list[PlayerEntry]] = {}
        batch_sections: list[SectionDiagnostic] = []

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
                batch_sections.append(_make_section(
                    region, batch_index, category=None, confidence=0.0,
                    method="unclassified", players_found=0, note="no_ocr_blocks",
                ))
                continue

            if is_override:
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
                batch_sections.append(_make_section(
                    region, batch_index, category=None, confidence=confidence,
                    method="unclassified", players_found=0, note="classification_failed",
                ))
                continue

            section_height = region.y_end - region.y_start
            players = extract_players(
                section_blocks,
                screen_type=category,
                image_height=section_height,
                image_width=stitched_image.width,
            )

            result.add_entries(category, players)
            batch_results.setdefault(category, []).extend(players)
            batch_sections.append(_make_section(
                region, batch_index, category=category, confidence=confidence,
                method=classification_method(category, confidence, override=is_override),
                players_found=len(players),
                note=None if players else "no_players",
            ))

            logger.info(
                "Section processed",
                extra={
                    "image_filename": region.filename,
                    "category":       category,
                    "confidence":     round(confidence, 2),
                    "players_found":  len(players),
                },
            )

        _result_cache[img_hash] = {"results": batch_results, "sections": batch_sections}
        section_diags.extend(batch_sections)
        batch_diags.append(BatchDiagnostic(
            batch_index=batch_index, stitched_size=stitched_image.size,
            source_images=source_images, cache_hit=False,
        ))

    # ------------------------------------------------------------------ #
    # 5. Assemble diagnostics + return the {results, diagnostics} envelope
    # ------------------------------------------------------------------ #
    diagnostics = BatchDiagnostics(
        engine=active_engine(),
        image_count=len(loaded),
        batch_count=len(batches),
        category_override=category_override,
        batches=batch_diags,
        sections=section_diags,
    ).model_dump(exclude_none=True)

    if result.is_empty():
        logger.warning("Batch produced no results", extra={"image_count": len(loaded)})
        return jsonify({
            "results": {},
            "diagnostics": diagnostics,
            "warning": "No player data could be extracted from the provided images.",
        }), 200

    results = result.to_response_dict()
    logger.info(
        "Batch complete",
        extra={
            "categories_returned": list(results.keys()),
            "total_players": sum(len(v) for v in results.values()),
        },
    )

    return jsonify({"results": results, "diagnostics": diagnostics}), 200


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
