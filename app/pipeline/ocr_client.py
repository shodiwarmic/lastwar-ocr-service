"""
app/pipeline/ocr_client.py

Google Cloud Vision API wrapper for text detection.

Design decisions:
    - Uses document_text_detection rather than text_detection. The document
      variant is optimised for dense layouts (tables, columns) and returns
      richer bounding box data at the word and symbol level — essential for
      correctly associating player names with their scores.

    - The Vision client is initialised once per container instance and reused
      across all requests. Cloud Run keeps warm instances alive between requests,
      so this avoids the ~300ms authentication overhead on every call.

    - Dependency injection: run_ocr() accepts an optional client parameter so
      tests can pass a mock client and never touch the real API. The fixture
      capture tool uses the real client exactly once per image.

    - Image hashing: a SHA-256 hash of the image bytes is computed before
      calling the API. The hash is returned alongside results so the route
      handler can implement a simple cache (avoiding duplicate API calls if
      the same image is submitted twice in the same or future batches).

Authentication:
    Cloud Run automatically provides Application Default Credentials via the
    service account attached to the Cloud Run service. No API key or credential
    file is needed in the container. For local development, run:
        gcloud auth application-default login
    or set GOOGLE_APPLICATION_CREDENTIALS to point to a service account JSON.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from PIL import Image
from google.cloud import vision

from app.utils.image_utils import pil_to_bytes
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level client cache — initialised on first use, reused thereafter
_vision_client: Optional[vision.ImageAnnotatorClient] = None


# ---------------------------------------------------------------------------
# Client management
# ---------------------------------------------------------------------------

def get_vision_client() -> vision.ImageAnnotatorClient:
    """
    Returns a cached Google Cloud Vision ImageAnnotatorClient.

    Initialises the client on the first call and reuses it for all subsequent
    calls within the same container instance. This follows the Cloud Run best
    practice of performing expensive initialisation outside the request handler.

    Returns:
        An authenticated ImageAnnotatorClient ready to make API calls.

    Raises:
        google.auth.exceptions.DefaultCredentialsError: If Application Default
        Credentials are not configured. See module docstring for setup steps.
    """
    global _vision_client
    if _vision_client is None:
        logger.info("Initialising Google Cloud Vision client")
        _vision_client = vision.ImageAnnotatorClient()
    return _vision_client


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def run_ocr(
    pil_image: Image.Image,
    client: Optional[vision.ImageAnnotatorClient] = None,
) -> tuple[Optional[object], str]:
    """
    Submits a PIL Image to the Vision API and returns the text annotation.

    Uses document_text_detection for superior handling of the dense tabular
    layout found in Last War ranking screens. This mode returns a full
    TextAnnotation with page → block → paragraph → word → symbol hierarchy,
    giving precise bounding boxes for spatial row clustering in the extractor.

    Args:
        pil_image: The image to analyse. For best results this should be a
                   stitched image prepared by the stitcher module.
        client:    Optional Vision client for dependency injection in tests.
                   If None, get_vision_client() is called automatically.

    Returns:
        Tuple of (TextAnnotation_or_None, image_hash_string).
        TextAnnotation is None if the API call fails or returns no text.
        image_hash is a SHA-256 hex digest of the PNG bytes submitted —
        use this as a cache key to avoid re-processing identical images.

    Example:
        annotation, img_hash = run_ocr(stitched_image)
        if annotation:
            blocks = extract_text_blocks(annotation)
    """
    client = client or get_vision_client()

    img_bytes = pil_to_bytes(pil_image, fmt="PNG")
    img_hash  = _hash_image_bytes(img_bytes)

    vision_image = vision.Image(content=img_bytes)

    try:
        response = client.document_text_detection(image=vision_image)
    except Exception as exc:
        logger.error(
            "Vision API call failed",
            extra={"image_hash": img_hash, "error": str(exc)},
        )
        return None, img_hash

    if response.error.message:
        logger.error(
            "Vision API returned an error",
            extra={"image_hash": img_hash, "api_error": response.error.message},
        )
        return None, img_hash

    annotation = response.full_text_annotation
    if not annotation or not annotation.text:
        logger.warning(
            "Vision API returned empty annotation",
            extra={"image_hash": img_hash},
        )
        return None, img_hash

    logger.info(
        "OCR completed successfully",
        extra={
            "image_hash": img_hash,
            "text_length": len(annotation.text),
        },
    )

    return annotation, img_hash


# ---------------------------------------------------------------------------
# Text block extraction
# ---------------------------------------------------------------------------

def extract_text_blocks(annotation) -> list[dict]:
    """
    Parses a Vision API TextAnnotation into a flat list of word-level blocks.

    Iterates through the page → block → paragraph → word hierarchy and
    assembles each word into a dict with its text and bounding box. Words
    are returned sorted top-to-bottom then left-to-right, which matches the
    natural reading order of the ranking tables and simplifies row clustering
    in the extractor.

    Operates at the word level (not symbol level) because player names and
    scores are single words or short multi-word strings. Symbol-level
    granularity would produce hundreds of blocks per image and make row
    clustering significantly more expensive.

    Args:
        annotation: A google.cloud.vision.TextAnnotation proto object as
                    returned by run_ocr(). Also accepts a plain dict in the
                    format produced by capture_ocr_fixture.py for test use.

    Returns:
        List of dicts, each representing one word:
        [
            {
                "text": "SirBucksALot",
                "bbox": <BoundingPoly or dict with vertices key>,
                "avg_x": 245.0,
                "avg_y": 388.0,
            },
            ...
        ]
        Sorted by avg_y ascending (top of image first), then avg_x ascending.
    """
    blocks: list[dict] = []

    try:
        pages = annotation.pages
    except AttributeError:
        # Dict form from fixtures
        pages = annotation.get("pages", [])

    for page in pages:
        try:
            block_list = page.blocks
        except AttributeError:
            block_list = page.get("blocks", [])

        for block in block_list:
            try:
                paragraphs = block.paragraphs
            except AttributeError:
                paragraphs = block.get("paragraphs", [])

            for paragraph in paragraphs:
                try:
                    words = paragraph.words
                except AttributeError:
                    words = paragraph.get("words", [])

                for word in words:
                    text = _word_to_string(word)
                    if not text.strip():
                        continue

                    bbox     = _get_bbox(word)
                    avg_x, avg_y = _avg_xy_from_bbox(bbox)

                    blocks.append({
                        "text":  text,
                        "bbox":  bbox,
                        "avg_x": avg_x,
                        "avg_y": avg_y,
                    })

    # Sort top-to-bottom, left-to-right
    blocks.sort(key=lambda b: (b["avg_y"], b["avg_x"]))

    return blocks


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _hash_image_bytes(data: bytes) -> str:
    """Returns a SHA-256 hex digest of the image bytes for use as a cache key."""
    return hashlib.sha256(data).hexdigest()


def _word_to_string(word) -> str:
    """
    Assembles a word proto or dict into a plain string.

    Vision API stores words as a sequence of symbols. This joins them without
    spaces (symbols within a word are always adjacent) and strips whitespace.
    """
    try:
        return "".join(s.text for s in word.symbols).strip()
    except AttributeError:
        symbols = word.get("symbols", [])
        return "".join(s.get("text", "") for s in symbols).strip()


def _get_bbox(word):
    """
    Returns the bounding_box from a word proto or the 'boundingBox' key
    from a dict, normalising to a consistent accessor for downstream code.
    """
    try:
        return word.bounding_box
    except AttributeError:
        return word.get("boundingBox") or word.get("bounding_box", {})


def _avg_xy_from_bbox(bbox) -> tuple[float, float]:
    """
    Computes (avg_x, avg_y) from a bounding box proto or dict.

    Returns (0.0, 0.0) on any failure so a malformed bbox doesn't crash
    the extraction pipeline — the block is kept but sorted to the top.
    """
    try:
        vertices = list(bbox.vertices)
        avg_x = sum(v.x for v in vertices) / len(vertices)
        avg_y = sum(v.y for v in vertices) / len(vertices)
        return avg_x, avg_y
    except (AttributeError, ZeroDivisionError):
        pass

    try:
        vertices = bbox.get("vertices", [])
        if not vertices:
            return 0.0, 0.0
        avg_x = sum(v.get("x", 0) for v in vertices) / len(vertices)
        avg_y = sum(v.get("y", 0) for v in vertices) / len(vertices)
        return avg_x, avg_y
    except (TypeError, ZeroDivisionError):
        return 0.0, 0.0
