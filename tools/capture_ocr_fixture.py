"""
tools/capture_ocr_fixture.py

CLI tool to capture real Vision API responses and save them as JSON fixtures
for use in offline classification and extraction tests.

Run this tool ONCE against your sample screenshots. The resulting JSON files
are saved to tests/fixtures/ocr_responses/ and are used by all test suites
so the Vision API is never called during test runs.

Usage:
    # Capture a single image
    python tools/capture_ocr_fixture.py path/to/screenshot.png

    # Capture all images in a directory
    python tools/capture_ocr_fixture.py path/to/screenshots/

    # Specify a custom output directory
    python tools/capture_ocr_fixture.py path/to/screenshots/ --output tests/fixtures/ocr_responses/

    # Preview what would be captured without calling the API
    python tools/capture_ocr_fixture.py path/to/screenshots/ --dry-run

Prerequisites:
    - Google Cloud credentials configured (gcloud auth application-default login)
    - pip install -r requirements.txt run from the project root
    - Run from the project root directory so imports resolve correctly

Output:
    One JSON file per image, named <original_filename_without_extension>.json
    Each file contains the full Vision API TextAnnotation serialised to JSON,
    which can be loaded back and passed to classify_from_ocr_text() and
    extract_players() in tests.

Cost:
    Each image costs 1 Vision API unit. With 10 sample screenshots this tool
    uses 10 units from your 1,000/month free tier. After the initial capture
    you should not need to run this tool again unless you acquire new screenshot
    types to cover edge cases.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from app.pipeline.ocr_client import get_vision_client, run_ocr, extract_text_blocks
from app.utils.image_utils import pil_from_bytes
from app.utils.logger import get_logger

logger = get_logger("capture_ocr_fixture")

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Default output directory relative to project root
DEFAULT_OUTPUT_DIR = Path("tests/fixtures/ocr_responses")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _vertex_to_dict(vertex) -> dict:
    """Converts a Vision API Vertex proto to a plain dict."""
    return {"x": vertex.x, "y": vertex.y}


def _bbox_to_dict(bbox) -> dict:
    """Converts a BoundingPoly proto to a serialisable dict."""
    return {"vertices": [_vertex_to_dict(v) for v in bbox.vertices]}


def _word_to_dict(word) -> dict:
    """Converts a Word proto to a serialisable dict including symbols."""
    return {
        "symbols": [
            {
                "text": s.text,
                "bounding_box": _bbox_to_dict(s.bounding_box),
            }
            for s in word.symbols
        ],
        "boundingBox": _bbox_to_dict(word.bounding_box),
    }


def _paragraph_to_dict(paragraph) -> dict:
    return {
        "words": [_word_to_dict(w) for w in paragraph.words],
        "boundingBox": _bbox_to_dict(paragraph.bounding_box),
    }


def _block_to_dict(block) -> dict:
    return {
        "paragraphs": [_paragraph_to_dict(p) for p in block.paragraphs],
        "boundingBox": _bbox_to_dict(block.bounding_box),
    }


def _page_to_dict(page) -> dict:
    return {
        "blocks": [_block_to_dict(b) for b in page.blocks],
        "width":  page.width,
        "height": page.height,
    }


def annotation_to_dict(annotation) -> dict:
    """
    Serialises a full Vision API TextAnnotation proto to a plain dict.

    The resulting dict mirrors the structure that ocr_client.extract_text_blocks()
    expects so tests can load fixtures and pass them directly to the extractor
    without any adaptation layer.

    Args:
        annotation: google.cloud.vision.TextAnnotation proto.

    Returns:
        JSON-serialisable dict containing pages, blocks, words, and bounding boxes.
    """
    return {
        "text":  annotation.text,
        "pages": [_page_to_dict(p) for p in annotation.pages],
    }


# ---------------------------------------------------------------------------
# Core capture logic
# ---------------------------------------------------------------------------

def capture_fixture_for_image(
    image_path: Path,
    output_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """
    Captures the Vision API response for a single image and writes it to JSON.

    Args:
        image_path: Path to the source image file.
        output_dir: Directory where the JSON fixture will be written.
        dry_run:    If True, skips the API call and file write — useful for
                    validating file discovery before spending API units.

    Returns:
        True if the fixture was written successfully (or dry_run is True),
        False if the API call or file write failed.
    """
    output_path = output_dir / f"{image_path.stem}.json"

    if output_path.exists() and not force:
        print(f"  ⚠  Fixture already exists, skipping: {output_path.name}")
        return True
    if output_path.exists() and force:
        print(f"  ↻  Re-capturing (--force): {output_path.name}")

    if dry_run:
        print(f"  [DRY RUN] Would capture: {image_path.name} → {output_path.name}")
        return True

    print(f"  → Capturing: {image_path.name}")

    # Load image
    try:
        img_bytes = image_path.read_bytes()
        pil_image = pil_from_bytes(img_bytes)
        if pil_image is None:
            print(f"  ✗  Failed to open image: {image_path.name}")
            return False
    except Exception as exc:
        print(f"  ✗  Error reading {image_path.name}: {exc}")
        return False

    # Call Vision API
    annotation, img_hash = run_ocr(pil_image)
    if annotation is None:
        print(f"  ✗  OCR failed for: {image_path.name}")
        return False

    # Serialise to dict.
    # Build text_blocks with proper dict bboxes — calling extract_text_blocks()
    # on the live proto would store bbox as a proto string via _json_serialise_fallback.
    def _extract_blocks_as_dicts(ann):
        blocks = []
        for page in ann.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        text = "".join(s.text for s in word.symbols).strip()
                        if not text:
                            continue
                        bbox_dict = _bbox_to_dict(word.bounding_box)
                        verts = bbox_dict.get("vertices", [])
                        xs = [v["x"] for v in verts]
                        ys = [v["y"] for v in verts]
                        avg_x = sum(xs) / len(xs) if xs else 0.0
                        avg_y = sum(ys) / len(ys) if ys else 0.0
                        blocks.append({
                            "text":  text,
                            "bbox":  bbox_dict,
                            "avg_x": avg_x,
                            "avg_y": avg_y,
                        })
        blocks.sort(key=lambda b: (b["avg_y"], b["avg_x"]))
        return blocks

    fixture_data = {
        "source_file":  image_path.name,
        "image_hash":   img_hash,
        "image_width":  pil_image.width,
        "image_height": pil_image.height,
        "annotation":   annotation_to_dict(annotation),
        # Pre-extract text blocks with properly serialised dict bboxes
        "text_blocks":  _extract_blocks_as_dicts(annotation),
    }

    # Write JSON — text_blocks contain bbox objects, need custom serialisation
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(fixture_data, f, indent=2, default=_json_serialise_fallback)
        print(f"  ✓  Saved: {output_path}")
        return True
    except Exception as exc:
        print(f"  ✗  Failed to write fixture {output_path.name}: {exc}")
        return False


def _json_serialise_fallback(obj):
    """
    JSON serialisation fallback for proto objects that aren't natively serialisable.
    Converts them to their string representation as a last resort.
    """
    return str(obj)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture Google Cloud Vision API responses as JSON test fixtures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to a single image file or a directory of images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for JSON fixtures. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List images that would be processed without calling the API.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-capture fixtures even if they already exist. Use after tool fixes.",
    )

    args = parser.parse_args()

    source: Path = args.source
    output_dir: Path = args.output

    # Collect image paths
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            print(f"Error: {source} is not a supported image file.")
            sys.exit(1)
        image_paths = [source]
    elif source.is_dir():
        image_paths = sorted(
            p for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            print(f"No images found in {source}")
            sys.exit(1)
    else:
        print(f"Error: {source} does not exist.")
        sys.exit(1)

    print(f"\nLast War OCR Fixture Capture Tool")
    print(f"{'=' * 40}")
    print(f"Source:    {source}")
    print(f"Output:    {output_dir}")
    print(f"Images:    {len(image_paths)}")
    print(f"Dry run:   {args.dry_run}")
    print(f"{'=' * 40}\n")

    if not args.dry_run:
        # Warm up Vision client before processing loop
        print("Initialising Vision API client...")
        try:
            get_vision_client()
            print("✓ Vision client ready\n")
        except Exception as exc:
            print(f"✗ Failed to initialise Vision client: {exc}")
            print("  Ensure Application Default Credentials are configured.")
            print("  Run: gcloud auth application-default login")
            sys.exit(1)

    success_count = 0
    fail_count    = 0

    for image_path in image_paths:
        ok = capture_fixture_for_image(image_path, output_dir, dry_run=args.dry_run, force=args.force)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 40}")
    print(f"Done. Success: {success_count}  Failed: {fail_count}")
    if not args.dry_run:
        print(f"Fixtures written to: {output_dir.resolve()}")
    print(f"{'=' * 40}\n")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
