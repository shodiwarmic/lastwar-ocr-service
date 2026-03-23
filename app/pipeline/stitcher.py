"""
app/pipeline/stitcher.py

Groups classified screenshots by category and resolution, then stitches
each group into a single tall image before OCR.

Why stitching matters:
    The Vision API bills per image submitted. Stitching 5 screenshots of the
    same day into one image costs 1 unit instead of 5. For an alliance
    submitting a full week of daily + weekly + power screenshots (~10 images),
    stitching reduces Vision API calls from 10 to ~3-4, keeping usage well
    within the free tier.

Resolution grouping:
    Images are grouped by (category, width, height) so only identically-sized
    screenshots are concatenated. This avoids distortion and OCR errors that
    would result from resizing images before stitching. In the rare case that
    a batch contains screenshots from two different devices, each resolution
    produces its own stitched image and the extracted results are merged.

Chrome removal:
    Before stitching, each image has its top navigation bar and bottom
    self-player row cropped out. These UI elements are identical across
    screenshots and would confuse the extractor if duplicated repeatedly
    in a tall stitched image (e.g. "Commander Points" column headers
    appearing 5 times). The first image in a group retains its header
    and the last image retains its footer so the stitched image still has
    complete context at the top and bottom.
"""

from __future__ import annotations

from PIL import Image

from app.utils.image_utils import crop_top_bottom, get_image_dimensions
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Chrome crop fractions per screen type
# Tuned against sample screenshots — adjust if layouts change.
# ---------------------------------------------------------------------------

# Fraction of image height to remove from the TOP (navigation/tab bar area)
TOP_CROP_FRACTIONS = {
    "power":   0.22,   # Strength Ranking has a taller header with 3 tabs
    "weekly":  0.18,   # Weekly Rank header with 2 tabs
    "default": 0.22,   # Daily Rank header with 2 tabs + 6 day tabs
}

# Fraction of image height to remove from the BOTTOM (self-player highlight row)
BOTTOM_CROP_FRACTIONS = {
    "power":   0.12,
    "weekly":  0.12,
    "default": 0.12,
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def group_images_by_category_and_resolution(
    classified_images: list[tuple[Image.Image, str, str]],
) -> dict[tuple[str, int, int], list[tuple[Image.Image, str]]]:
    """
    Groups classified images by (category, width, height).

    Using resolution as a secondary key ensures only identically-sized images
    are stitched together, preventing width mismatches and OCR quality issues
    that would result from concatenating images of different resolutions.

    Args:
        classified_images: List of (pil_image, category, filename) tuples.
                           category is one of the VALID_CATEGORIES strings.

    Returns:
        Dict mapping (category, width, height) → list of (pil_image, filename).

    Example:
        {
            ("friday",  1080, 2400): [(img1, "8851.png"), (img2, "8836.png")],
            ("power",   1080, 2400): [(img3, "8725.png")],
            ("weekly",  1284, 2778): [(img4, "extra.png")],
        }
    """
    groups: dict[tuple[str, int, int], list[tuple[Image.Image, str]]] = {}

    for image, category, filename in classified_images:
        if category is None:
            logger.warning(
                "Skipping image with no category",
                extra={"filename": filename},
            )
            continue

        w, h = get_image_dimensions(image)
        key = (category, w, h)

        if key not in groups:
            groups[key] = []
        groups[key].append((image, filename))

    # Log grouping summary
    for (cat, w, h), items in groups.items():
        logger.info(
            "Image group formed",
            extra={
                "category": cat,
                "resolution": f"{w}x{h}",
                "image_count": len(items),
                "filenames": [f for _, f in items],
            },
        )

    return groups


def stitch_images_vertically(
    image_list: list[tuple[Image.Image, str]],
    category: str,
) -> Image.Image:
    """
    Concatenates a list of images into a single tall stitched image.

    Chrome removal strategy:
    - First image:  keep header (top chrome), remove bottom chrome
    - Middle images: remove both top and bottom chrome
    - Last image:   remove top chrome, keep footer (bottom chrome)
    - Single image: no cropping applied (nothing to de-duplicate)

    All images in the list are guaranteed to have the same width by the
    grouping step, so no width validation is needed here.

    Args:
        image_list: List of (pil_image, filename) tuples in the order they
                    should appear top-to-bottom in the stitched output.
        category:   The category string, used to select the correct crop fractions.

    Returns:
        A single PIL Image containing all input images concatenated vertically
        with duplicated UI chrome removed from interior boundaries.
    """
    if len(image_list) == 1:
        logger.debug(
            "Single image — no stitching required",
            extra={"filename": image_list[0][1], "category": category},
        )
        return image_list[0][0]

    top_crop    = TOP_CROP_FRACTIONS.get(category, TOP_CROP_FRACTIONS["default"])
    bottom_crop = BOTTOM_CROP_FRACTIONS.get(category, BOTTOM_CROP_FRACTIONS["default"])

    cropped_images: list[Image.Image] = []

    for i, (img, filename) in enumerate(image_list):
        is_first = i == 0
        is_last  = i == len(image_list) - 1

        if is_first and is_last:
            # Should not reach here (handled above) but be safe
            cropped_images.append(img)
        elif is_first:
            # Keep header, remove footer
            cropped = crop_top_bottom(img, top_fraction=0.0, bottom_fraction=bottom_crop)
            cropped_images.append(cropped)
        elif is_last:
            # Remove header, keep footer
            cropped = crop_top_bottom(img, top_fraction=top_crop, bottom_fraction=0.0)
            cropped_images.append(cropped)
        else:
            # Interior image — remove both header and footer
            cropped = crop_top_bottom(img, top_fraction=top_crop, bottom_fraction=bottom_crop)
            cropped_images.append(cropped)

        logger.debug(
            "Cropped image for stitching",
            extra={
                "filename": filename,
                "position": "first" if is_first else "last" if is_last else "middle",
                "original_size": f"{img.width}x{img.height}",
                "cropped_size": f"{cropped_images[-1].width}x{cropped_images[-1].height}",
            },
        )

    total_height = sum(img.height for img in cropped_images)
    width = cropped_images[0].width

    stitched = Image.new("RGB", (width, total_height))
    y_offset = 0
    for img in cropped_images:
        stitched.paste(img, (0, y_offset))
        y_offset += img.height

    logger.info(
        "Stitching complete",
        extra={
            "category": category,
            "input_count": len(image_list),
            "stitched_size": f"{width}x{total_height}",
        },
    )

    return stitched


def prepare_stitched_batches(
    classified_images: list[tuple[Image.Image, str, str]],
) -> list[tuple[Image.Image, str]]:
    """
    Full preparation pipeline: group → stitch → return list ready for OCR.

    This is the primary entry point called by the route handler. It combines
    grouping and stitching into a single call and returns a flat list of
    (stitched_image, category) tuples — one per resolution group — ready
    to be passed directly to the OCR client.

    Args:
        classified_images: List of (pil_image, category, filename) tuples
                           from the classification step.

    Returns:
        List of (stitched_pil_image, category) tuples. Each tuple represents
        one Vision API call.
    """
    groups = group_images_by_category_and_resolution(classified_images)
    result: list[tuple[Image.Image, str]] = []

    for (category, w, h), image_list in groups.items():
        stitched = stitch_images_vertically(image_list, category)
        result.append((stitched, category))

    logger.info(
        "Batch preparation complete",
        extra={
            "total_input_images": len(classified_images),
            "total_ocr_calls_needed": len(result),
        },
    )

    return result
