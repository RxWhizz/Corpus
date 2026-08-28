"""Pixel-to-nanometre calibration from a printed TEM scale bar.

Every downstream number in Corpus is a multiple of ``nm_per_pixel``, so this
module keeps that conversion in one deterministic place:

    nm_per_pixel = scale_nm / scale_pixels

A manual scale line always wins over automatic detection. Auto-detected bars
are reported with a confidence score and a warning, never silently trusted.
"""

import math

import cv2
import numpy as np

from corpus.errors import CalibrationError
from corpus.measurement.geometry import overlap_ratio

__all__ = [
    "nm_per_pixel",
    "parse_scale_line",
    "scale_bar_confidence",
    "collect_scale_candidates",
    "resolve_scale",
]


def nm_per_pixel(scale_nm, scale_pixels):
    """Nanometres per pixel for a bar of ``scale_nm`` spanning ``scale_pixels``.

    Raises :class:`~corpus.errors.CalibrationError` for non-positive inputs so
    a bad calibration fails loudly instead of producing zero-sized particles.
    """
    scale_nm = float(scale_nm)
    scale_pixels = float(scale_pixels)
    if not math.isfinite(scale_nm) or scale_nm <= 0:
        raise CalibrationError(f"Scale length must be a positive number of nm, got {scale_nm!r}.")
    if not math.isfinite(scale_pixels) or scale_pixels <= 0:
        raise CalibrationError(f"Scale bar length must be a positive number of pixels, got {scale_pixels!r}.")
    return scale_nm / scale_pixels


def parse_scale_line(value):
    """Parse a ``"x1,y1,x2,y2"`` manual scale line into a dict with ``length``.

    Returns ``None`` for an empty value, which means "no manual line given".
    """
    if not value:
        return None
    if isinstance(value, dict):
        coords = [float(value[key]) for key in ("x1", "y1", "x2", "y2")]
    else:
        try:
            coords = [float(part.strip()) for part in str(value).split(",")]
        except ValueError as error:
            raise CalibrationError("Manual scale line must be x1,y1,x2,y2.") from error
    if len(coords) != 4:
        raise CalibrationError("Manual scale line must be x1,y1,x2,y2.")
    x1, y1, x2, y2 = coords
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        raise CalibrationError("Manual scale line points must not be identical.")
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "length": length}


def scale_bar_confidence(x, y, w, h, image_width, image_height):
    """Heuristic ``[0, 1]`` score that a bright bar is the printed scale bar.

    TEM scale bars are long, thin, and printed low and to the right; a bar
    flush against the very bottom edge is usually frame furniture, not a bar.
    """
    length_score = min(1.0, w / max(image_width * 0.18, 1))
    bottom_score = min(1.0, max(0.0, y / max(image_height * 0.85, 1)))
    right_score = min(1.0, max(0.0, x / max(image_width * 0.65, 1)))
    thin_score = min(1.0, (w / max(h, 1)) / 12)
    edge_penalty = 0.45 if y >= image_height - 10 else 0
    score = 0.45 * length_score + 0.25 * bottom_score + 0.2 * right_score + 0.1 * thin_score - edge_penalty
    return round(max(0.0, score), 4)


def collect_scale_candidates(image):
    """Return scale-bar candidates, best first, de-duplicated by overlap.

    Two independent detectors are used so a bar that fails one still shows up:
    bright thin contours, and near-horizontal Hough segments in the lower-right
    quadrant.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape
    min_bar_width = max(20, int(width * 0.08))
    image_area = height * width
    candidates = []

    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > image_area * 0.25:
            continue
        if y >= height - 10:
            continue
        if w >= min_bar_width and 1 <= h <= 35 and w / max(h, 1) > 6:
            candidates.append({
                "x": x,
                "y": y,
                "width_px": float(w),
                "height_px": float(h),
                "method": "bright_contour",
                "confidence": scale_bar_confidence(x, y, w, h, width, height),
            })

    x0 = int(width * 0.35)
    y0 = int(height * 0.55)
    roi = gray[y0:height, x0:width]
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=15, minLineLength=min_bar_width, maxLineGap=4)
    if lines is not None:
        # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4).
        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = [int(value) for value in line]
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            absolute_y = y0 + min(y1, y2)
            absolute_x = x0 + min(x1, x2)
            if abs(dy) <= 3 and length >= min_bar_width and absolute_y < height - 10:
                thickness = max(1, abs(dy) + 1)
                candidates.append({
                    "x": int(absolute_x),
                    "y": int(absolute_y),
                    "width_px": float(length),
                    "height_px": float(thickness),
                    "method": "hough_line",
                    "confidence": scale_bar_confidence(absolute_x, absolute_y, length, thickness, width, height),
                })

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    deduped = []
    for candidate in candidates:
        bbox = (candidate["x"], candidate["y"], int(candidate["width_px"]), max(1, int(candidate["height_px"])))
        if any(
            overlap_ratio(bbox, (row["x"], row["y"], int(row["width_px"]), max(1, int(row["height_px"])))) > 0.5
            for row in deduped
        ):
            continue
        deduped.append(candidate)
    return deduped


def resolve_scale(image, scale_length, manual_scale_px=0, manual_scale_line=None):
    """Resolve the calibration for ``image``.

    Precedence: manual line, then manual pixel override, then best automatic
    candidate. Returns ``(nm_per_px, selected, candidates, ignored_regions)``,
    where ``ignored_regions`` masks the bar out of particle detection.

    Raises :class:`~corpus.errors.CalibrationError` when nothing can be used.
    """
    candidates = collect_scale_candidates(image)

    if manual_scale_line:
        padding = 8
        min_x = int(math.floor(min(manual_scale_line["x1"], manual_scale_line["x2"]))) - padding
        min_y = int(math.floor(min(manual_scale_line["y1"], manual_scale_line["y2"]))) - padding
        max_x = int(math.ceil(max(manual_scale_line["x1"], manual_scale_line["x2"]))) + padding
        max_y = int(math.ceil(max(manual_scale_line["y1"], manual_scale_line["y2"]))) + padding
        height, width = image.shape[:2]
        x = max(0, min_x)
        y = max(0, min_y)
        w = max(1, min(width, max_x) - x)
        h = max(1, min(height, max_y) - y)
        selected = {
            "x": x,
            "y": y,
            "width_px": float(manual_scale_line["length"]),
            "height_px": float(max(1, h)),
            "method": "manual_line",
            "confidence": 1.0,
            "line": {
                "x1": float(manual_scale_line["x1"]),
                "y1": float(manual_scale_line["y1"]),
                "x2": float(manual_scale_line["x2"]),
                "y2": float(manual_scale_line["y2"]),
            },
        }
        ignored_regions = [(x, y, w, h)]
    elif manual_scale_px and float(manual_scale_px) > 0:
        selected = {
            "x": "",
            "y": "",
            "width_px": float(manual_scale_px),
            "height_px": "",
            "method": "manual_override",
            "confidence": 1.0,
        }
        ignored_regions = []
    elif candidates:
        selected = candidates[0]
        ignored_regions = [(
            int(selected["x"]),
            int(selected["y"]),
            int(round(selected["width_px"])),
            max(1, int(round(float(selected["height_px"])))),
        )]
    else:
        raise CalibrationError(
            "Could not detect a white scale bar. Enter Manual Scale px or use an image with a visible scale bar."
        )

    return nm_per_pixel(scale_length, selected["width_px"]), selected, candidates, ignored_regions
