"""Pure geometry helpers shared by calibration and measurement.

Nothing in this module touches OpenCV, so it is the cheapest layer to test.
"""

import math

__all__ = [
    "overlap_ratio",
    "touches_edge",
    "circularity",
    "aspect_ratio",
    "equivalent_diameter_px",
    "classify_shape",
    "distance",
    "is_duplicate",
    "nearest_measurement",
]


def overlap_ratio(first, second):
    """Fraction of ``first`` covered by ``second`` (both ``(x, y, w, h)``)."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return (ix * iy) / max(1, aw * ah)


def touches_edge(bbox, image_shape, margin=2):
    """True when ``bbox`` reaches within ``margin`` pixels of the frame."""
    x, y, w, h = bbox
    height, width = image_shape[:2]
    return x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin


def circularity(area, perimeter):
    """Isoperimetric circularity ``4*pi*A / P^2``; 1.0 for a perfect disc."""
    if area <= 0 or perimeter <= 0:
        return 0.0
    return 4 * math.pi * area / (perimeter * perimeter)


def aspect_ratio(major_px, minor_px):
    """Elongation ``major/minor`` with a floor on the denominator."""
    return major_px / max(minor_px, 1e-6)


def equivalent_diameter_px(area_px):
    """Diameter of the disc with the same area: ``sqrt(4A/pi)``."""
    if area_px <= 0:
        return 0.0
    return math.sqrt(4 * area_px / math.pi)


def classify_shape(shape_circularity, shape_aspect_ratio):
    """Return ``"round"``, ``"elongated"`` or ``None`` when neither applies.

    These thresholds define which contours Corpus accepts at all, so they are
    kept in one place and covered by tests rather than inlined per detector.
    """
    is_round = shape_circularity >= 0.55 and shape_aspect_ratio < 1.8
    if is_round:
        return "round"
    is_elongated = shape_circularity >= 0.25 and shape_aspect_ratio >= 1.8
    if is_elongated:
        return "elongated"
    return None


def distance(first, second):
    """Euclidean distance between two ``(x, y)`` points."""
    return math.hypot(first[0] - second[0], first[1] - second[1])


def is_duplicate(measurement, measurements, factor=0.65):
    """True when ``measurement`` sits inside an already accepted detection."""
    for existing in measurements:
        dx = measurement["center_x"] - existing["center_x"]
        dy = measurement["center_y"] - existing["center_y"]
        separation = math.sqrt(dx * dx + dy * dy)
        radius = max(measurement["radius_px"], existing["radius_px"], 1)
        if separation < radius * factor:
            return True
    return False


def nearest_measurement(center, candidates, max_distance):
    """Closest candidate to ``center`` within ``max_distance``.

    Returns ``(candidate, distance)`` or ``(None, None)``.
    """
    best = None
    best_distance = None
    for candidate in candidates:
        dx = center[0] - candidate["center_x"]
        dy = center[1] - candidate["center_y"]
        separation = math.sqrt(dx * dx + dy * dy)
        if separation <= max_distance and (best_distance is None or separation < best_distance):
            best = candidate
            best_distance = separation
    return best, best_distance
