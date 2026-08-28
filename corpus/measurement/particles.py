"""Turn contours into calibrated per-particle measurement records.

Every record is a flat dict in nanometres plus the pixel quantities needed to
re-derive it, so ``measurements.json`` stays auditable without the source
image. ``overlay`` is optional: pass ``None`` to measure head-less.
"""

import math

import cv2
import numpy as np

from corpus.measurement.geometry import (
    aspect_ratio as _aspect_ratio,
)
from corpus.measurement.geometry import (
    circularity as _circularity,
)
from corpus.measurement.geometry import (
    classify_shape,
    equivalent_diameter_px,
    is_duplicate,
    overlap_ratio,
    touches_edge,
)
from corpus.review.overlay import draw_detection

__all__ = ["flat_measurement", "contour_measurements", "hough_circle_measurements"]

#: A contour may exceed the nominal max-radius disc area by this factor before
#: being rejected, which keeps slightly irregular particles measurable.
MAX_AREA_TOLERANCE = 3.0

#: ...and may fall this far below the nominal min-radius disc area.
MIN_AREA_TOLERANCE = 0.65


def flat_measurement(class_name, center_x, center_y, major_px, minor_px, area_px, angle,
                     shape, confidence, nm_per_px, flags=None, separation_method="contour",
                     backend="classical"):
    """One calibrated particle record.

    ``diameter`` is the major axis of the minimum-area rectangle, which is the
    quantity a human reproduces when measuring the same particle by hand.
    """
    flags = flags or []
    major_axis = float(major_px * nm_per_px)
    minor_axis = float(minor_px * nm_per_px)
    return {
        "class": class_name,
        "diameter": major_axis,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "equivalent_diameter": float(equivalent_diameter_px(area_px) * nm_per_px),
        "radius_px": float(major_px / 2),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "area_px": float(area_px),
        "aspect_ratio": round(float(_aspect_ratio(major_px, minor_px)), 4),
        "shape": shape,
        "angle": float(angle),
        "separation_method": separation_method,
        "backend": backend,
        "confidence_hint": round(float(confidence), 4),
        "flags": flags,
    }


def contour_measurements(contours, class_name, min_radius_px, max_radius_px, nm_per_px, color,
                         overlay, ignored_regions, exclude_edges, measurement_flags=None,
                         separation_method="contour", filter_settings=None, image_shape=None,
                         backend="classical"):
    """Measure every contour that survives the size, shape and region filters."""
    measurements = []
    min_area = math.pi * min_radius_px**2
    max_area = math.pi * max_radius_px**2 * MAX_AREA_TOLERANCE
    measurement_flags = measurement_flags or []
    filter_settings = filter_settings or {}
    min_circularity = filter_settings.get("min_circularity", 0)
    max_circularity = filter_settings.get("max_circularity", 1)
    min_elongation = filter_settings.get("min_elongation", 1)
    max_elongation = filter_settings.get("max_elongation", 999)
    if image_shape is None:
        if overlay is None:
            raise ValueError("contour_measurements needs either an overlay or an explicit image_shape.")
        image_shape = overlay.shape

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area <= 0 or perimeter <= 0:
            continue

        bbox = cv2.boundingRect(contour)
        edge = touches_edge(bbox, image_shape)
        if exclude_edges and edge:
            continue
        if any(overlap_ratio(bbox, region) > 0.2 for region in ignored_regions):
            continue

        rect = cv2.minAreaRect(contour)
        (x, y), (width, height), angle = rect
        major_px = max(width, height)
        minor_px = min(width, height)
        if major_px <= 0 or minor_px <= 0:
            continue

        shape_circularity = _circularity(area, perimeter)
        shape_aspect = _aspect_ratio(major_px, minor_px)
        radius_px = major_px / 2
        if not (min_circularity <= shape_circularity <= max_circularity):
            continue
        if not (min_elongation <= shape_aspect <= max_elongation):
            continue
        shape = classify_shape(shape_circularity, shape_aspect)
        if shape is None:
            continue
        if not (min_radius_px <= radius_px <= max_radius_px):
            continue
        if not (min_area * MIN_AREA_TOLERANCE <= area <= max_area):
            continue

        flags = list(measurement_flags)
        if "watershed_split" in flags and (area < min_area * 0.55 or shape_circularity < 0.35):
            flags.append("low_split_confidence")
        if edge:
            flags.append("edge")

        draw_detection(overlay, contour, rect, shape, color)
        measurements.append(flat_measurement(
            class_name=class_name,
            center_x=x,
            center_y=y,
            major_px=major_px,
            minor_px=minor_px,
            area_px=area,
            angle=angle,
            shape=shape,
            confidence=shape_circularity,
            nm_per_px=nm_per_px,
            flags=flags,
            separation_method=separation_method,
            backend=backend,
        ))

    return measurements


def hough_circle_measurements(gray, class_name, min_radius_px, max_radius_px, nm_per_px, color,
                              overlay, ignored_regions, exclude_edges, image_shape=None,
                              backend="classical"):
    """Detect round carriers by Hough transform when thresholding fails.

    Used for large low-contrast spheres whose edges survive gradient detection
    even though their interior grey level overlaps the background.
    """
    measurements = []
    if image_shape is None:
        image_shape = overlay.shape if overlay is not None else gray.shape

    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, int(min_radius_px * 1.7)),
        param1=90,
        param2=28,
        minRadius=max(1, int(round(min_radius_px))),
        maxRadius=max(2, int(round(max_radius_px))),
    )
    if circles is None:
        return measurements

    image_area = gray.shape[0] * gray.shape[1]
    max_circle_area = image_area * 0.25

    for circle in np.round(circles[0]).astype(int):
        x, y, radius = [int(value) for value in circle]
        if math.pi * radius * radius > max_circle_area:
            continue
        bbox = (x - radius, y - radius, radius * 2, radius * 2)
        edge = touches_edge(bbox, image_shape)
        if exclude_edges and edge:
            continue
        if any(overlap_ratio(bbox, region) > 0.2 for region in ignored_regions):
            continue
        measurement = flat_measurement(
            class_name=class_name,
            center_x=x,
            center_y=y,
            major_px=2 * radius,
            minor_px=2 * radius,
            area_px=math.pi * radius * radius,
            angle=0,
            shape="round",
            confidence=0.75,
            nm_per_px=nm_per_px,
            flags=["edge"] if edge else [],
            separation_method="hough",
            backend=backend,
        )
        if is_duplicate(measurement, measurements):
            continue
        if overlay is not None:
            cv2.circle(overlay, (x, y), radius, color, 2)
        measurements.append(measurement)
    return measurements
