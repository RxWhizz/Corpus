"""Overlay rendering for human review.

Drawing lives here rather than in :mod:`corpus.measurement` so that the
numerical layer can be exercised head-less with ``overlay=None``.
"""

import cv2
import numpy as np

__all__ = [
    "INNER_COLOR",
    "OUTER_COLOR",
    "REVIEW_COLOR",
    "draw_detection",
    "draw_review_labels",
    "draw_review_markers",
]

INNER_COLOR = (0, 0, 255)
OUTER_COLOR = (255, 180, 0)
REVIEW_COLOR = (0, 220, 255)


def draw_detection(overlay, contour, rect, shape, color):
    """Outline one detection: rotated box for rods, circle for round particles."""
    if overlay is None:
        return
    if shape == "elongated":
        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(overlay, [box], -1, color, 2)
    else:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        cv2.circle(overlay, (int(x), int(y)), int(radius), color, 2)
    cv2.drawContours(overlay, [contour], -1, color, 1)


def draw_review_labels(overlay, objects):
    """Stamp the numeric part of each ``object_id`` next to its centre."""
    if overlay is None:
        return
    for row in objects:
        x = int(row.get("center_x", 0))
        y = int(row.get("center_y", 0))
        label = str(row.get("object_id", "")).replace("obj_", "")
        if not label:
            continue
        cv2.putText(overlay, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(overlay, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def draw_review_markers(overlay, objects):
    """Dot every object that still needs a human decision."""
    if overlay is None:
        return
    for row in objects:
        if row.get("review_status") == "ready":
            continue
        x = int(row.get("center_x", 0))
        y = int(row.get("center_y", 0))
        cv2.circle(overlay, (x, y), 4, REVIEW_COLOR, -1)
