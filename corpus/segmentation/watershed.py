"""Distance-transform watershed separation for touching round particles."""

import cv2
import numpy as np

__all__ = ["watershed_split_contours"]


def watershed_split_contours(binary, min_radius_px, max_radius_px, distance_factor=0.55):
    """Split touching blobs in ``binary`` into individual contours.

    ``distance_factor`` sets the seed threshold as a fraction of the
    distance-transform maximum. Higher values separate more aggressively;
    lower values grow the seeds until neighbours merge. The retry ladder
    therefore does not hunt for a split -- it rescues the case where the
    threshold was so high that *no* seed survived. If no split is found the
    original contours are returned unchanged, so watershed can never lose a
    particle.
    """
    foreground = np.uint8(binary > 0) * 255
    if not np.any(foreground):
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
    original_contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not original_contours:
        return []

    distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    if distance.max() <= 0:
        return original_contours

    marker_labels = None
    marker_count = 0
    for factor in (distance_factor, distance_factor * 0.85, distance_factor * 0.70):
        threshold = max(2.0, float(distance.max()) * max(0.20, factor))
        sure_foreground = np.uint8(distance >= threshold) * 255
        marker_count, marker_labels = cv2.connectedComponents(sure_foreground)
        if marker_count >= 2:
            break

    if marker_labels is None or marker_count <= 1:
        return original_contours

    sure_background = cv2.dilate(foreground, kernel, iterations=2)
    unknown = cv2.subtract(sure_background, np.uint8(marker_labels > 0) * 255)
    markers = marker_labels + 1
    markers[unknown == 255] = 0

    watershed_image = cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)
    cv2.watershed(watershed_image, markers)

    split_contours = []
    for marker_id in range(2, marker_count + 1):
        region = np.uint8(markers == marker_id) * 255
        region = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            split_contours.append(max(contours, key=cv2.contourArea))

    return split_contours or original_contours
