"""Corpus measurement CLI -- the contract used by the Electron main process.

This file is intentionally thin. The scientific logic lives in the ``corpus``
package so it can be imported and tested without Electron:

    corpus.calibration  nm/pixel from a scale bar
    corpus.segmentation binarisation, watershed, backend contract
    corpus.measurement  contour geometry -> calibrated particle records
    corpus.metrology    core-shell derivations, distribution summaries
    corpus.review       confidence scoring, review status, overlays
    corpus.io           image decoding, exports, run provenance

What remains here is argument parsing, the per-preset orchestration
(``run_spheres`` / ``run_pellets`` / ``run_generic`` / ``run_decorated``), and
the JSON envelope. Names migrated into the package are re-exported below so
existing imports of this module keep working.

Usage (also see ``python -m corpus.dev.smoke``):

    python measurement_modes.py --image IMG --scale 100 --shape-preset spheres
"""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from corpus import __version__ as CORPUS_VERSION
from corpus.calibration import collect_scale_candidates, nm_per_pixel
from corpus.calibration import parse_scale_line as _parse_scale_line
from corpus.calibration import resolve_scale as _resolve_scale
from corpus.calibration import scale_bar_confidence as scale_confidence
from corpus.errors import CalibrationError, CorpusError
from corpus.io import (
    DIAMETERS_TXT,
    MEASUREMENTS_JSON,
    OUTPUT_IMAGE,
    read_image,
    run_fingerprint,
    write_compat_files,
    write_measurements_json,
)
from corpus.measurement import (
    clamp_filter_settings,
    contour_measurements,
    flat_measurement,
    hough_circle_measurements,
    is_duplicate,
    nearest_measurement,
    overlap_ratio,
    parse_bool,
    resolve_watershed,
    touches_edge,
)
from corpus.metrology import (
    build_normality_report,
    normality_stats,
    object_row,
    summarize_class_measurements,
    summarize_decorated,
    summarize_objects,
)
from corpus.review import (
    INNER_COLOR,
    OUTER_COLOR,
    REVIEW_COLOR,
    build_warnings,
    draw_detection,
    draw_review_labels,
    draw_review_markers,
)
from corpus.segmentation import particle_binary, sio2_mask, watershed_split_contours

AU_CLASS = "Au_decorations"
SIO2_CLASS = "SiO2_carrier"

#: Names this module has always exported. Several are now re-exports from the
#: `corpus` package; they stay listed here so code that imported them from
#: `measurement_modes` keeps working through the migration.
__all__ = [
    # preset orchestration, still implemented here
    "main", "parse_args", "fail",
    "detect_au_contours", "detect_sio2_watershed", "detect_sio2_generic",
    "anchored_pellet_outer", "run_spheres", "run_pellets", "run_generic", "run_decorated",
    # constants
    "AU_CLASS", "SIO2_CLASS", "SEGMENTATION_BACKEND",
    "OUTPUT_IMAGE", "DIAMETERS_TXT", "MEASUREMENTS_JSON",
    "INNER_COLOR", "OUTER_COLOR", "REVIEW_COLOR",
    # re-exported from corpus.calibration
    "collect_scale_candidates", "nm_per_pixel", "scale_confidence",
    "parse_manual_scale_line", "resolve_scale",
    # re-exported from corpus.segmentation
    "particle_binary", "sio2_mask", "watershed_split_contours",
    # re-exported from corpus.measurement
    "clamp_filter_settings", "contour_measurements", "flat_measurement",
    "hough_circle_measurements", "is_duplicate", "nearest_measurement",
    "overlap_ratio", "parse_bool", "resolve_watershed", "touches_edge",
    # re-exported from corpus.metrology
    "build_normality_report", "normality_stats", "object_row",
    "summarize_class_measurements", "summarize_decorated", "summarize_objects",
    # re-exported from corpus.review
    "build_warnings", "draw_detection", "draw_review_labels", "draw_review_markers",
    # re-exported from corpus.io
    "read_image", "run_fingerprint", "write_compat_files", "write_measurements_json",
]

#: Backend that produced the masks in this build. See corpus.segmentation.backends.
SEGMENTATION_BACKEND = "classical"


def fail(message):
    print(json.dumps({"ok": False, "message": message}))
    raise SystemExit(1)


def parse_manual_scale_line(value):
    """CLI wrapper: report a bad manual scale line as JSON instead of a traceback."""
    try:
        return _parse_scale_line(value)
    except CalibrationError as error:
        fail(str(error))


def resolve_scale(image, scale_length, manual_scale_px, manual_scale_line=None):
    """CLI wrapper around :func:`corpus.calibration.resolve_scale`."""
    try:
        return _resolve_scale(image, scale_length, manual_scale_px, manual_scale_line)
    except CalibrationError as error:
        fail(str(error))


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Au/SiO2 particles without AI.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mode", choices=["au", "sio2", "both"], default="au")
    parser.add_argument("--shape-preset", choices=["generic", "spheres", "pellets", "decorated"], default="generic")
    parser.add_argument("--ui-mode", choices=["easy", "advanced"], default="easy")
    parser.add_argument("--contrast-strategy", choices=["dark_particles", "bright_shells", "manual_gray_range"], default="dark_particles")
    parser.add_argument("--manual-threshold-min", type=float, default=0)
    parser.add_argument("--manual-threshold-max", type=float, default=255)
    parser.add_argument("--min-circularity", type=float, default=0)
    parser.add_argument("--max-circularity", type=float, default=1)
    parser.add_argument("--min-elongation", type=float, default=1)
    parser.add_argument("--max-elongation", type=float, default=999)
    parser.add_argument("--include-holes", default="false")
    parser.add_argument("--review-view", choices=["overlay", "numbered"], default="overlay")
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--manual-scale-px", type=float, default=0)
    parser.add_argument("--manual-scale-line", default="")
    parser.add_argument("--exclude-edges", default="true")
    parser.add_argument("--watershed", default="auto")
    parser.add_argument("--watershed-min-distance-factor", type=float, default=0.55)
    parser.add_argument("--au-min-radius", type=float, default=1)
    parser.add_argument("--au-max-radius", type=float, default=50)
    parser.add_argument("--sio2-min-radius", type=float, default=20)
    parser.add_argument("--sio2-max-radius", type=float, default=500)
    parser.add_argument("--histogram-bin-width", type=float, default=5)
    return parser.parse_args()


def detect_au_contours(image, min_radius_px, max_radius_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled=False, watershed_factor=0.55, filter_settings=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filter_settings = filter_settings or {}
    binary = particle_binary(gray, filter_settings, 80)
    watershed_binary = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    original_contours, _ = cv2.findContours(watershed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    legacy_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if watershed_enabled:
        contours = watershed_split_contours(watershed_binary, min_radius_px, max_radius_px, watershed_factor)
        if len(contours) > len(original_contours):
            return contour_measurements(
                contours,
                AU_CLASS,
                min_radius_px,
                max_radius_px,
                nm_per_px,
                INNER_COLOR,
                overlay,
                ignored_regions,
                exclude_edges,
                ["watershed_split"],
                "watershed",
                filter_settings
            )
    return contour_measurements(legacy_contours, AU_CLASS, min_radius_px, max_radius_px, nm_per_px, INNER_COLOR, overlay, ignored_regions, exclude_edges, filter_settings=filter_settings)


def detect_sio2_watershed(image, min_radius_px, max_radius_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_factor=0.55, filter_settings=None):
    filter_settings = filter_settings or {}
    binary = sio2_mask(image, filter_settings.get("include_holes", False))
    original_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = watershed_split_contours(binary, min_radius_px, max_radius_px, watershed_factor)
    if len(contours) <= len(original_contours):
        return []
    return contour_measurements(
        contours,
        SIO2_CLASS,
        min_radius_px,
        max_radius_px,
        nm_per_px,
        OUTER_COLOR,
        overlay,
        ignored_regions,
        exclude_edges,
        ["watershed_split"],
        "watershed",
        filter_settings
    )


def run_spheres(image, mode, au_min_px, au_max_px, sio2_min_px, sio2_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled=False, watershed_factor=0.55, filter_settings=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inner = detect_au_contours(image, au_min_px, au_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled, watershed_factor, filter_settings) if mode in ("au", "both") else []
    outer = hough_circle_measurements(gray, SIO2_CLASS, sio2_min_px, sio2_max_px, nm_per_px, OUTER_COLOR, overlay, ignored_regions, exclude_edges) if mode in ("sio2", "both") else []
    if watershed_enabled and mode in ("sio2", "both"):
        for row in detect_sio2_watershed(image, sio2_min_px, sio2_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_factor, filter_settings):
            if not is_duplicate(row, outer):
                outer.append(row)

    objects = []
    used_inner = set()
    index = 1
    for outer_measurement in outer:
        match, _ = nearest_measurement(
            (outer_measurement["center_x"], outer_measurement["center_y"]),
            [row for row in inner if id(row) not in used_inner],
            max(outer_measurement["radius_px"] * 0.8, 8)
        )
        if match:
            used_inner.add(id(match))
        objects.append(object_row(index, "spheres", match, outer_measurement))
        index += 1

    if mode == "au":
        for row in inner:
            objects.append(object_row(index, "spheres", row, None))
            index += 1
    elif mode == "both":
        for row in inner:
            if id(row) not in used_inner:
                objects.append(object_row(index, "spheres", row, None))
                index += 1

    return objects, inner + outer


def anchored_pellet_outer(gray, anchors, min_radius_px, max_radius_px, nm_per_px, overlay, ignored_regions, exclude_edges):
    measurements = []
    window_radius = max(12, int(round(max_radius_px * 0.86)))
    for anchor in anchors:
        center_x = int(round(anchor["center_x"]))
        center_y = int(round(anchor["center_y"]))
        x1 = max(0, center_x - window_radius)
        y1 = max(0, center_y - window_radius)
        x2 = min(gray.shape[1], center_x + window_radius)
        y2 = min(gray.shape[0], center_y + window_radius)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        _, binary = cv2.threshold(patch, 175, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        label_count, labels, _, _ = cv2.connectedComponentsWithStats(binary)
        local_x = center_x - x1
        local_y = center_y - y1
        if not (0 <= local_x < labels.shape[1] and 0 <= local_y < labels.shape[0]):
            continue
        label = labels[local_y, local_x]
        if label == 0 or label_count <= label:
            continue

        mask = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area <= 0:
            continue

        rect = cv2.minAreaRect(contour)
        (local_cx, local_cy), (width, height), angle = rect
        major_px = max(width, height)
        minor_px = min(width, height)
        if minor_px <= 0:
            continue
        radius_px = major_px / 2
        if not (min_radius_px <= radius_px <= max_radius_px * 1.15):
            continue

        global_cx = local_cx + x1
        global_cy = local_cy + y1
        bbox = (int(global_cx - major_px / 2), int(global_cy - major_px / 2), int(major_px), int(major_px))
        edge = touches_edge(bbox, gray.shape)
        if exclude_edges and edge:
            continue
        if any(overlap_ratio(bbox, region) > 0.2 for region in ignored_regions):
            continue

        box = cv2.boxPoints(rect)
        box[:, 0] += x1
        box[:, 1] += y1
        box = np.intp(box)
        cv2.drawContours(overlay, [box], -1, OUTER_COLOR, 2)
        measurements.append(flat_measurement(
            class_name=SIO2_CLASS,
            center_x=global_cx,
            center_y=global_cy,
            major_px=major_px,
            minor_px=minor_px,
            area_px=area,
            angle=angle,
            shape="elongated",
            confidence=0.7,
            nm_per_px=nm_per_px,
            flags=["edge"] if edge else []
        ))

    return measurements


def run_pellets(image, mode, au_min_px, au_max_px, sio2_min_px, sio2_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled=False, watershed_factor=0.55, filter_settings=None):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    inner = detect_au_contours(image, au_min_px, au_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled, watershed_factor, filter_settings) if mode in ("au", "both") else []
    anchors = [row for row in inner if row.get("aspect_ratio", 1) >= 1.45] or inner
    outer = anchored_pellet_outer(gray, anchors, sio2_min_px, sio2_max_px, nm_per_px, overlay, ignored_regions, exclude_edges) if mode in ("sio2", "both") else []

    objects = []
    used_outer = set()
    index = 1
    for inner_measurement in inner:
        match, _ = nearest_measurement(
            (inner_measurement["center_x"], inner_measurement["center_y"]),
            [row for row in outer if id(row) not in used_outer],
            max(inner_measurement["radius_px"] * 1.4, 12)
        )
        if match:
            used_outer.add(id(match))
        objects.append(object_row(index, "pellets", inner_measurement, match))
        index += 1

    if mode == "sio2":
        for row in outer:
            objects.append(object_row(index, "pellets", None, row))
            index += 1
    elif mode == "both":
        for row in outer:
            if id(row) not in used_outer:
                objects.append(object_row(index, "pellets", None, row))
                index += 1

    return objects, inner + outer


def detect_sio2_generic(image, min_radius_px, max_radius_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled=False, watershed_factor=0.55, filter_settings=None):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    filter_settings = filter_settings or {}
    closed = sio2_mask(image, filter_settings.get("include_holes", False))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if watershed_enabled:
        split_contours = watershed_split_contours(closed, min_radius_px, max_radius_px, watershed_factor)
        if len(split_contours) > len(contours):
            measurements = contour_measurements(split_contours, SIO2_CLASS, min_radius_px, max_radius_px, nm_per_px, OUTER_COLOR, overlay, ignored_regions, exclude_edges, ["watershed_split"], "watershed", filter_settings)
        else:
            measurements = contour_measurements(contours, SIO2_CLASS, min_radius_px, max_radius_px, nm_per_px, OUTER_COLOR, overlay, ignored_regions, exclude_edges, filter_settings=filter_settings)
    else:
        measurements = contour_measurements(contours, SIO2_CLASS, min_radius_px, max_radius_px, nm_per_px, OUTER_COLOR, overlay, ignored_regions, exclude_edges, filter_settings=filter_settings)

    for row in hough_circle_measurements(gray, SIO2_CLASS, min_radius_px, max_radius_px, nm_per_px, OUTER_COLOR, overlay, ignored_regions, exclude_edges):
        if not is_duplicate(row, measurements):
            measurements.append(row)
    return measurements


def run_generic(image, mode, au_min_px, au_max_px, sio2_min_px, sio2_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled=False, watershed_factor=0.55, filter_settings=None):
    inner = detect_au_contours(image, au_min_px, au_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled, watershed_factor, filter_settings) if mode in ("au", "both") else []
    outer = detect_sio2_generic(image, sio2_min_px, sio2_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled, watershed_factor, filter_settings) if mode in ("sio2", "both") else []
    objects = []
    index = 1
    for row in inner:
        objects.append(object_row(index, "generic", row, None))
        index += 1
    for row in outer:
        objects.append(object_row(index, "generic", None, row))
        index += 1
    return objects, inner + outer


def run_decorated(image, mode, au_min_px, au_max_px, sio2_min_px, sio2_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled=False, watershed_factor=0.55, filter_settings=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    decorations = detect_au_contours(image, au_min_px, au_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled, watershed_factor, filter_settings) if mode in ("au", "both") else []
    carriers = hough_circle_measurements(gray, SIO2_CLASS, sio2_min_px, sio2_max_px, nm_per_px, OUTER_COLOR, overlay, ignored_regions, exclude_edges) if mode in ("sio2", "both") else []
    if mode in ("sio2", "both"):
        for row in detect_sio2_generic(image, sio2_min_px, sio2_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled, watershed_factor, filter_settings):
            if not is_duplicate(row, carriers, factor=0.8):
                carriers.append(row)

    objects = []
    assigned_decorations = set()
    index = 1
    for carrier in carriers:
        radius_px = max(carrier.get("radius_px", 0), 1)
        inside = []
        for decoration in decorations:
            dx = decoration["center_x"] - carrier["center_x"]
            dy = decoration["center_y"] - carrier["center_y"]
            if math.hypot(dx, dy) <= radius_px * 1.05:
                inside.append(decoration)
                assigned_decorations.add(id(decoration))

        row = object_row(index, "decorated", None, carrier)
        row["pair_status"] = "decorated_carrier"
        row["decoration_count"] = len(inside)
        projected_area = math.pi * max(carrier.get("major_axis", 0) / 2, 1e-6) * max(carrier.get("minor_axis", 0) / 2, 1e-6)
        row["decoration_density_per_1000_nm2"] = (len(inside) / projected_area) * 1000 if projected_area > 0 else 0
        row["mean_decoration_diameter"] = sum(item["diameter"] for item in inside) / len(inside) if inside else 0
        row["flags"] = sorted({flag for flag in row["flags"] if flag != "unpaired_inner"})
        if inside:
            row["review_status"] = "ready" if row["confidence_score"] >= 0.7 else "needs_review"
        else:
            row["review_status"] = "needs_review"
            row["flags"] = sorted(set(row["flags"] + ["no_decorations_detected"]))
        objects.append(row)
        index += 1

    if mode in ("au", "both"):
        for decoration in decorations:
            if mode == "au" or id(decoration) not in assigned_decorations:
                objects.append(object_row(index, "decorated", decoration, None, ["unassigned_decoration"]))
                index += 1

    return objects, decorations + carriers


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        fail(f"Image not found: {image_path}")

    image = read_image(image_path)
    if image is None:
        fail(f"Could not read image: {image_path}")

    filter_settings = clamp_filter_settings(args)
    exclude_edges = parse_bool(args.exclude_edges)
    watershed_enabled = resolve_watershed(args.watershed, args.shape_preset)
    manual_scale_line = parse_manual_scale_line(args.manual_scale_line)
    nm_per_px, selected_scale, scale_candidates, ignored_regions = resolve_scale(
        image,
        args.scale,
        args.manual_scale_px,
        manual_scale_line
    )
    overlay = image.copy()

    au_min_px = args.au_min_radius / nm_per_px
    au_max_px = args.au_max_radius / nm_per_px
    sio2_min_px = args.sio2_min_radius / nm_per_px
    sio2_max_px = args.sio2_max_radius / nm_per_px

    if args.shape_preset == "spheres":
        objects, class_measurements = run_spheres(image, args.mode, au_min_px, au_max_px, sio2_min_px, sio2_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled, args.watershed_min_distance_factor, filter_settings)
    elif args.shape_preset == "pellets":
        objects, class_measurements = run_pellets(image, args.mode, au_min_px, au_max_px, sio2_min_px, sio2_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled, args.watershed_min_distance_factor, filter_settings)
    elif args.shape_preset == "decorated":
        objects, class_measurements = run_decorated(image, args.mode, au_min_px, au_max_px, sio2_min_px, sio2_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled, args.watershed_min_distance_factor, filter_settings)
    else:
        objects, class_measurements = run_generic(image, args.mode, au_min_px, au_max_px, sio2_min_px, sio2_max_px, nm_per_px, overlay, ignored_regions, exclude_edges, watershed_enabled, args.watershed_min_distance_factor, filter_settings)

    draw_review_markers(overlay, objects)

    if filter_settings["review_view"] == "numbered":
        draw_review_labels(overlay, objects)

    cv2.imwrite(OUTPUT_IMAGE, overlay)
    fingerprint = run_fingerprint(
        image_path,
        {
            "mode": args.mode,
            "shape_preset": args.shape_preset,
            "scale_nm": args.scale,
            "manual_scale_px": args.manual_scale_px,
            "manual_scale_line": args.manual_scale_line,
            "exclude_edges": exclude_edges,
            "watershed": watershed_enabled,
            "watershed_min_distance_factor": args.watershed_min_distance_factor,
            "au_min_radius": args.au_min_radius,
            "au_max_radius": args.au_max_radius,
            "sio2_min_radius": args.sio2_min_radius,
            "sio2_max_radius": args.sio2_max_radius,
            **filter_settings,
        },
        CORPUS_VERSION,
    )
    payload = {
        "ok": True,
        "mode": args.mode,
        "shape_preset": args.shape_preset,
        "exclude_edges": exclude_edges,
        "watershed": watershed_enabled,
        "separation_method": "watershed" if watershed_enabled else "contour/hough",
        "analysis_settings": {
            "ui_mode": args.ui_mode,
            "measurement_mode": args.mode,
            "shape_preset": args.shape_preset,
            "scale_nm": args.scale,
            "histogram_bin_width": args.histogram_bin_width,
            "review_view": args.review_view
        },
        "filter_settings": filter_settings,
        "corpus_version": CORPUS_VERSION,
        "segmentation_backend": SEGMENTATION_BACKEND,
        "run_fingerprint": fingerprint,
        "processed_image_path": os.path.abspath(OUTPUT_IMAGE),
        "measurements_path": os.path.abspath(MEASUREMENTS_JSON),
        "nm_per_px": nm_per_px,
        "scale_bar_px": selected_scale["width_px"],
        "selected_scale": selected_scale,
        "scale_candidates": scale_candidates,
        "histogram_bin_width": args.histogram_bin_width,
        "measurements": objects,
        "class_measurements": class_measurements,
        "summary": summarize_class_measurements(class_measurements),
        "object_summary": summarize_objects(objects),
        "normality_report": build_normality_report(objects, class_measurements),
        "manual_edits": [],
        "decorated_particle_metrics": summarize_decorated(objects),
        "warnings": build_warnings(image, objects, selected_scale)
    }

    preferred_class = AU_CLASS if args.mode != "sio2" else SIO2_CLASS
    write_compat_files(nm_per_px, class_measurements, preferred_class)
    write_measurements_json(payload)

    print(json.dumps(payload))


if __name__ == "__main__":
    try:
        main()
    except CorpusError as error:
        fail(str(error))
