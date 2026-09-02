import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from common_training import file_sha256, layers_for, write_json

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

NORMALIZED_MANIFEST_FIELDS = [
    "image_id",
    "file_name",
    "original_image_path",
    "source_id",
    "group_key",
    "split",
    "dataset_layer",
    "content_layer",
    "distribution_layer",
    "width",
    "height",
    "instances",
    "file_sha256",
    "license",
    "license_status",
    "doi",
    "source_url",
    "figure_label",
    "panel_label",
    "caption",
]


def reset_output_dir(output_dir, clean=False):
    output_dir = Path(output_dir)
    if output_dir.exists() and clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    return output_dir


def image_paths(image_dir):
    image_dir = Path(image_dir)
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_metadata_csv(path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            keys = [
                Path(row.get("file_name", "")).stem,
                Path(row.get("filename", "")).stem,
                str(row.get("image_id", "")),
                str(row.get("id", "")),
            ]
            for key in keys:
                if key:
                    rows[key] = row
    return rows


def read_image_size(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return width, height


def polygon_area(points):
    pairs = list(zip(points[0::2], points[1::2]))
    if len(pairs) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(pairs):
        x2, y2 = pairs[(index + 1) % len(pairs)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def polygon_bbox(points):
    xs = points[0::2]
    ys = points[1::2]
    if not xs or not ys:
        return [0, 0, 0, 0]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def contour_to_polygon(contour, min_area=8):
    area = cv2.contourArea(contour)
    if area < min_area:
        return None
    epsilon = max(0.7, 0.0025 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    values = []
    for x, y in approx:
        values.extend([float(x), float(y)])
    return values


def instance_mask_polygons(mask_path, min_area=8):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    labels = sorted(int(value) for value in np.unique(mask) if int(value) != 0)
    polygons = []
    for label in labels:
        binary = (mask == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            polygon = contour_to_polygon(contour, min_area=min_area)
            if polygon:
                polygons.append(polygon)
    return polygons


def group_key(row, fallback):
    for field in (
        "micrograph_id",
        "original_micrograph_id",
        "parent_micrograph_id",
        "figure_id",
        "source_figure_id",
        "source_id",
        "source_group",
        "doi",
        "acquisition_session",
    ):
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return fallback


def manifest_row(image_id, file_name, original_path, width, height, instances, meta, default_dataset_layer):
    row = dict(meta or {})
    row.setdefault("dataset_layer", default_dataset_layer)
    layer_info = layers_for(row)
    source_id = row.get("source_id") or row.get("source_group") or row.get("doi") or image_id
    return {
        "image_id": image_id,
        "file_name": file_name,
        "original_image_path": str(original_path),
        "source_id": source_id,
        "group_key": group_key(row, source_id),
        "split": row.get("split", ""),
        "dataset_layer": row.get("dataset_layer", layer_info["dataset_layer"]),
        "content_layer": layer_info["content_layer"],
        "distribution_layer": layer_info["distribution_layer"],
        "width": width,
        "height": height,
        "instances": instances,
        "file_sha256": file_sha256(original_path),
        "license": row.get("license", ""),
        "license_status": row.get("license_status", ""),
        "doi": row.get("doi", ""),
        "source_url": row.get("source_url", ""),
        "figure_label": row.get("figure_label", ""),
        "panel_label": row.get("panel_label", ""),
        "caption": row.get("caption", ""),
    }


def write_manifest_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in NORMALIZED_MANIFEST_FIELDS})


def write_normalized_dataset(output_dir, coco, manifest_rows, provenance):
    output_dir = Path(output_dir)
    write_json(output_dir / "annotations.json", coco)
    write_manifest_csv(output_dir / "manifest.csv", manifest_rows)
    write_json(output_dir / "provenance.json", provenance)
    return {
        "ok": True,
        "output": str(output_dir),
        "images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
        "manifest": str(output_dir / "manifest.csv"),
        "annotations_json": str(output_dir / "annotations.json"),
        "provenance": str(output_dir / "provenance.json"),
    }
