import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from build_splits import assign_splits
from common_training import (
    DATA_DIR,
    annotation_review_status,
    file_sha256,
    layers_for,
    load_json,
    normalize_polygon,
    read_csv,
    resolve_image_path,
    safe_stem,
    write_manifest,
)

DEFAULT_COCO = DATA_DIR / "normalized" / "agglomerated_non_spherical_em" / "annotations.json"
DEFAULT_OUT = DATA_DIR / "training" / "agglomerated_non_spherical_em_yolo_seg"
SPLITS = ("train", "val", "test")


def _clean_dir(path):
    path = Path(path).resolve()
    allowed = (DATA_DIR / "training").resolve()
    if path.exists():
        if not str(path).lower().startswith(str(allowed).lower()):
            raise ValueError(f"Refusing to remove outside data/training: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _class_names(categories, requested=()):
    by_name = {str(category.get("name", "")).strip(): category for category in categories}
    if requested:
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise SystemExit(f"Requested classes are missing from COCO categories: {missing}")
        return list(requested)
    if "particle" in by_name:
        return ["particle"]
    names = [str(category.get("name", "")).strip() for category in categories if category.get("name")]
    if len(names) == 1:
        return names
    raise SystemExit(
        "Could not infer a single particle class. Pass --class-name with one "
        "category present in the COCO file."
    )


def _category_map(categories, class_names):
    index_by_name = {name: index for index, name in enumerate(class_names)}
    mapping = {}
    for category in categories:
        name = str(category.get("name", "")).strip()
        if name in index_by_name:
            mapping[category["id"]] = index_by_name[name]
    return mapping


def _annotation_groups(coco, category_map):
    grouped = defaultdict(list)
    skipped_by_image = defaultdict(int)
    warnings = []
    for annotation in coco.get("annotations", []):
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if category_id not in category_map:
            continue
        segmentation = annotation.get("segmentation") or []
        if not isinstance(segmentation, list):
            warnings.append(f"Annotation {annotation.get('id')} uses RLE segmentation; skipped.")
            continue
        if annotation_review_status(annotation) != "ready":
            skipped_by_image[image_id] += 1
            continue
        grouped[image_id].append((category_map[category_id], segmentation))
    return grouped, skipped_by_image, warnings


def _manifest_index(path):
    rows = read_csv(path)
    by_image_id = {}
    by_file_name = {}
    for row in rows:
        if row.get("image_id"):
            by_image_id[str(row["image_id"])] = row
        if row.get("file_name"):
            by_file_name[Path(row["file_name"]).name] = row
    return by_image_id, by_file_name


def _image_rows(coco, manifest_by_id, manifest_by_name):
    rows = []
    for image in coco.get("images", []):
        metadata = dict(image.get("metadata") or {})
        row = {
            **metadata,
            **manifest_by_name.get(Path(str(image.get("file_name", ""))).name, {}),
            **manifest_by_id.get(str(image.get("id", "")), {}),
            **{key: value for key, value in image.items() if key != "metadata"},
        }
        row["image_id"] = image.get("id")
        row["split"] = row.get("split") or image.get("split") or metadata.get("split", "")
        rows.append(row)
    return assign_splits(rows, preserve_existing=True)


def _write_data_yaml(out_dir, class_names):
    names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    path = out_dir / "data.yaml"
    path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names_yaml}\n",
        encoding="utf-8",
    )
    return path


def _read_optional_manifest(path, coco_path):
    if path:
        return Path(path)
    candidate = Path(coco_path).resolve().parent / "manifest.csv"
    return candidate if candidate.exists() else None


def prepare_particle_yolo(
    coco_path=DEFAULT_COCO,
    output_dir=DEFAULT_OUT,
    manifest_path="",
    class_name="particle",
    clean=False,
):
    coco_path = Path(coco_path)
    output_dir = Path(output_dir)
    if clean:
        _clean_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    coco = load_json(coco_path)
    class_names = _class_names(coco.get("categories", []), requested=[class_name] if class_name else [])
    category_map = _category_map(coco.get("categories", []), class_names)
    if not category_map:
        raise SystemExit(f"No COCO category matched export classes: {class_names}")

    manifest_path = _read_optional_manifest(manifest_path, coco_path)
    manifest_by_id, manifest_by_name = _manifest_index(manifest_path) if manifest_path else ({}, {})
    assigned_rows = _image_rows(coco, manifest_by_id, manifest_by_name)
    assigned_by_id = {row["image_id"]: row for row in assigned_rows}
    annotations, skipped_by_image, warnings = _annotation_groups(coco, category_map)

    class_counts = dict.fromkeys(class_names, 0)
    manifest_rows = []
    exported_images = 0
    exported_labels = 0
    skipped_without_labels = 0

    for image in coco.get("images", []):
        image_id = image.get("id")
        source_path = resolve_image_path(image.get("file_name", ""), coco_path)
        if not source_path:
            warnings.append(f"Missing image file for {image_id}: {image.get('file_name', '')}")
            continue
        row = assigned_by_id.get(image_id, {})
        split = row.get("split") if row.get("split") in SPLITS else "train"
        width = int(image.get("width") or row.get("width") or 0)
        height = int(image.get("height") or row.get("height") or 0)

        label_lines = []
        for class_id, polygons in annotations.get(image_id, []):
            for polygon in polygons:
                normalized = normalize_polygon(polygon, width, height)
                if not normalized:
                    warnings.append(f"Invalid polygon in image {image_id}; skipped.")
                    continue
                label_lines.append(" ".join([str(class_id)] + [f"{value:.6f}" for value in normalized]))
                class_counts[class_names[class_id]] += 1

        if not label_lines:
            skipped_without_labels += 1
            continue

        stem = f"{safe_stem(image_id)}_{safe_stem(Path(source_path).stem)}"
        image_target = output_dir / "images" / split / f"{stem}{source_path.suffix.lower()}"
        label_target = output_dir / "labels" / split / f"{stem}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, image_target)
        label_target.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        layer_row = dict(row)
        layer_row.update({key: value for key, value in image.items() if key != "metadata"})
        layer_info = layers_for(layer_row)
        declared = str(layer_row.get("dataset_layer", "") or "").strip()
        if declared:
            layer_info["dataset_layer"] = declared
        calibration_state = "confirmed" if row.get("nm_per_px") else "not_required_transfer"
        if layer_info["content_layer"] == "real_exact" and not row.get("nm_per_px"):
            calibration_state = "missing"

        exported_images += 1
        exported_labels += len(label_lines)
        manifest_rows.append(
            {
                "image_id": image_id,
                "source_id": row.get("source_id") or image.get("source_id") or image_id,
                "split": split,
                "dataset_layer": layer_info["dataset_layer"],
                "content_layer": layer_info["content_layer"],
                "distribution_layer": layer_info["distribution_layer"],
                "image_path": str(image_target),
                "label_path": str(label_target),
                "file_sha256": file_sha256(image_target),
                "labels": len(label_lines),
                "au_core_labels": 0,
                "sio2_outer_labels": 0,
                "annotation_review": "partial" if skipped_by_image.get(image_id) else "ready",
                "skipped_review_labels": skipped_by_image.get(image_id, 0),
                "nm_per_px": row.get("nm_per_px", ""),
                "calibration_state": calibration_state,
                "license": row.get("license", ""),
                "license_status": row.get("license_status", ""),
                "doi": row.get("doi", ""),
                "source_url": row.get("source_url", ""),
                "figure_label": row.get("figure_label", ""),
                "panel_label": row.get("panel_label", ""),
                "caption": row.get("caption", ""),
            }
        )

    data_yaml = _write_data_yaml(output_dir, class_names)
    write_manifest(output_dir / "manifest.csv", manifest_rows)
    (output_dir / "prepare_warnings.txt").write_text("\n".join(warnings), encoding="utf-8")

    return {
        "ok": not warnings,
        "output": str(output_dir),
        "data_yaml": str(data_yaml),
        "manifest": str(output_dir / "manifest.csv"),
        "images": exported_images,
        "labels": exported_labels,
        "class_counts": class_counts,
        "splits": {
            split: sum(1 for row in manifest_rows if row.get("split") == split)
            for split in SPLITS
        },
        "skipped_without_labels": skipped_without_labels,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare a one-class particle COCO dataset as YOLO-seg.")
    parser.add_argument("--coco", default=str(DEFAULT_COCO))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--manifest", default="")
    parser.add_argument("--class-name", default="particle")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    result = prepare_particle_yolo(
        coco_path=args.coco,
        output_dir=args.out,
        manifest_path=args.manifest,
        class_name=args.class_name,
        clean=args.clean,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
