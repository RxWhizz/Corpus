import argparse
import csv
import re
from pathlib import Path

from common_training import CLASS_NAMES, DEFAULT_YOLO_DIR, TRAINING_AUDIT_MD

#: Dataset milestones from training/README.md, expressed as checks rather than
#: prose. `min_images` gates the milestone; `min_sources` and `min_real_exact`
#: capture the "3-5 sources" and "real Au@SiO2 truth" requirements that image
#: count alone cannot express.
DATASET_TARGETS = {
    "smoke": {
        "min_images": 5, "max_images": 10, "min_sources": 1, "min_real_exact": 0,
        "description": "Smoke dataset: end-to-end pipeline check.",
    },
    "pilot": {
        "min_images": 150, "max_images": 250, "min_sources": 2, "min_real_exact": 1,
        "description": "Pilot: first dataset large enough to train on.",
    },
    "v0": {
        "min_images": 300, "max_images": 600, "min_sources": 3, "min_real_exact": 1,
        "description": "Useful v0: a model worth evaluating.",
    },
    "publication": {
        "min_images": 600, "max_images": 1200, "min_sources": 3, "min_real_exact": 1,
        "description": "Publication target: 3-5 sources, multiple magnifications "
                       "and shell-thickness ranges.",
    },
}

#: Order from smallest to largest, for reporting which milestone is reached.
TARGET_ORDER = ("smoke", "pilot", "v0", "publication")


def evaluate_target(name, total_images, sources, real_exact_images):
    """Check a dataset against one documented milestone."""
    target = DATASET_TARGETS[name]
    checks = {
        "images": {
            "value": total_images,
            "minimum": target["min_images"],
            "pass": total_images >= target["min_images"],
        },
        "sources": {
            "value": sources,
            "minimum": target["min_sources"],
            "pass": sources >= target["min_sources"],
        },
        "real_exact_images": {
            "value": real_exact_images,
            "minimum": target["min_real_exact"],
            "pass": real_exact_images >= target["min_real_exact"],
        },
    }
    return {
        "target": name,
        "description": target["description"],
        "range": [target["min_images"], target["max_images"]],
        "checks": checks,
        "pass": all(check["pass"] for check in checks.values()),
    }


def reached_target(total_images, sources, real_exact_images):
    """The largest milestone the dataset currently satisfies, or None."""
    reached = None
    for name in TARGET_ORDER:
        if evaluate_target(name, total_images, sources, real_exact_images)["pass"]:
            reached = name
    return reached


def write_report(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Corpus Training Dataset Audit",
        "",
        f"- OK: {result['ok']}",
        f"- Images train/val/test: {result['counts']}",
        f"- Label rows: {result['label_rows']}",
        f"- Class counts: {result['class_counts']}",
        f"- Dataset layers: {result['dataset_layers']}",
        f"- Content layers: {result.get('content_layers', {})}",
        f"- Source groups: {result.get('source_groups', 0)}",
        f"- Manifest rows: {result.get('manifest_rows', 0)}",
        f"- Provenance gaps: {result.get('provenance', {})}",
        f"- Milestone reached: {result.get('reached_target') or 'none'}",
        "",
        "## Dataset target",
    ]
    target = result.get("target")
    if not target:
        lines.append("- No target requested (pass --target smoke|pilot|v0|publication).")
    else:
        lines.append(f"- Target `{target['target']}`: {'PASS' if target['pass'] else 'FAIL'} "
                     f"— {target['description']}")
        lines.append(f"- Image range for this milestone: {target['range'][0]}–{target['range'][1]}")
        for name, check in target["checks"].items():
            lines.append(f"  - {name}: {check['value']} (min {check['minimum']}) "
                         f"{'OK' if check['pass'] else 'SHORT'}")
    lines.extend([
        "",
        "## Errors",
    ])
    lines.extend(f"- {item}" for item in result["errors"]) if result["errors"] else lines.append("- None")
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {item}" for item in result["warnings"]) if result["warnings"] else lines.append("- None")
    path.write_text("\n".join(lines), encoding="utf-8")


def class_names_from_data_yaml(data_yaml):
    if not data_yaml.exists():
        return list(CLASS_NAMES)
    names = {}
    in_names = False
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if re.match(r"^\s*names\s*:", line):
            in_names = True
            continue
        if in_names:
            if not line.strip():
                continue
            if not raw_line.startswith((" ", "\t")):
                break
            match = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", raw_line)
            if match:
                names[int(match.group(1))] = match.group(2).strip().strip("'\"")
    if not names:
        return list(CLASS_NAMES)
    return [names[index] for index in sorted(names)]


def audit_dataset(dataset_dir, min_images=0, min_au_core=0, min_sio2_outer=0, require_test=False,
                  target=None):
    dataset_dir = Path(dataset_dir)
    errors = []
    warnings = []
    counts = {"train": 0, "val": 0, "test": 0}
    dataset_layers = {}
    content_layers = {}
    source_groups = set()
    label_rows = 0

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        errors.append("Missing data.yaml.")
    class_names = class_names_from_data_yaml(data_yaml)
    class_counts = dict.fromkeys(class_names, 0)

    groups_by_split = {}
    manifest_rows = 0
    provenance = {"missing_license": 0, "missing_source": 0, "missing_checksum": 0,
                  "missing_calibration": 0, "partial_annotation_review": 0}
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        errors.append("Missing manifest.csv. A training bundle without a manifest is not traceable.")
    else:
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                manifest_rows += 1
                label = row.get("image_id") or row.get("image_path") or "<unknown image>"
                group = row.get("source_id") or row.get("source_group") or row.get("doi") or row.get("image_id") or row.get("file_name", "")
                groups_by_split.setdefault(group, set()).add(row.get("split", ""))
                layer = row.get("dataset_layer", "") or "unspecified"
                dataset_layers[layer] = dataset_layers.get(layer, 0) + 1
                content = row.get("content_layer", "") or "unspecified"
                content_layers[content] = content_layers.get(content, 0) + 1
                if group:
                    source_groups.add(group)

                if not row.get("nm_per_px") and layer not in {"real_near_emps"}:
                    provenance["missing_calibration"] += 1
                    warnings.append(f"Missing nm_per_px in manifest for {label}.")

                # Licence and source are surfaced for every row, not only the
                # public demo layer: an untraceable image cannot be published
                # later even if it trains fine now.
                if not row.get("license") and not row.get("license_status"):
                    provenance["missing_license"] += 1
                    warnings.append(f"Missing license and license_status for {label}.")
                if not row.get("source_url") and not row.get("doi"):
                    provenance["missing_source"] += 1
                    warnings.append(f"Missing source_url and doi for {label}.")
                if "file_sha256" in (row or {}) and not row.get("file_sha256"):
                    provenance["missing_checksum"] += 1
                    warnings.append(f"Missing file_sha256 for {label}.")
                if row.get("annotation_review") == "partial":
                    provenance["partial_annotation_review"] += 1

                if layer == "public_demo" and row.get("license_status") not in {"accepted", "public", "cc_by", "cc0"}:
                    errors.append(f"Public demo row lacks accepted license status: {label}.")

    for group, splits in groups_by_split.items():
        if len(splits) > 1:
            errors.append(f"Source group appears in multiple splits: {group} -> {sorted(splits)}")

    for split in counts:
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        images = sorted(path for path in image_dir.glob("*") if path.is_file()) if image_dir.exists() else []
        counts[split] = len(images)
        if images and not label_dir.exists():
            errors.append(f"Missing label directory for split {split}.")
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"Missing label for {image}.")
                continue
            lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                errors.append(f"Empty label file for {image}.")
                continue
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) < 7 or len(parts) % 2 == 0:
                    errors.append(f"Invalid polygon line {label}:{line_number}.")
                    continue
                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(f"Non-numeric label values {label}:{line_number}.")
                    continue
                if class_id < 0 or class_id >= len(class_names):
                    errors.append(f"Invalid class id {class_id} in {label}:{line_number}.")
                else:
                    class_counts[class_names[class_id]] += 1
                if any(value < 0 or value > 1 for value in coords):
                    errors.append(f"Coordinates outside [0,1] in {label}:{line_number}.")
                label_rows += 1

    if counts["train"] == 0:
        errors.append("No training images.")
    if counts["val"] == 0:
        warnings.append("No validation images. Add more annotated sources before real training.")
    if counts["test"] == 0 and require_test:
        errors.append("No test images.")
    elif counts["test"] == 0:
        warnings.append("No test images. This is acceptable for smoke tests, but not for dataset v0.")
    if label_rows == 0:
        errors.append("No label polygons found.")
    if sum(counts.values()) < min_images:
        errors.append(f"Dataset has {sum(counts.values())} images; minimum required is {min_images}.")
    if "Au_core" in class_counts and class_counts["Au_core"] < min_au_core:
        errors.append(f"Au_core labels {class_counts['Au_core']} below minimum {min_au_core}.")
    if "SiO2_outer" in class_counts and class_counts["SiO2_outer"] < min_sio2_outer:
        errors.append(f"SiO2_outer labels {class_counts['SiO2_outer']} below minimum {min_sio2_outer}.")

    if manifest_rows and sum(counts.values()) != manifest_rows:
        warnings.append(
            f"Manifest lists {manifest_rows} images but {sum(counts.values())} are present on disk."
        )

    total_images = sum(counts.values())
    real_exact_images = content_layers.get("real_exact", 0)
    target_result = None
    if target:
        target_result = evaluate_target(target, total_images, len(source_groups), real_exact_images)
        if not target_result["pass"]:
            for name, check in target_result["checks"].items():
                if not check["pass"]:
                    errors.append(
                        f"Target '{target}' not met: {name} is {check['value']}, "
                        f"needs at least {check['minimum']}."
                    )

    # Au@SiO2 core/shell metrology may only be reported from `real_exact` data.
    # A dataset with none is still trainable, but it is transfer material.
    if manifest_rows and real_exact_images == 0:
        warnings.append(
            "No `real_exact` images: this dataset carries no real Au@SiO2 core-shell truth "
            "and must not be used to report core/shell metrology."
        )

    return {
        "ok": not errors,
        "counts": counts,
        "label_rows": label_rows,
        "class_counts": class_counts,
        "dataset_layers": dataset_layers,
        "content_layers": content_layers,
        "source_groups": len(source_groups),
        "manifest_rows": manifest_rows,
        "provenance": provenance,
        "target": target_result,
        "reached_target": reached_target(total_images, len(source_groups), real_exact_images),
        "errors": errors,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit a prepared YOLO segmentation dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_YOLO_DIR))
    parser.add_argument("--min-images", type=int, default=0)
    parser.add_argument("--min-au-core", type=int, default=0)
    parser.add_argument("--min-sio2-outer", type=int, default=0)
    parser.add_argument("--require-test", action="store_true")
    parser.add_argument(
        "--target",
        choices=list(TARGET_ORDER),
        default=None,
        help="Fail the audit unless the dataset meets this documented milestone.",
    )
    parser.add_argument("--report", default=str(TRAINING_AUDIT_MD))
    args = parser.parse_args()
    result = audit_dataset(
        args.dataset,
        min_images=args.min_images,
        min_au_core=args.min_au_core,
        min_sio2_outer=args.min_sio2_outer,
        require_test=args.require_test,
        target=args.target,
    )
    write_report(args.report, result)
    print(result)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
