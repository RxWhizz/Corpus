import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

from build_splits import SPLITS, leakage_group
from common_training import load_json, write_json
from PIL import Image


def _read_manifest(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_from_coco(path):
    coco = load_json(path)
    rows = []
    for image in coco.get("images", []):
        meta = dict(image.get("metadata") or {})
        row = {**meta, **{key: value for key, value in image.items() if key != "metadata"}}
        row["image_id"] = image.get("id")
        row.setdefault("split", image.get("split") or meta.get("split", ""))
        rows.append(row)
    return rows


def _image_hash(path, size=8):
    path = Path(path)
    if not path.exists():
        return ""
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size))
        get_pixels = getattr(image, "get_flattened_data", image.getdata)
        pixels = list(get_pixels())
    mean = sum(pixels) / len(pixels)
    bits = ["1" if pixel >= mean else "0" for pixel in pixels]
    return "".join(bits)


def _hamming(left, right):
    if not left or not right or len(left) != len(right):
        return None
    return sum(a != b for a, b in zip(left, right))


def audit_rows(rows, require_test=False, exact_test_only=False, near_duplicate_distance=3):
    errors = []
    warnings = []
    groups = {}
    checksums = {}
    image_hashes = {}

    for row in rows:
        label = row.get("image_id") or row.get("file_name") or row.get("image_path") or "<unknown image>"
        split = row.get("split", "")
        if split not in SPLITS:
            errors.append(f"{label}: split must be one of {SPLITS}, got {split!r}.")
        group = leakage_group(row)
        groups.setdefault(group, set()).add(split)
        checksum = row.get("file_sha256", "")
        if checksum:
            checksums.setdefault(checksum, []).append(row)
        if near_duplicate_distance >= 0:
            image_path = row.get("image_path") or row.get("file_name") or ""
            if image_path and Path(image_path).exists():
                digest = _image_hash(image_path)
                if digest:
                    image_hashes[label] = (digest, row)
        if exact_test_only and split == "test" and row.get("content_layer") != "real_exact":
            errors.append(
                f"{label}: exact-domain test set may only contain content_layer=real_exact, "
                f"got {row.get('content_layer')!r}."
            )

    for group, splits in groups.items():
        real_splits = {split for split in splits if split}
        if len(real_splits) > 1:
            errors.append(f"Leakage: group {group!r} appears in multiple splits: {sorted(real_splits)}")

    for checksum, items in checksums.items():
        if len(items) < 2:
            continue
        splits = sorted({item.get("split", "") for item in items})
        message = f"Duplicate image checksum {checksum} appears in {len(items)} manifest rows."
        if len(set(splits)) > 1:
            errors.append(message + f" Splits: {splits}")
        else:
            warnings.append(message)

    if near_duplicate_distance >= 0:
        for (left_label, (left_hash, left_row)), (right_label, (right_hash, right_row)) in combinations(
            image_hashes.items(), 2
        ):
            distance = _hamming(left_hash, right_hash)
            if distance is None or distance > near_duplicate_distance:
                continue
            message = f"Near-duplicate image hashes: {left_label} and {right_label} (distance {distance})."
            if left_row.get("split") != right_row.get("split"):
                errors.append(message)
            else:
                warnings.append(message)

    split_counts = {split: sum(1 for row in rows if row.get("split") == split) for split in SPLITS}
    if require_test and split_counts["test"] == 0:
        errors.append("No locked test split.")
    return {
        "ok": not errors,
        "rows": len(rows),
        "split_counts": split_counts,
        "groups": len(groups),
        "errors": errors,
        "warnings": warnings,
    }


def audit_split_leakage(manifest="", coco="", require_test=False, exact_test_only=False, near_duplicate_distance=3):
    if bool(manifest) == bool(coco):
        raise ValueError("Provide exactly one of manifest or coco.")
    rows = _read_manifest(manifest) if manifest else _rows_from_coco(coco)
    return audit_rows(
        rows,
        require_test=require_test,
        exact_test_only=exact_test_only,
        near_duplicate_distance=near_duplicate_distance,
    )


def write_report(path, result):
    path = Path(path)
    lines = [
        "# Corpus Split Leakage Audit",
        "",
        f"- OK: {result['ok']}",
        f"- Rows: {result['rows']}",
        f"- Groups: {result['groups']}",
        f"- Split counts: {result['split_counts']}",
        "",
        "## Errors",
    ]
    lines.extend(f"- {error}" for error in result["errors"]) if result["errors"] else lines.append("- None")
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in result["warnings"]) if result["warnings"] else lines.append("- None")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Audit train/val/test splits for group leakage and duplicates.")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--coco", default="")
    parser.add_argument("--require-test", action="store_true")
    parser.add_argument("--exact-test-only", action="store_true")
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=3,
        help="Maximum average-hash Hamming distance to flag; use -1 to disable perceptual checks.",
    )
    parser.add_argument("--json-out", default="")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    result = audit_split_leakage(
        manifest=args.manifest,
        coco=args.coco,
        require_test=args.require_test,
        exact_test_only=args.exact_test_only,
        near_duplicate_distance=args.near_duplicate_distance,
    )
    if args.json_out:
        write_json(args.json_out, result)
    if args.report:
        write_report(args.report, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
