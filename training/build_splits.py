import argparse
import csv
import json
from pathlib import Path

from common_training import load_json, stable_hash, write_json

SPLITS = ("train", "val", "test")
GROUP_PRIORITY = (
    "micrograph_id",
    "original_micrograph_id",
    "parent_micrograph_id",
    "figure_id",
    "source_figure_id",
    "source_id",
    "source_group",
    "doi",
    "acquisition_session",
)


def leakage_group(row):
    for field in GROUP_PRIORITY:
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return str(row.get("image_id") or row.get("id") or row.get("file_name") or "").strip()


def split_for_group(group, ranked_groups, train_fraction=0.70, val_fraction=0.15):
    if len(ranked_groups) == 1:
        return "train"
    if len(ranked_groups) == 2:
        return "train" if group == ranked_groups[0] else "val"
    index = ranked_groups.index(group)
    train_cut = max(1, int(round(len(ranked_groups) * train_fraction)))
    val_cut = max(train_cut + 1, int(round(len(ranked_groups) * (train_fraction + val_fraction))))
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def assign_splits(rows, train_fraction=0.70, val_fraction=0.15, preserve_existing=True):
    groups = sorted({leakage_group(row) for row in rows if leakage_group(row)}, key=stable_hash)
    assignments = {group: split_for_group(group, groups, train_fraction, val_fraction) for group in groups}
    output = []
    for row in rows:
        item = dict(row)
        group = leakage_group(item)
        existing = item.get("split", "")
        if preserve_existing and existing in SPLITS:
            item["split"] = existing
        else:
            item["split"] = assignments.get(group, "train")
        item["split_group"] = group
        output.append(item)
    return output


def read_manifest(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_manifest(manifest_path, output_path, train_fraction=0.70, val_fraction=0.15, overwrite=False):
    rows = read_manifest(manifest_path)
    assigned = assign_splits(rows, train_fraction=train_fraction, val_fraction=val_fraction,
                             preserve_existing=not overwrite)
    write_manifest(output_path, assigned)
    counts = {split: sum(1 for row in assigned if row.get("split") == split) for split in SPLITS}
    return {"ok": True, "output": str(output_path), "rows": len(assigned), "counts": counts}


def _image_rows_from_coco(coco):
    rows = []
    for image in coco.get("images", []):
        meta = dict(image.get("metadata") or {})
        row = {**meta, **{key: value for key, value in image.items() if key != "metadata"}}
        row["image_id"] = image.get("id")
        rows.append(row)
    return rows


def split_coco(coco_path, output_path, train_fraction=0.70, val_fraction=0.15, overwrite=False):
    coco = load_json(coco_path)
    assigned = assign_splits(_image_rows_from_coco(coco), train_fraction=train_fraction, val_fraction=val_fraction,
                             preserve_existing=not overwrite)
    by_id = {row["image_id"]: row for row in assigned}
    for image in coco.get("images", []):
        row = by_id.get(image.get("id"), {})
        if row.get("split"):
            image["split"] = row["split"]
            meta = dict(image.get("metadata") or {})
            meta["split"] = row["split"]
            meta["split_group"] = row.get("split_group", "")
            image["metadata"] = meta
    write_json(output_path, coco)
    counts = {split: sum(1 for row in assigned if row.get("split") == split) for split in SPLITS}
    return {"ok": True, "output": str(output_path), "images": len(assigned), "counts": counts}


def main():
    parser = argparse.ArgumentParser(description="Assign leakage-safe train/val/test splits by source group.")
    parser.add_argument("--manifest", default="", help="Manifest CSV to split.")
    parser.add_argument("--coco", default="", help="COCO annotations JSON to split.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing split values.")
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.coco):
        raise SystemExit("Provide exactly one of --manifest or --coco.")
    if args.manifest:
        result = split_manifest(args.manifest, args.out, args.train_fraction, args.val_fraction, args.overwrite)
    else:
        result = split_coco(args.coco, args.out, args.train_fraction, args.val_fraction, args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
