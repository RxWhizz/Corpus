import argparse
import csv
import json
import shutil
from pathlib import Path

from common_training import file_sha256, write_manifest

SPLITS = ("train", "val", "test")


def _read_class_names(data_yaml):
    path = Path(data_yaml)
    names = {}
    in_names = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            if not line.strip():
                continue
            if not raw_line.startswith((" ", "\t")):
                break
            key, _, value = raw_line.partition(":")
            try:
                names[int(key.strip())] = value.strip().strip("'\"")
            except ValueError:
                continue
    return [names[index] for index in sorted(names)]


def _write_data_yaml(output_dir, class_names):
    names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    (Path(output_dir) / "data.yaml").write_text(
        f"path: {Path(output_dir).resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names_yaml}\n",
        encoding="utf-8",
    )


def _read_manifest(dataset_dir):
    path = Path(dataset_dir) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest.csv in {dataset_dir}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _clean_output(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _dataset_prefix(dataset_dir):
    return Path(dataset_dir).name.replace("_yolo_seg", "")


def combine_yolo_datasets(dataset_dirs, output_dir, clean=False):
    dataset_dirs = [Path(item) for item in dataset_dirs]
    output_dir = Path(output_dir)
    if clean:
        _clean_output(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    class_names = None
    manifest_rows = []
    warnings = []
    copied = 0
    labels = 0

    for dataset_dir in dataset_dirs:
        names = _read_class_names(dataset_dir / "data.yaml")
        if class_names is None:
            class_names = names
        elif names != class_names:
            raise ValueError(f"Class names mismatch in {dataset_dir}: {names} != {class_names}")

        prefix = _dataset_prefix(dataset_dir)
        for row in _read_manifest(dataset_dir):
            split = row.get("split", "")
            if split not in SPLITS:
                warnings.append(f"Skipped {row.get('image_id', '<unknown>')}: invalid split {split!r}.")
                continue
            image_path = Path(row.get("image_path", ""))
            label_path = Path(row.get("label_path", ""))
            if not image_path.exists() or not label_path.exists():
                warnings.append(f"Skipped {row.get('image_id', '<unknown>')}: missing image or label file.")
                continue
            stem = f"{prefix}_{image_path.stem}"
            target_image = output_dir / "images" / split / f"{stem}{image_path.suffix.lower()}"
            target_label = output_dir / "labels" / split / f"{stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_image)
            shutil.copy2(label_path, target_label)
            copied += 1
            labels += int(row.get("labels") or 0)

            item = dict(row)
            item["image_id"] = f"{prefix}:{row.get('image_id')}"
            item["source_id"] = row.get("source_id") or item["image_id"]
            item["image_path"] = str(target_image)
            item["label_path"] = str(target_label)
            item["file_sha256"] = file_sha256(target_image)
            manifest_rows.append(item)

    if class_names is None:
        raise ValueError("No datasets provided.")
    _write_data_yaml(output_dir, class_names)
    write_manifest(output_dir / "manifest.csv", manifest_rows)
    (output_dir / "prepare_warnings.txt").write_text("\n".join(warnings), encoding="utf-8")
    return {
        "ok": not warnings,
        "output": str(output_dir),
        "images": copied,
        "labels": labels,
        "splits": {split: sum(1 for row in manifest_rows if row.get("split") == split) for split in SPLITS},
        "class_names": class_names,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description="Combine prepared YOLO-seg datasets with the same ontology.")
    parser.add_argument("--dataset", action="append", required=True, help="Prepared YOLO dataset directory.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    result = combine_yolo_datasets(args.dataset, args.out, clean=args.clean)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
