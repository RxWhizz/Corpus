from pathlib import Path

from adapters.generic_masks import normalize_generic_masks


def _read_ids(path):
    path = Path(path)
    if not path.exists():
        return []
    return [line.strip().replace(".png", "") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_emps(emps_dir, output_dir, clean=False):
    emps_dir = Path(emps_dir)
    split_by_stem = {}
    for image_id in _read_ids(emps_dir / "train.csv"):
        split_by_stem[image_id] = "train"
    for image_id in _read_ids(emps_dir / "test.csv"):
        split_by_stem[image_id] = "test"
    return normalize_generic_masks(
        emps_dir,
        output_dir,
        dataset_id="emps",
        image_subdir="images",
        mask_subdir="segmaps",
        class_name="particle",
        default_dataset_layer="real_near_emps",
        split_by_stem=split_by_stem,
        clean=clean,
    )
