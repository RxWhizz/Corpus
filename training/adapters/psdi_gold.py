import shutil
import zipfile
from pathlib import Path

from common_training import file_sha256, load_json

from adapters.common import (
    manifest_row,
    polygon_area,
    polygon_bbox,
    read_image_size,
    reset_output_dir,
    write_normalized_dataset,
)
from adapters.corpus_native import normalize_coco_dataset

DATASET_ID = "psdi_gold_tem_2026"
SOURCE_URL = "https://data-collections.psdi.ac.uk/records/sgvf0-j3g53"
LICENSE = "CC BY 4.0"
LICENSE_STATUS = "cc_by"
CITATION = (
    "Stewart, Andrew; Da Silva De Sa, Natalia. Testing, Training and Validation "
    "Synthetic Dataset of Transmission Electron Microscopy (TEM) Images of Gold "
    "Nano-particles for Segmentation. PSDI record sgvf0-j3g53, 2026."
)
SPLIT_FILES = {
    "train": "instances_annotations_train.json",
    "test": "instances_annotations_test.json",
    "val": "instances_annotations_val.json",
}


def _ensure_images_extracted(input_dir):
    input_dir = Path(input_dir)
    if all((input_dir / split).exists() for split in SPLIT_FILES):
        return False
    zip_path = input_dir / "images.zip"
    if not zip_path.exists():
        return False
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(input_dir)
    return True


def _source_image(input_dir, split, file_name):
    file_name = Path(str(file_name))
    candidates = [
        Path(input_dir) / split / file_name.name,
        Path(input_dir) / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _source_metadata(split, image_id):
    synthetic = split in {"train", "test"}
    return {
        "dataset_layer": "synthetic_core_shell" if synthetic else "real_near",
        "source_dataset": DATASET_ID,
        "source_id": f"{DATASET_ID}:{split}:{image_id}",
        "source_group": f"{DATASET_ID}:{split}:{image_id}",
        "split": split,
        "modality": "TEM",
        "source_type": "synthetic_tempos" if synthetic else "experimental_manual_validation",
        "license": LICENSE,
        "license_status": LICENSE_STATUS,
        "source_url": SOURCE_URL,
        "citation": CITATION,
        "caption": (
            "PSDI synthetic TEM gold nanoparticle image."
            if synthetic
            else "PSDI experimental TEM gold nanoparticle validation image."
        ),
    }


def _category_remap(coco, categories):
    mapping = {}
    existing = {category["name"]: category["id"] for category in categories}
    for category in coco.get("categories", []):
        name = category.get("name", "")
        if name not in existing:
            existing[name] = len(existing)
            categories.append(
                {
                    "id": existing[name],
                    "name": name,
                    "supercategory": category.get("supercategory", "nanoparticle"),
                }
            )
        mapping[category["id"]] = existing[name]
    return mapping


def _source_checksums(input_dir):
    checksums = {}
    for name in ["images.zip", *SPLIT_FILES.values(), "val_binary_masks.zip", "croissant_metadata.json"]:
        path = Path(input_dir) / name
        if path.exists():
            checksums[name] = f"sha256:{file_sha256(path)}"
    return checksums


def _normalize_psdi_dir(input_dir, output_dir, clean=False):
    input_dir = Path(input_dir)
    _ensure_images_extracted(input_dir)
    output_dir = reset_output_dir(output_dir, clean=clean)
    normalized = {"images": [], "annotations": [], "categories": []}
    manifest = []
    warnings = []
    copied_image_ids = set()
    instance_counts = {}

    for split, file_name in SPLIT_FILES.items():
        coco_path = input_dir / file_name
        if not coco_path.exists():
            warnings.append(f"Missing PSDI annotation file: {file_name}")
            continue
        source = load_json(coco_path)
        category_map = _category_remap(source, normalized["categories"])
        for image in source.get("images", []):
            original_image_id = image.get("id")
            image_id = f"{split}:{original_image_id}"
            source_path = _source_image(input_dir, split, image.get("file_name", ""))
            if not source_path:
                warnings.append(f"Missing PSDI image: {split}/{image.get('file_name', '')}")
                continue
            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            if not width or not height:
                width, height = read_image_size(source_path)
            target_name = f"{split}_{Path(source_path).name}"
            target = output_dir / "images" / target_name
            shutil.copy2(source_path, target)
            metadata = _source_metadata(split, original_image_id)
            normalized["images"].append(
                {
                    "id": image_id,
                    "file_name": f"images/{target_name}",
                    "width": width,
                    "height": height,
                    "source_id": metadata["source_id"],
                    "split": split,
                    "metadata": metadata,
                }
            )
            copied_image_ids.add((split, original_image_id))
            instance_counts[image_id] = 0
            manifest.append(
                manifest_row(
                    image_id=image_id,
                    file_name=f"images/{target_name}",
                    original_path=source_path,
                    width=width,
                    height=height,
                    instances=0,
                    meta=metadata,
                    default_dataset_layer=metadata["dataset_layer"],
                )
            )

        for annotation in source.get("annotations", []):
            original_image_id = annotation.get("image_id")
            if (split, original_image_id) not in copied_image_ids:
                continue
            category_id = annotation.get("category_id")
            if category_id not in category_map:
                warnings.append(f"Skipped PSDI annotation {annotation.get('id')} with category {category_id}")
                continue
            new_annotation = dict(annotation)
            new_annotation["id"] = f"{split}:{annotation.get('id')}"
            new_annotation["image_id"] = f"{split}:{original_image_id}"
            new_annotation["category_id"] = category_map[category_id]
            if not new_annotation.get("area"):
                first = (new_annotation.get("segmentation") or [[]])[0]
                new_annotation["area"] = polygon_area(first)
            if not new_annotation.get("bbox"):
                first = (new_annotation.get("segmentation") or [[]])[0]
                new_annotation["bbox"] = polygon_bbox(first)
            normalized["annotations"].append(new_annotation)
            instance_counts[new_annotation["image_id"]] += 1

    for row in manifest:
        row["instances"] = instance_counts.get(row["image_id"], 0)

    provenance = {
        "dataset_id": DATASET_ID,
        "adapter": "psdi_gold",
        "source": str(input_dir),
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "license_status": LICENSE_STATUS,
        "citation": CITATION,
        "source_files": SPLIT_FILES,
        "source_checksums": _source_checksums(input_dir),
        "images_extracted": all((input_dir / split).exists() for split in SPLIT_FILES),
        "warnings": warnings,
    }
    result = write_normalized_dataset(output_dir, normalized, manifest, provenance)
    result["warnings"] = warnings
    result["splits"] = {
        split: sum(1 for row in manifest if row.get("split") == split)
        for split in SPLIT_FILES
    }
    return result


def normalize_psdi_gold(coco_path, output_dir, clean=False):
    """Normalize PSDI-style gold-particle COCO without inventing core/shell labels."""
    if Path(coco_path).is_dir():
        return _normalize_psdi_dir(coco_path, output_dir, clean=clean)
    return normalize_coco_dataset(
        coco_path,
        output_dir,
        dataset_id=DATASET_ID,
        default_dataset_layer="real_near",
        preserve_categories=True,
        clean=clean,
    )
