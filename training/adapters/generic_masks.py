import shutil
from pathlib import Path

from adapters.common import (
    image_paths,
    instance_mask_polygons,
    manifest_row,
    polygon_area,
    polygon_bbox,
    read_image_size,
    read_metadata_csv,
    reset_output_dir,
    write_normalized_dataset,
)


def _find_mask(mask_dir, stem):
    for suffix in (".png", ".tif", ".tiff", ".bmp"):
        candidate = Path(mask_dir) / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def normalize_generic_masks(
    input_dir,
    output_dir,
    dataset_id="generic_masks",
    image_subdir="images",
    mask_subdir="masks",
    class_name="particle",
    default_dataset_layer="real_near",
    metadata_csv="",
    split_by_stem=None,
    clean=False,
):
    """Normalize image + instance-mask folders into COCO.

    The output ontology is exactly ``class_name``. Generic particle masks are
    never upgraded into Corpus' two-class Au_core/SiO2_outer ontology.
    """
    input_dir = Path(input_dir)
    mask_dir = input_dir / mask_subdir
    if not mask_dir.exists() and mask_subdir == "masks":
        mask_dir = input_dir / "segmaps"
    output_dir = reset_output_dir(output_dir, clean=clean)
    metadata = read_metadata_csv(metadata_csv or input_dir / "metadata.csv")
    split_by_stem = split_by_stem or {}

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": class_name, "supercategory": "nanoparticle"}],
    }
    manifest = []
    warnings = []
    annotation_id = 1

    for index, image_path in enumerate(image_paths(input_dir / image_subdir), start=1):
        mask_path = _find_mask(mask_dir, image_path.stem)
        if not mask_path:
            warnings.append(f"Missing mask for {image_path.name}")
            continue
        width, height = read_image_size(image_path)
        target = output_dir / "images" / image_path.name
        shutil.copy2(image_path, target)
        image_id = image_path.stem
        meta = dict(metadata.get(image_path.stem, {}))
        if image_path.stem in split_by_stem:
            meta["split"] = split_by_stem[image_path.stem]
        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"images/{target.name}",
                "width": width,
                "height": height,
                "source_id": meta.get("source_id") or meta.get("source_group") or dataset_id,
                "metadata": meta,
            }
        )

        polygons = instance_mask_polygons(mask_path)
        for polygon in polygons:
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "segmentation": [polygon],
                    "area": polygon_area(polygon),
                    "bbox": polygon_bbox(polygon),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        manifest.append(
            manifest_row(
                image_id=image_id,
                file_name=f"images/{target.name}",
                original_path=image_path,
                width=width,
                height=height,
                instances=len(polygons),
                meta=meta,
                default_dataset_layer=default_dataset_layer,
            )
        )

    provenance = {
        "dataset_id": dataset_id,
        "adapter": "generic_masks",
        "source": str(input_dir),
        "image_subdir": image_subdir,
        "mask_subdir": str(mask_dir.relative_to(input_dir)) if mask_dir.is_relative_to(input_dir) else str(mask_dir),
        "class_name": class_name,
        "warnings": warnings,
    }
    result = write_normalized_dataset(output_dir, coco, manifest, provenance)
    result["warnings"] = warnings
    return result
