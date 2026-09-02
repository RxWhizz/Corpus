import shutil
from pathlib import Path

from common_training import canonical_categories, category_mapping, load_json, resolve_image_path

from adapters.common import (
    manifest_row,
    polygon_area,
    polygon_bbox,
    read_image_size,
    reset_output_dir,
    write_normalized_dataset,
)


def _remap_categories(coco, preserve_categories):
    categories = coco.get("categories", [])
    if preserve_categories:
        return categories, {category["id"]: category["id"] for category in categories}, []
    mapping, unknown = category_mapping(categories)
    if unknown:
        raise SystemExit(f"Unsupported Corpus COCO categories: {unknown}")
    return canonical_categories(), mapping, unknown


def _first_category_name(categories, category_id):
    for category in categories:
        if category.get("id") == category_id:
            return category.get("name", "")
    return ""


def normalize_coco_dataset(
    coco_path,
    output_dir,
    dataset_id="corpus_native",
    default_dataset_layer="real_exact",
    preserve_categories=False,
    clean=False,
):
    coco_path = Path(coco_path)
    source = load_json(coco_path)
    categories, category_map, _unknown = _remap_categories(source, preserve_categories)
    output_dir = reset_output_dir(output_dir, clean=clean)

    copied_ids = set()
    image_id_to_meta = {}
    normalized = {"images": [], "annotations": [], "categories": categories}
    manifest = []
    warnings = []

    for image in source.get("images", []):
        source_path = resolve_image_path(image.get("file_name", ""), coco_path)
        if not source_path:
            warnings.append(f"Missing image file: {image.get('file_name', '')}")
            continue
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if not width or not height:
            width, height = read_image_size(source_path)
        target_name = Path(source_path).name
        target = output_dir / "images" / target_name
        shutil.copy2(source_path, target)
        meta = dict(image.get("metadata") or {})
        for key in (
            "source_id",
            "micrograph_id",
            "figure_id",
            "split",
            "dataset_layer",
            "license",
            "license_status",
            "doi",
            "source_url",
            "figure_label",
            "panel_label",
            "caption",
            "nm_per_px",
        ):
            if image.get(key) not in (None, ""):
                meta[key] = image.get(key)
        image_id = image.get("id")
        image_id_to_meta[image_id] = meta
        copied_ids.add(image_id)
        normalized["images"].append(
            {
                "id": image_id,
                "file_name": f"images/{target_name}",
                "width": width,
                "height": height,
                "source_id": meta.get("source_id", dataset_id),
                "metadata": meta,
            }
        )

    instance_counts = {image.get("id"): 0 for image in normalized["images"]}
    for annotation in source.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id not in copied_ids:
            continue
        category_id = annotation.get("category_id")
        if category_id not in category_map:
            warnings.append(f"Skipped annotation {annotation.get('id')} with unknown category {category_id}")
            continue
        new_annotation = dict(annotation)
        new_annotation["category_id"] = category_map[category_id]
        if not new_annotation.get("area"):
            first = (new_annotation.get("segmentation") or [[]])[0]
            new_annotation["area"] = polygon_area(first)
        if not new_annotation.get("bbox"):
            first = (new_annotation.get("segmentation") or [[]])[0]
            new_annotation["bbox"] = polygon_bbox(first)
        normalized["annotations"].append(new_annotation)
        instance_counts[image_id] += 1

    for image in normalized["images"]:
        meta = image_id_to_meta.get(image["id"], {})
        original_file = next(
            (
                source_image.get("file_name", "")
                for source_image in source.get("images", [])
                if source_image.get("id") == image["id"]
            ),
            "",
        )
        original_path = resolve_image_path(original_file, coco_path) or output_dir / image["file_name"]
        manifest.append(
            manifest_row(
                image_id=str(image["id"]),
                file_name=image["file_name"],
                original_path=original_path,
                width=image["width"],
                height=image["height"],
                instances=instance_counts.get(image["id"], 0),
                meta=meta,
                default_dataset_layer=default_dataset_layer,
            )
        )

    provenance = {
        "dataset_id": dataset_id,
        "adapter": "corpus_native" if not preserve_categories else "coco_preserve_categories",
        "source_coco": str(coco_path),
        "ontology": [_first_category_name(categories, category["id"]) for category in categories],
        "warnings": warnings,
    }
    result = write_normalized_dataset(output_dir, normalized, manifest, provenance)
    result["warnings"] = warnings
    return result


def normalize_corpus_native(coco_path, output_dir, clean=False):
    return normalize_coco_dataset(
        coco_path,
        output_dir,
        dataset_id="corpus_exact_au_sio2",
        default_dataset_layer="real_exact",
        preserve_categories=False,
        clean=clean,
    )
