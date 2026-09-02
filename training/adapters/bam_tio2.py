import hashlib
import shutil
import zipfile
from pathlib import Path

from build_splits import assign_splits
from common_training import ROOT, file_sha256

from adapters.common import (
    image_paths,
    instance_mask_polygons,
    manifest_row,
    polygon_area,
    polygon_bbox,
    read_image_size,
    reset_output_dir,
    write_normalized_dataset,
)

DATASET_ID = "agglomerated_non_spherical_em"
DATASET_NAME = "Agglomerated / Non-Spherical EM Particle Dataset"
SOURCE_URL = "https://zenodo.org/records/4563942"
DATASET_DOI = "10.5281/zenodo.4563942"
ARTICLE_DOI = "10.1038/s41598-021-84287-6"
LICENSE = "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International"
LICENSE_STATUS = "verified_restricted"
CITATION = (
    "B. Ruehle, J. F. Krumrey, V.-D. Hodoroaba, Workflow towards Automated "
    "Segmentation of Agglomerated, Non-Spherical Particles from Electron "
    f"Microscopy Images using Artificial Neural Networks, DOI {ARTICLE_DOI}."
)


def _dataset_root(input_dir):
    input_dir = Path(input_dir)
    if (input_dir / "Electron Microscopy Images").exists():
        return input_dir
    if (input_dir / "Datasets" / "Electron Microscopy Images").exists():
        return input_dir / "Datasets"
    raise FileNotFoundError(
        "Expected BAM TiO2 dataset root with 'Electron Microscopy Images', "
        f"or an extracted folder containing Datasets/: {input_dir}"
    )


def _mask_for(image_path, mask_dir):
    candidate = Path(mask_dir) / f"{image_path.stem}_m.tif"
    return candidate if candidate.exists() else None


def _archive_checksums(input_dir):
    input_dir = Path(input_dir)
    checksums = {}
    for label, candidate in (
        ("local_wrapper_sha256", ROOT / "4563942.zip"),
        ("datasets_zip_sha256", input_dir / "Datasets.zip"),
        ("parent_datasets_zip_sha256", input_dir.parent / "Datasets.zip"),
        ("local_archive_sha256", input_dir / "4563942.zip"),
    ):
        if candidate.exists() and candidate.is_file():
            checksums[label] = f"sha256:{file_sha256(candidate)}"
    wrapper = ROOT / "4563942.zip"
    if wrapper.exists():
        try:
            with zipfile.ZipFile(wrapper) as archive:
                with archive.open("Datasets.zip") as handle:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
                    checksums["embedded_datasets_zip_sha256"] = f"sha256:{digest.hexdigest()}"
        except (KeyError, zipfile.BadZipFile):
            pass
    return checksums


def _source_metadata(stem):
    return {
        "dataset_layer": "real_near",
        "source_dataset": DATASET_NAME,
        "source_id": f"{DATASET_ID}:{stem}",
        "source_group": f"{DATASET_ID}:{stem}",
        "micrograph_id": stem,
        "modality": "SEM",
        "license": LICENSE,
        "license_status": LICENSE_STATUS,
        "doi": DATASET_DOI,
        "source_url": SOURCE_URL,
        "citation": CITATION,
        "article_doi": ARTICLE_DOI,
        "caption": "BAM TiO2 SEM micrograph with manual particle mask.",
    }


def normalize_bam_tio2(input_dir, output_dir, clean=False):
    """Normalize the aligned BAM TiO2 SEM branch into one-class COCO.

    The archive also includes TSEM and class-coded masks. Several of those mask
    branches require the registration/crop transforms shipped in the source
    data, so this adapter intentionally exports only the SEM images whose
    manual binary masks match image dimensions directly.
    """
    input_dir = Path(input_dir)
    root = _dataset_root(input_dir)
    images_dir = root / "Electron Microscopy Images" / "SEM"
    masks_dir = root / "Electron Microscopy Image Masks" / "TiO2_Masks_Manual_4connected"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing SEM image directory: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing manual mask directory: {masks_dir}")

    output_dir = reset_output_dir(output_dir, clean=clean)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "particle", "supercategory": "nanoparticle"}],
    }
    manifest = []
    warnings = []
    annotation_id = 1

    for image_path in image_paths(images_dir):
        mask_path = _mask_for(image_path, masks_dir)
        if not mask_path:
            warnings.append(f"Missing manual mask for SEM image {image_path.name}.")
            continue
        width, height = read_image_size(image_path)
        mask_width, mask_height = read_image_size(mask_path)
        if (width, height) != (mask_width, mask_height):
            warnings.append(
                f"Skipped {image_path.name}: image size {(width, height)} "
                f"!= mask size {(mask_width, mask_height)}."
            )
            continue

        target = output_dir / "images" / image_path.name
        shutil.copy2(image_path, target)
        image_id = image_path.stem
        metadata = _source_metadata(image_path.stem)
        polygons = instance_mask_polygons(mask_path)

        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"images/{target.name}",
                "width": width,
                "height": height,
                "source_id": metadata["source_id"],
                "metadata": dict(metadata),
            }
        )
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
                meta=metadata,
                default_dataset_layer="real_near",
            )
        )

    assigned = assign_splits(manifest, preserve_existing=False)
    split_by_image_id = {row["image_id"]: row["split"] for row in assigned}
    for image in coco["images"]:
        split = split_by_image_id.get(image["id"], "")
        if split:
            image["split"] = split
            image["metadata"]["split"] = split
            image["metadata"]["split_group"] = image["metadata"].get("source_group", image["id"])

    archive_checksums = _archive_checksums(input_dir)
    registered_checksum = archive_checksums.get("embedded_datasets_zip_sha256") or archive_checksums.get(
        "datasets_zip_sha256",
        "",
    )
    provenance = {
        "dataset_id": DATASET_ID,
        "adapter": "bam_tio2",
        "source": str(input_dir),
        "source_url": SOURCE_URL,
        "doi": DATASET_DOI,
        "article_doi": ARTICLE_DOI,
        "license": LICENSE,
        "license_status": LICENSE_STATUS,
        "citation": CITATION,
        "archive_checksum": registered_checksum,
        "archive_checksums": archive_checksums,
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "exported_branch": "SEM + TiO2_Masks_Manual_4connected",
        "excluded_branches": [
            "TSEM masks and registration folders",
            "2-class manual masks with different frame height",
            "4-class manual masks with different frame height",
        ],
        "warnings": warnings,
    }
    result = write_normalized_dataset(output_dir, coco, assigned, provenance)
    result["warnings"] = warnings
    result["splits"] = {
        split: sum(1 for row in assigned if row.get("split") == split)
        for split in ("train", "val", "test")
    }
    return result
