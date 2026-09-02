import argparse
import json
from pathlib import Path

from adapters.bam_tio2 import normalize_bam_tio2
from adapters.corpus_native import normalize_coco_dataset, normalize_corpus_native
from adapters.emps import normalize_emps
from adapters.generic_masks import normalize_generic_masks
from adapters.psdi_gold import normalize_psdi_gold
from common_training import DATA_DIR
from dataset_registry import REGISTRY_PATH, get_dataset, load_registry, validate_registry

DEFAULT_NORMALIZED_DIR = DATA_DIR / "normalized"


def normalize_from_registry(dataset_id, input_path=None, output_dir=None, adapter=None, clean=False, registry_path=REGISTRY_PATH):
    registry = load_registry(registry_path)
    validation = validate_registry(registry)
    if not validation["ok"]:
        raise SystemExit(f"Dataset registry is invalid: {validation['errors']}")
    entry = get_dataset(dataset_id, registry)
    adapter = adapter or entry.get("adapter", "")
    input_path = Path(input_path or DATA_DIR / "external" / dataset_id)
    output_dir = Path(output_dir or DEFAULT_NORMALIZED_DIR / dataset_id)

    if adapter == "emps":
        return normalize_emps(input_path, output_dir, clean=clean)
    if adapter == "bam_tio2":
        return normalize_bam_tio2(input_path, output_dir, clean=clean)
    if adapter == "generic_masks":
        return normalize_generic_masks(input_path, output_dir, dataset_id=dataset_id, clean=clean)
    if adapter == "corpus_native":
        coco_path = input_path
        if input_path.is_dir():
            coco_path = input_path / "annotations.json"
        return normalize_corpus_native(coco_path, output_dir, clean=clean)
    if adapter == "psdi_gold":
        return normalize_psdi_gold(input_path, output_dir, clean=clean)
    if adapter == "coco_preserve_categories":
        return normalize_coco_dataset(
            input_path,
            output_dir,
            dataset_id=dataset_id,
            default_dataset_layer=entry.get("corpus_layer", "real_near"),
            preserve_categories=True,
            clean=clean,
        )
    raise SystemExit(f"Unsupported adapter {adapter!r} for dataset {dataset_id!r}.")


def main():
    parser = argparse.ArgumentParser(description="Normalize an approved/curated source dataset into Corpus COCO.")
    parser.add_argument("--dataset", required=True, help="Dataset id from training/datasets/registry.yaml.")
    parser.add_argument("--input", default="", help="Downloaded folder or COCO file. Defaults to data/external/<dataset>.")
    parser.add_argument("--out", default="", help="Output folder. Defaults to data/normalized/<dataset>.")
    parser.add_argument("--adapter", default="", help="Override registry adapter.")
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    result = normalize_from_registry(
        args.dataset,
        input_path=args.input or None,
        output_dir=args.out or None,
        adapter=args.adapter or None,
        clean=args.clean,
        registry_path=args.registry,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
