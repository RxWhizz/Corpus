import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from audit_split_leakage import audit_split_leakage
from audit_split_leakage import write_report as write_split_report
from audit_training_dataset import audit_dataset
from audit_training_dataset import write_report as write_dataset_report
from common_training import ROOT, file_sha256, write_json
from dataset_registry import REGISTRY_PATH, get_dataset, load_registry, model_dataset_manifest, validate_registry

DEFAULT_CONFIG = ROOT / "configs" / "training" / "au_sio2_v1.yaml"


def _parse_inline_list(value):
    value = value.strip()
    if not value.startswith("[") or not value.endswith("]"):
        return None
    body = value[1:-1].strip()
    if not body:
        return []
    return [_parse_scalar(part.strip()) for part in body.split(",")]


def _parse_scalar(value):
    value = value.strip()
    inline = _parse_inline_list(value)
    if inline is not None:
        return inline
    if value[0:1] == value[-1:] and value.startswith(("'", '"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", ""}:
        return ""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    config = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, value = line.split(":", 1)
        config[key.strip()] = _parse_scalar(value)
    return config


def resolve_repo_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def verify_registered_datasets(dataset_ids, registry, allow_restricted=False):
    errors = []
    warnings = []
    for dataset_id in dataset_ids:
        entry = get_dataset(dataset_id, registry)
        source_url = str(entry.get("source_url", ""))
        is_external = not source_url.startswith("local://")
        license_status = entry.get("license_status")
        if is_external and license_status == "verified_restricted" and allow_restricted:
            warnings.append(
                f"{dataset_id}: registry license_status is 'verified_restricted'; "
                "allowed only for private/local training by config."
            )
        elif is_external and license_status != "verified":
            errors.append(
                f"{dataset_id}: external training data must have license_status='verified' "
                f"before it can be used."
            )
        elif license_status != "verified":
            warnings.append(
                f"{dataset_id}: registry license_status is {license_status!r}; "
                "the per-image manifest audit must carry the legal proof."
            )
    return errors, warnings


def prepare_run_dir(config, force=False):
    run_id = str(config.get("run_id") or config.get("name") or "corpus_seg_run")
    project = resolve_repo_path(config.get("project", "runs/training"))
    run_dir = project / run_id
    if run_dir.exists() and not force:
        raise FileExistsError(f"Run directory already exists; use --force to replace it: {run_dir}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_model_card(run_dir, config, registry_manifest, dataset_audit, split_audit, status, weights_path=""):
    lines = [
        f"# {config.get('run_id', 'Corpus segmentation model')}",
        "",
        "## Status",
        "",
        f"- Status: {status}",
        f"- Created: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        f"- Weights: {weights_path or 'not produced'}",
        "",
        "## Intended Use",
        "",
        "AI-assisted Au@SiO2 TEM instance-segmentation proposals for Corpus human review.",
        "The backend must not enter final metrology without accepted review state.",
        "",
        "## Ontology",
        "",
        "- `Au_core`: visible Au core mask.",
        "- `SiO2_outer`: full visible outer particle contour.",
        "",
        "## Dataset Provenance",
        "",
    ]
    for entry in registry_manifest.get("datasets", []):
        lines.append(
            f"- `{entry.get('id')}`: {entry.get('name')} "
            f"({entry.get('license_status')}, {entry.get('corpus_layer')})"
        )
    lines.extend(
        [
            "",
            "## Audit Summary",
            "",
            f"- Dataset audit OK: {dataset_audit.get('ok')}",
            f"- Split audit OK: {split_audit.get('ok')}",
            f"- Images: {dataset_audit.get('counts')}",
            f"- Class counts: {dataset_audit.get('class_counts')}",
            "",
            "## Limitations",
            "",
            "- No universal TEM segmentation claim is made by this artifact.",
            "- Low-contrast and agglomerated particles require subgroup validation.",
            "- Human review remains the final authority.",
        ]
    )
    (run_dir / "model_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_artifacts(run_dir, config_path, config, registry_manifest, dataset_audit, split_audit, status):
    shutil.copy2(config_path, run_dir / "training_config.yaml")
    write_json(run_dir / "dataset_manifest.json", registry_manifest)
    write_json(
        run_dir / "metrics.json",
        {
            "status": status,
            "dataset_audit": dataset_audit,
            "split_audit": split_audit,
        },
    )
    write_json(
        run_dir / "command.json",
        {
            "argv": sys.argv,
            "python": sys.executable,
            "cwd": str(Path.cwd()),
        },
    )
    checksums = []
    for path in ("training_config.yaml", "dataset_manifest.json", "metrics.json", "command.json"):
        target = run_dir / path
        checksums.append(f"sha256:{file_sha256(target)}  {path}")
    (run_dir / "checksums.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    write_dataset_report(run_dir / "training_dataset_audit.md", dataset_audit)
    write_split_report(run_dir / "split_leakage_audit.md", split_audit)
    write_model_card(run_dir, config, registry_manifest, dataset_audit, split_audit, status)


def run_training(dataset_dir, config, run_dir):
    command = [
        sys.executable,
        str(ROOT / "training" / "colab_run_training.py"),
        "--dataset",
        str(dataset_dir),
        "--full",
        "--epochs",
        str(config.get("epochs", 75)),
        "--imgsz",
        str(config.get("imgsz", 1024)),
        "--batch",
        str(config.get("batch", 4)),
        "--patience",
        str(config.get("patience", 20)),
        "--project",
        str(run_dir.parent),
        "--name",
        str(config.get("name", run_dir.name)),
        "--full-model",
        str(config.get("model_name", "yolo11s-seg.pt")),
        "--fallback-model",
        str(config.get("fallback_model", "yolo11n-seg.pt")),
    ]
    if config.get("device") not in (None, ""):
        command.extend(["--device", str(config.get("device"))])
    subprocess.check_call(command)


def run_experiment(config_path=DEFAULT_CONFIG, dry_run=False, force=False):
    config_path = Path(config_path)
    config = load_config(config_path)
    registry_path = resolve_repo_path(config.get("registry", REGISTRY_PATH))
    registry = load_registry(registry_path)
    registry_validation = validate_registry(registry)
    if not registry_validation["ok"]:
        raise SystemExit(f"Dataset registry is invalid: {registry_validation['errors']}")

    dataset_ids = [str(item) for item in config.get("dataset_ids", [])]
    registry_errors, registry_warnings = verify_registered_datasets(
        dataset_ids,
        registry,
        allow_restricted=bool(config.get("allow_restricted", False)),
    )
    if registry_errors:
        raise SystemExit("Dataset registry gate failed: " + "; ".join(registry_errors))

    dataset_dir = resolve_repo_path(config.get("dataset_dir", "data/training/yolo_seg"))
    manifest_path = dataset_dir / "manifest.csv"
    if not (dataset_dir / "data.yaml").exists() or not manifest_path.exists():
        raise SystemExit(
            "Prepared YOLO dataset is missing. Build it first with "
            "`python training/prepare_yolo_seg.py` or `python training/package_colab_bundle.py --synthetic-smoke`."
        )

    dataset_audit = audit_dataset(
        dataset_dir,
        min_images=int(config.get("min_images", 0)),
        min_au_core=int(config.get("min_au_core", 0)),
        min_sio2_outer=int(config.get("min_sio2_outer", 0)),
        require_test=bool(config.get("require_test", False)),
        target=config.get("target") or None,
    )
    split_audit = audit_split_leakage(
        manifest=str(manifest_path),
        require_test=bool(config.get("require_test", False)),
        exact_test_only=bool(config.get("exact_test_only", False)),
        near_duplicate_distance=int(config.get("near_duplicate_distance", 3)),
    )
    if registry_warnings:
        dataset_audit.setdefault("warnings", []).extend(registry_warnings)
    if not dataset_audit["ok"] or not split_audit["ok"]:
        status = "blocked_by_audit"
    elif dry_run:
        status = "dry_run_ready"
    else:
        status = "training_started"

    run_dir = prepare_run_dir(config, force=force)
    registry_manifest = model_dataset_manifest(dataset_ids, registry)
    write_run_artifacts(run_dir, config_path, config, registry_manifest, dataset_audit, split_audit, status)

    if status == "blocked_by_audit":
        raise SystemExit(f"Training gates failed. See {run_dir}")
    if not dry_run:
        run_training(dataset_dir, config, run_dir)
    return {
        "ok": True,
        "status": status,
        "run_dir": str(run_dir),
        "dataset_audit": dataset_audit,
        "split_audit": split_audit,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible Corpus segmentation experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true", help="Run gates and write artifacts without training.")
    parser.add_argument("--force", action="store_true", help="Replace an existing run directory.")
    args = parser.parse_args()
    result = run_experiment(args.config, dry_run=args.dry_run, force=args.force)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
