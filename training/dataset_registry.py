import argparse
import json
from pathlib import Path

from common_training import ROOT

REGISTRY_PATH = ROOT / "training" / "datasets" / "registry.yaml"

REQUIRED_DATASET_FIELDS = (
    "id",
    "name",
    "source_url",
    "publisher",
    "domain",
    "material_scope",
    "image_count",
    "annotation_type",
    "ontology",
    "license",
    "license_status",
    "redistribution",
    "citation",
    "checksum",
    "corpus_layer",
    "intended_use",
    "download_status",
    "enabled",
)

VERIFIED_LICENSE_STATUS = "verified"


def _strip_comment(line):
    quote = None
    output = []
    for char in line:
        if char in ("'", '"'):
            quote = None if quote == char else char
        if char == "#" and quote is None:
            break
        output.append(char)
    return "".join(output).rstrip()


def _parse_inline_list(value):
    body = value[1:-1].strip()
    if not body:
        return []
    return [_parse_scalar(part.strip()) for part in body.split(",")]


def _parse_scalar(value):
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        return _parse_inline_list(value)
    if value[0:1] == value[-1:] and value.startswith(("'", '"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return ""
    try:
        return int(value)
    except ValueError:
        return value


def _parse_key_value(text):
    if ":" not in text:
        raise ValueError(f"Expected 'key: value' YAML line, got: {text!r}")
    key, value = text.split(":", 1)
    return key.strip(), _parse_scalar(value)


def load_registry(path=REGISTRY_PATH):
    """Load Corpus' small registry YAML subset without an optional dependency.

    The registry intentionally uses only top-level scalars plus
    ``datasets:`` as a list of flat mappings. Keeping the format simple makes
    the legal gate usable in bare Python environments and in Colab notebooks.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset registry not found: {path}")

    registry = {}
    current_section = None
    current_entry = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0:
            key, value = _parse_key_value(stripped)
            if key == "datasets":
                registry[key] = []
                current_section = key
                current_entry = None
            else:
                registry[key] = value
                current_section = None
                current_entry = None
            continue

        if current_section != "datasets":
            raise ValueError(f"{path}:{line_number}: nested YAML is only supported under datasets.")

        if stripped.startswith("- "):
            current_entry = {}
            registry.setdefault("datasets", []).append(current_entry)
            rest = stripped[2:].strip()
            if rest:
                key, value = _parse_key_value(rest)
                current_entry[key] = value
            continue

        if current_entry is None:
            raise ValueError(f"{path}:{line_number}: dataset field appeared before a '- id:' row.")
        key, value = _parse_key_value(stripped)
        current_entry[key] = value

    registry.setdefault("datasets", [])
    return registry


def dataset_index(registry=None):
    registry = registry or load_registry()
    index = {}
    for entry in registry.get("datasets", []):
        dataset_id = str(entry.get("id", "")).strip()
        if dataset_id:
            index[dataset_id] = entry
    return index


def get_dataset(dataset_id, registry=None):
    try:
        return dataset_index(registry)[dataset_id]
    except KeyError:
        raise KeyError(f"Unknown dataset {dataset_id!r}. Registered: {sorted(dataset_index(registry))}") from None


def validate_dataset_entry(entry):
    errors = []
    warnings = []
    dataset_id = entry.get("id", "<missing id>")
    missing = [field for field in REQUIRED_DATASET_FIELDS if field not in entry]
    if missing:
        errors.append(f"{dataset_id}: missing required registry fields: {missing}")

    for field in ("id", "name", "source_url", "license_status", "corpus_layer", "intended_use"):
        if field in entry and not str(entry.get(field, "")).strip():
            errors.append(f"{dataset_id}: {field} must not be blank.")

    if entry.get("license_status") == VERIFIED_LICENSE_STATUS:
        if not entry.get("checksum"):
            errors.append(f"{dataset_id}: verified datasets must record a checksum before download/training.")
        if not entry.get("citation"):
            errors.append(f"{dataset_id}: verified datasets must record citation text.")
        if not entry.get("license"):
            errors.append(f"{dataset_id}: verified datasets must record license terms.")
    else:
        for field in ("license", "citation", "checksum"):
            if field in entry and not entry.get(field):
                warnings.append(f"{dataset_id}: {field} is not confirmed yet.")

    if entry.get("download_urls") is None:
        warnings.append(f"{dataset_id}: download_urls is not declared; downloader will refuse it.")
    return errors, warnings


def validate_registry(registry=None):
    registry = registry or load_registry()
    errors = []
    warnings = []
    seen = set()
    for entry in registry.get("datasets", []):
        dataset_id = str(entry.get("id", "")).strip()
        if dataset_id in seen:
            errors.append(f"Duplicate dataset id: {dataset_id}")
        seen.add(dataset_id)
        entry_errors, entry_warnings = validate_dataset_entry(entry)
        errors.extend(entry_errors)
        warnings.extend(entry_warnings)
    return {"ok": not errors, "datasets": len(registry.get("datasets", [])), "errors": errors, "warnings": warnings}


def download_gate_errors(entry):
    """Reasons a registry entry cannot be downloaded automatically."""
    errors = []
    dataset_id = entry.get("id", "<missing id>")
    if entry.get("license_status") != VERIFIED_LICENSE_STATUS:
        errors.append(
            f"{dataset_id}: license_status={entry.get('license_status')!r}; expected "
            f"{VERIFIED_LICENSE_STATUS!r}."
        )
    if entry.get("enabled") is not True:
        errors.append(f"{dataset_id}: enabled must be true for automatic download.")
    if not entry.get("download_urls"):
        errors.append(f"{dataset_id}: no download_urls configured.")
    return errors


def is_downloadable(entry):
    return not download_gate_errors(entry)


def approved_datasets(registry=None):
    registry = registry or load_registry()
    return [entry for entry in registry.get("datasets", []) if is_downloadable(entry)]


def model_dataset_manifest(dataset_ids, registry=None):
    registry = registry or load_registry()
    entries = []
    for dataset_id in dataset_ids:
        entry = get_dataset(dataset_id, registry)
        entries.append(
            {
                "id": entry.get("id", ""),
                "name": entry.get("name", ""),
                "source_url": entry.get("source_url", ""),
                "doi": entry.get("doi", ""),
                "publisher": entry.get("publisher", ""),
                "license": entry.get("license", ""),
                "license_status": entry.get("license_status", ""),
                "redistribution": entry.get("redistribution", ""),
                "citation": entry.get("citation", ""),
                "checksum": entry.get("checksum", ""),
                "corpus_layer": entry.get("corpus_layer", ""),
                "intended_use": entry.get("intended_use", ""),
            }
        )
    return {"registry": str(Path(REGISTRY_PATH).resolve()), "datasets": entries}


def main():
    parser = argparse.ArgumentParser(description="Validate and inspect the Corpus dataset registry.")
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    parser.add_argument("--dataset", default="")
    parser.add_argument("--model-manifest", nargs="*", default=None)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    validation = validate_registry(registry)
    payload = validation
    if args.dataset:
        payload = {"dataset": get_dataset(args.dataset, registry), "validation": validation}
    if args.model_manifest is not None:
        payload = model_dataset_manifest(args.model_manifest, registry)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    raise SystemExit(0 if validation["ok"] else 1)


if __name__ == "__main__":
    main()
