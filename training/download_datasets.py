import argparse
import json
import shutil
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from common_training import DATA_DIR, file_sha256, write_json
from dataset_registry import (
    REGISTRY_PATH,
    approved_datasets,
    download_gate_errors,
    get_dataset,
    load_registry,
    validate_registry,
)

DEFAULT_EXTERNAL_DIR = DATA_DIR / "external"


def _now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _filename_from_url(url, dataset_id, index):
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    if name:
        return name
    return f"{dataset_id}_{index:02d}.download"


def _expected_sha256(entry, artifact_count):
    checksum = str(entry.get("checksum", "")).strip()
    if not checksum or artifact_count != 1:
        return ""
    return checksum.split(":", 1)[1] if checksum.lower().startswith("sha256:") else checksum


def _download_once(url, target, timeout=60):
    partial = target.with_name(target.name + ".part")
    headers = {"User-Agent": "Corpus dataset downloader"}
    mode = "wb"
    if partial.exists() and partial.stat().st_size:
        headers["Range"] = f"bytes={partial.stat().st_size}-"
        mode = "ab"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if status != 206:
            mode = "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle)
    partial.replace(target)


def download_file(url, target, retries=3, timeout=60):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            _download_once(url, target, timeout=timeout)
            return target
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(5, attempt))
    raise RuntimeError(f"Could not download {url}: {last_error}") from last_error


def write_sidecars(destination, entry, artifacts, registry_path=REGISTRY_PATH):
    destination = Path(destination)
    source_payload = {
        "dataset": entry,
        "registry": str(Path(registry_path).resolve()),
        "downloaded_at": _now_iso(),
        "artifacts": artifacts,
    }
    write_json(destination / "SOURCE.json", source_payload)
    (destination / "CHECKSUMS.txt").write_text(
        "\n".join(f"sha256:{item['sha256']}  {item['filename']}" for item in artifacts) + "\n",
        encoding="utf-8",
    )
    (destination / "LICENSE.txt").write_text(
        "\n".join(
            [
                f"Dataset: {entry.get('name', entry.get('id', 'unknown'))}",
                f"License status: {entry.get('license_status', '')}",
                f"License: {entry.get('license', '')}",
                f"Redistribution: {entry.get('redistribution', '')}",
                "",
                "Verify this text against the upstream source before redistribution.",
            ]
        ),
        encoding="utf-8",
    )
    (destination / "README.source.md").write_text(
        "\n".join(
            [
                f"# {entry.get('name', entry.get('id', 'Dataset'))}",
                "",
                f"- Registry ID: `{entry.get('id', '')}`",
                f"- Source URL: {entry.get('source_url', '')}",
                f"- DOI: {entry.get('doi', '')}",
                f"- Corpus layer: `{entry.get('corpus_layer', '')}`",
                f"- Intended use: {entry.get('intended_use', '')}",
                f"- Downloaded at: {_now_iso()}",
                "",
                "Downloaded files are intentionally kept under ignored `data/external/`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def download_dataset(entry, external_dir=DEFAULT_EXTERNAL_DIR, force=False, dry_run=False, retries=3, timeout=60):
    gate_errors = download_gate_errors(entry)
    if gate_errors:
        raise PermissionError("Dataset failed download gate: " + "; ".join(gate_errors))

    dataset_id = entry["id"]
    destination = Path(external_dir) / dataset_id
    urls = list(entry.get("download_urls") or [])
    if dry_run:
        return {
            "ok": True,
            "dataset": dataset_id,
            "destination": str(destination),
            "would_download": urls,
            "dry_run": True,
        }

    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(f"Refusing to overwrite existing dataset directory without --force: {destination}")
    if destination.exists() and force:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    expected_sha = _expected_sha256(entry, len(urls))
    artifacts = []
    for index, url in enumerate(urls, start=1):
        filename = _filename_from_url(url, dataset_id, index)
        target = destination / filename
        download_file(url, target, retries=retries, timeout=timeout)
        actual_sha = file_sha256(target)
        if expected_sha and actual_sha.lower() != expected_sha.lower():
            raise ValueError(f"Checksum mismatch for {target}: expected {expected_sha}, got {actual_sha}")
        artifacts.append({"url": url, "filename": filename, "sha256": actual_sha, "bytes": target.stat().st_size})

    write_sidecars(destination, entry, artifacts)
    return {"ok": True, "dataset": dataset_id, "destination": str(destination), "artifacts": artifacts}


def selected_entries(args, registry):
    if args.all_approved:
        return approved_datasets(registry)
    if args.dataset:
        return [get_dataset(args.dataset, registry)]
    raise SystemExit("Provide --dataset <id> or --all-approved.")


def main():
    parser = argparse.ArgumentParser(description="Download approved Corpus training datasets from registry.yaml.")
    parser.add_argument("--dataset", default="", help="Registry dataset id to download.")
    parser.add_argument("--all-approved", action="store_true", help="Download every registry entry that passes gates.")
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    parser.add_argument("--out", default=str(DEFAULT_EXTERNAL_DIR))
    parser.add_argument("--force", action="store_true", help="Remove an existing destination before download.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    validation = validate_registry(registry)
    if not validation["ok"]:
        print(json.dumps(validation, indent=2, ensure_ascii=True))
        raise SystemExit("Dataset registry is invalid.")

    results = []
    for entry in selected_entries(args, registry):
        results.append(
            download_dataset(
                entry,
                external_dir=args.out,
                force=args.force,
                dry_run=args.dry_run,
                retries=args.retries,
                timeout=args.timeout,
            )
        )
    print(json.dumps({"ok": True, "results": results}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
