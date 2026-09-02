import argparse
import csv
import json
from pathlib import Path

from common_training import ROOT

DEFAULT_SEED = ROOT / "training" / "seed_sources.json"
DEFAULT_CSV = ROOT / "reports" / "dataset_discovery.csv"
DEFAULT_MD = ROOT / "reports" / "dataset_discovery.md"

COLUMNS = [
    "dataset",
    "url",
    "doi",
    "material",
    "modality",
    "images",
    "annotation_type",
    "license",
    "license_status",
    "exact_AuSiO2",
    "training_value",
    "decision",
    "reason",
]


def _decision_for_license(status, fallback="needs_review"):
    status = str(status or "").lower()
    if status in {"verified", "accepted", "public", "cc_by", "cc0"}:
        return "approved"
    if "reject" in status or "blocked" in status:
        return "rejected"
    return fallback


def rows_from_seed(seed_path=DEFAULT_SEED):
    payload = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("local_pdf_sources", []):
        layer = row.get("candidate_layer", "")
        license_status = row.get("license_status", "needs_review")
        rows.append(
            {
                "dataset": row.get("title", row.get("file", "")),
                "url": row.get("file", ""),
                "doi": row.get("doi", ""),
                "material": "Au@SiO2 core-shell" if layer == "real_exact" else "related Au/silica nanoparticles",
                "modality": "TEM",
                "images": "unknown_until_triage",
                "annotation_type": "manual_masks_needed",
                "license": row.get("license", ""),
                "license_status": license_status,
                "exact_AuSiO2": "yes" if layer == "real_exact" else "no_or_review",
                "training_value": row.get("training_use", ""),
                "decision": _decision_for_license(license_status, fallback="hold_for_license_review"),
                "reason": row.get("notes", row.get("reason", "")),
            }
        )
    for row in payload.get("related_datasets", []):
        caution = bool(row.get("redistribution_caution", False))
        rows.append(
            {
                "dataset": row.get("name", ""),
                "url": row.get("url", ""),
                "doi": row.get("doi", ""),
                "material": "related nanoparticle EM",
                "modality": "EM/TEM/SEM",
                "images": row.get("images", ""),
                "annotation_type": row.get("annotation_type", "varies"),
                "license": row.get("license", ""),
                "license_status": "needs_review" if caution else row.get("license_status", "needs_review"),
                "exact_AuSiO2": "no",
                "training_value": row.get("use", ""),
                "decision": "hold_for_license_review" if caution else "inspect",
                "reason": row.get("notes", row.get("use", "")),
            }
        )
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in COLUMNS} for row in rows)


def write_md(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Corpus Dataset Discovery Report",
        "",
        "This report is generated from `training/seed_sources.json`. It is an",
        "inspection inventory, not a license approval.",
        "",
        f"- Plausible sources listed: {len(rows)}",
        f"- Exact Au@SiO2 candidates: {sum(1 for row in rows if row.get('exact_AuSiO2') == 'yes')}",
        f"- Held for license review: {sum(1 for row in rows if 'review' in row.get('decision', ''))}",
        "",
        "| Dataset | DOI | Exact Au@SiO2 | Decision | Reason |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        reason = str(row.get("reason", "")).replace("|", "/")
        dataset = str(row.get("dataset", "")).replace("|", "/")
        lines.append(
            f"| {dataset} | {row.get('doi', '')} | {row.get('exact_AuSiO2', '')} | "
            f"{row.get('decision', '')} | {reason} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report(seed_path=DEFAULT_SEED, csv_path=DEFAULT_CSV, md_path=DEFAULT_MD):
    rows = rows_from_seed(seed_path)
    write_csv(csv_path, rows)
    write_md(md_path, rows)
    return {"ok": True, "rows": len(rows), "csv": str(csv_path), "md": str(md_path)}


def main():
    parser = argparse.ArgumentParser(description="Generate dataset discovery CSV/Markdown from seed sources.")
    parser.add_argument("--seed", default=str(DEFAULT_SEED))
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--md", default=str(DEFAULT_MD))
    args = parser.parse_args()
    print(json.dumps(generate_report(args.seed, args.csv, args.md), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
