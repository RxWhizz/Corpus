"""Run the validation harness against the demo phantoms' ground truth.

This is a *self-check*, not the scientific pilot. The reference here is the
exact geometry the phantom generator drew, so it measures how faithfully the
classical pipeline recovers a known object -- it says nothing about agreement
with a human expert on a real micrograph. That is Workstream F2, and it needs
real TEM data with manual measurements.

What it is good for: proving the F1 harness runs end to end from one command,
producing the CSV, JSON, Markdown and figures a real pilot will produce, and
catching regressions in measurement accuracy.

Usage::

    python scripts/build_synthetic_validation.py
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corpus.validation.__main__ import main as validation_main  # noqa: E402

DEMO_DIR = REPO_ROOT / "Examples" / "demo_dataset"
OUT_DIR = REPO_ROOT / "docs" / "validation" / "synthetic_selfcheck"

#: A detection is matched to a true particle when their centres are within this
#: fraction of the true radius. Generous enough for a soft-edged detection,
#: tight enough that two different particles cannot be confused.
MATCH_RADIUS_FRACTION = 0.75


def _value(number):
    return "" if not number else f"{number:.4f}"


def build_table(path):
    truth = json.loads((DEMO_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    rows = []

    for image_id, entry in sorted(truth.items()):
        expected_path = DEMO_DIR / "expected" / f"{image_id}.json"
        if not expected_path.exists():
            raise SystemExit(f"Missing expected output for {image_id}; run scripts/build_demo_expected.py first.")
        measured = json.loads(expected_path.read_text(encoding="utf-8"))
        nm_per_px = entry["nm_per_px"]
        detections = list(measured["objects"])
        used = set()

        for particle in entry["particles"]:
            true_x, true_y = particle["center"]
            true_outer = particle.get("outer_diameter_nm") or 0.0
            true_core = particle.get("core_diameter_nm") or 0.0
            true_shell = particle.get("shell_thickness_nm") or 0.0
            true_diameter = true_outer or true_core
            tolerance = max(8.0, (true_diameter / 2) * MATCH_RADIUS_FRACTION / max(nm_per_px, 1e-9) * nm_per_px)

            best = None
            best_distance = None
            for index, detection in enumerate(detections):
                if index in used:
                    continue
                distance_nm = math.hypot(
                    (detection["center_x"] - true_x) * nm_per_px,
                    (detection["center_y"] - true_y) * nm_per_px,
                )
                if distance_nm <= tolerance and (best_distance is None or distance_nm < best_distance):
                    best, best_distance = index, distance_nm

            detection = detections[best] if best is not None else {}
            if best is not None:
                used.add(best)

            corpus_outer = detection.get("outer_major_axis") or 0.0
            corpus_core = detection.get("inner_major_axis") or 0.0

            # The core/outer/shell triplet only exists for a genuine core-shell
            # particle. For solid discs, rods and bare carriers those columns
            # are left blank, which the harness reads as "not applicable to
            # this row" rather than as a missed detection.
            is_core_shell = particle["kind"] == "core_shell"
            rows.append({
                "image_id": image_id,
                "particle_id": particle["particle_id"],
                "reference_diameter_nm": _value(true_diameter),
                "corpus_diameter_nm": _value(corpus_outer or corpus_core),
                "reference_core_nm": _value(true_core) if is_core_shell else "",
                "corpus_core_nm": _value(corpus_core) if is_core_shell else "",
                "reference_outer_nm": _value(true_outer) if is_core_shell else "",
                "corpus_outer_nm": _value(corpus_outer) if is_core_shell else "",
                "reference_shell_nm": _value(true_shell) if is_core_shell else "",
                "corpus_shell_nm": (_value(detection.get("shell_thickness_estimate") or 0.0)
                                    if is_core_shell else ""),
                "morphology": entry["morphology"],
                "mode": " ".join(measured["config"]),
                "source": image_id,
                "status": "" if best is not None else "corpus_missed",
                "notes": "" if best is not None else "no detection within the match radius",
            })

        # Detections that matched no true particle are reported as extra, never
        # dropped -- they are what drives precision below 1.
        for index, detection in enumerate(detections):
            if index in used:
                continue
            corpus_outer = detection.get("outer_major_axis") or 0.0
            corpus_core = detection.get("inner_major_axis") or 0.0
            rows.append({
                "image_id": image_id,
                "particle_id": f"extra_{detection['object_id']}",
                "reference_diameter_nm": "",
                "corpus_diameter_nm": _value(corpus_outer or corpus_core),
                "reference_core_nm": "", "corpus_core_nm": "",
                "reference_outer_nm": "", "corpus_outer_nm": "",
                "reference_shell_nm": "", "corpus_shell_nm": "",
                "morphology": entry["morphology"],
                "mode": " ".join(measured["config"]),
                "source": image_id,
                "status": "unmatched_detection",
                "notes": "detected by Corpus with no corresponding true particle",
            })

    columns = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    out = Path(args.out)
    table = out / "table.csv"
    rows = build_table(table)
    code = validation_main([
        "--table", str(table),
        "--out", str(out),
        "--reference-name", "synthetic ground truth (phantom generator)",
        "--stratify-by", "morphology",
    ])
    print(json.dumps({"ok": code == 0, "rows": len(rows), "table": str(table.relative_to(REPO_ROOT))}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
