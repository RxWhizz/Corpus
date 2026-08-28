"""Build and verify the demo dataset's expected Corpus output (Workstream E1).

The demo dataset is only a regression fixture if the output it produces is
versioned. This script runs ``measurement_modes.py`` over every demo image with
a fixed per-morphology configuration and stores a trimmed, path-free record of
the result under ``Examples/demo_dataset/expected/``.

``--check`` re-runs and compares numerically, so a change in the measurement
pipeline shows up as a diff with the worst deviation named.

Usage::

    python scripts/build_demo_expected.py           # write expected/
    python scripts/build_demo_expected.py --check    # fail on drift
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEMO_DIR = REPO_ROOT / "Examples" / "demo_dataset"
EXPECTED_DIR = DEMO_DIR / "expected"
SCRIPT = REPO_ROOT / "measurement_modes.py"

#: Fixed configuration per image, chosen to match the phantom's morphology.
#: Scale is taken from metadata.csv; the bar is marked manually so the baseline
#: does not depend on the auto-detector's heuristics.
RUN_CONFIG = {
    # Core-shell spheres: the flagship preset, Au core inside a SiO2 shell.
    "demo_001_core_shell_spheres": [
        "--shape-preset", "spheres", "--mode", "both",
        "--au-min-radius", "5", "--au-max-radius", "40",
        "--sio2-min-radius", "30", "--sio2-max-radius", "90",
    ],
    # Low contrast: the default dark-particle threshold finds nothing here, so
    # the adaptive carrier mask is used instead. The core is not recoverable at
    # this contrast -- the run reports carriers only, every object flagged for
    # review, plus one false positive. That is the honest result and it is what
    # the baseline records.
    "demo_002_core_shell_low_contrast": [
        "--shape-preset", "generic", "--mode", "sio2",
        "--sio2-min-radius", "30", "--sio2-max-radius", "90",
    ],
    # Touching discs: watershed separation is the point of this image.
    "demo_003_touching_spheres": [
        "--shape-preset", "generic", "--mode", "au", "--watershed", "true",
        "--au-min-radius", "10", "--au-max-radius", "40",
    ],
    # Solid rods: no core to pair with, so the generic carrier path is correct
    # and watershed stays off to avoid over-splitting elongated particles.
    "demo_004_rods": [
        "--shape-preset", "generic", "--mode", "sio2", "--watershed", "false",
        "--sio2-min-radius", "90", "--sio2-max-radius", "200",
    ],
    # Decorated carriers: small Au decorations on large SiO2 spheres.
    "demo_005_decorated": [
        "--shape-preset", "decorated", "--mode", "both",
        "--au-min-radius", "4", "--au-max-radius", "20",
        "--sio2-min-radius", "40", "--sio2-max-radius", "110",
    ],
    # Wide size spread, to catch size-dependent regressions.
    "demo_006_mixed_sizes": [
        "--shape-preset", "spheres", "--mode", "both",
        "--au-min-radius", "8", "--au-max-radius", "60",
        "--sio2-min-radius", "30", "--sio2-max-radius", "140",
    ],
}

#: Numeric fields kept per object. Everything else is either presentation or a
#: path, and would make the baseline churn for no scientific reason.
OBJECT_FIELDS = (
    "object_id", "preset", "center_x", "center_y",
    "inner_major_axis", "inner_minor_axis", "outer_major_axis", "outer_minor_axis",
    "equivalent_diameter", "shell_thickness_estimate", "inner_outer_ratio",
    "pair_status", "review_status", "confidence_score", "separation_method",
    "backend", "flags",
)

DECIMALS = 6
DEFAULT_TOLERANCE = 1e-6


def _round(value):
    if isinstance(value, float):
        return round(value, DECIMALS)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def load_metadata():
    with (DEMO_DIR / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_one(row):
    """Run the measurement CLI for one demo image and trim the payload."""
    image_path = DEMO_DIR / row["file_path"]
    extra = RUN_CONFIG.get(row["image_id"], ["--shape-preset", "generic", "--mode", "both"])
    command = [
        sys.executable, str(SCRIPT),
        "--image", str(image_path),
        "--scale", row["scale_nm"],
        "--manual-scale-px", row["scale_px"],
        *extra,
    ]
    with tempfile.TemporaryDirectory() as workdir:
        result = subprocess.run(command, cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{row['image_id']}: measurement failed: {result.stdout or result.stderr}")
    payload = json.loads(result.stdout)

    return {
        "image_id": row["image_id"],
        "config": extra,
        "scale_nm": float(row["scale_nm"]),
        "manual_scale_px": float(row["scale_px"]),
        "nm_per_px": _round(payload["nm_per_px"]),
        "scale_method": payload["selected_scale"]["method"],
        "warnings": payload["warnings"],
        "summary": _round(payload["summary"]),
        "object_summary": _round(payload["object_summary"]),
        "decorated_particle_metrics": _round(payload.get("decorated_particle_metrics")),
        "objects": [
            {field: _round(obj.get(field)) for field in OBJECT_FIELDS}
            for obj in payload["measurements"]
        ],
    }


def ground_truth_comparison(row, record):
    """How the run compares against the phantom's exact ground truth."""
    truth = json.loads((DEMO_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    entry = truth[row["image_id"]]
    reference = [
        particle["outer_diameter_nm"] or particle["core_diameter_nm"]
        for particle in entry["particles"]
    ]
    detected = [
        obj["outer_major_axis"] or obj["inner_major_axis"]
        for obj in record["objects"]
        if (obj["outer_major_axis"] or obj["inner_major_axis"])
    ]
    comparison = {
        "true_particles": len(reference),
        "detected_objects": len(record["objects"]),
        "true_mean_diameter_nm": _round(sum(reference) / len(reference)) if reference else None,
        "detected_mean_diameter_nm": _round(sum(detected) / len(detected)) if detected else None,
    }
    if reference and detected:
        comparison["mean_diameter_ratio"] = _round(
            comparison["detected_mean_diameter_nm"] / comparison["true_mean_diameter_nm"]
        )
    return comparison


def _compare(expected, actual, tolerance, path="", differences=None):
    """Recursive numeric comparison; returns a list of human-readable diffs."""
    differences = differences if differences is not None else []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                differences.append(f"{path}.{key}: unexpected key (value {actual[key]!r})")
            elif key not in actual:
                differences.append(f"{path}.{key}: missing key (expected {expected[key]!r})")
            else:
                _compare(expected[key], actual[key], tolerance, f"{path}.{key}", differences)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append(f"{path}: length {len(actual)}, expected {len(expected)}")
        else:
            for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
                _compare(want, got, tolerance, f"{path}[{index}]", differences)
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)) \
            and not isinstance(expected, bool) and not isinstance(actual, bool):
        scale = max(abs(expected), abs(actual), 1.0)
        if abs(expected - actual) > tolerance * scale:
            differences.append(f"{path}: {actual!r}, expected {expected!r} "
                               f"(relative deviation {abs(expected - actual) / scale:.3g})")
    elif expected != actual:
        differences.append(f"{path}: {actual!r}, expected {expected!r}")
    return differences


def _readme(index):
    """Human-readable agreement summary for the committed baseline."""
    lines = [
        "# Expected demo results",
        "",
        "Baseline output of `measurement_modes.py` on the synthetic demo images,",
        "regenerated by `python scripts/build_demo_expected.py`. CI runs",
        "`--check` and fails if the measurement pipeline drifts.",
        "",
        f"Numeric comparison uses a relative tolerance of `{DEFAULT_TOLERANCE}` by default;",
        "`--tolerance` relaxes it for cross-platform runs.",
        "",
        "## Agreement with ground truth",
        "",
        "The demo images are phantoms, so the true particle geometry is known",
        "exactly. This table is a sanity check on the classical pipeline, not a",
        "scientific accuracy claim -- that is Workstream F (`docs/validation/`).",
        "",
        "| Image | Objects found | True particles | Mean diameter (Corpus) | Mean diameter (true) | Ratio |",
        "|---|---|---|---|---|---|",
    ]
    for image_id, entry in index["images"].items():
        comparison = entry["ground_truth_comparison"]
        detected = comparison["detected_mean_diameter_nm"]
        true_mean = comparison["true_mean_diameter_nm"]
        ratio = comparison.get("mean_diameter_ratio")
        lines.append(
            f"| `{image_id}` | {comparison['detected_objects']} | {comparison['true_particles']} | "
            f"{f'{detected:.1f} nm' if detected else 'n/a'} | "
            f"{f'{true_mean:.1f} nm' if true_mean else 'n/a'} | "
            f"{f'{ratio:.3f}' if ratio else 'n/a'} |"
        )
    lines += [
        "",
        "## Configuration per image",
        "",
        "Radius windows are set per image because they are physical bounds, not",
        "universal defaults. The scale bar is supplied as `--manual-scale-px 100`",
        "so the baseline does not depend on the auto-detector's heuristics.",
        "",
        "| Image | Arguments |",
        "|---|---|",
    ]
    for image_id, entry in index["images"].items():
        lines.append(f"| `{image_id}` | `{' '.join(entry['config'])}` |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Compare against the committed baseline.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help=f"Relative tolerance for numeric comparison (default {DEFAULT_TOLERANCE}).")
    args = parser.parse_args(argv)

    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_metadata()
    index = {"images": {}}
    all_differences = {}

    for row in rows:
        record = run_one(row)
        record["ground_truth_comparison"] = ground_truth_comparison(row, record)
        target = EXPECTED_DIR / f"{row['image_id']}.json"
        text = json.dumps(record, indent=2, sort_keys=True) + "\n"

        if args.check:
            if not target.exists():
                all_differences[row["image_id"]] = ["baseline file is missing"]
            else:
                expected = json.loads(target.read_text(encoding="utf-8"))
                differences = _compare(expected, record, args.tolerance)
                if differences:
                    all_differences[row["image_id"]] = differences[:20]
        else:
            target.write_text(text, encoding="utf-8")

        index["images"][row["image_id"]] = {
            "config": record["config"],
            "objects": len(record["objects"]),
            "nm_per_px": record["nm_per_px"],
            "ground_truth_comparison": record["ground_truth_comparison"],
        }

    index_path = EXPECTED_DIR / "index.json"
    readme_path = EXPECTED_DIR / "README.md"
    index_text = json.dumps(index, indent=2, sort_keys=True) + "\n"
    readme_text = _readme(index)

    if args.check:
        if not index_path.exists():
            all_differences.setdefault("index.json", []).append("baseline file is missing")
        else:
            committed = json.loads(index_path.read_text(encoding="utf-8"))
            differences = _compare(committed, index, args.tolerance)
            if differences:
                all_differences.setdefault("index.json", []).extend(differences[:20])
        if not readme_path.exists() or readme_path.read_text(encoding="utf-8") != readme_text:
            all_differences.setdefault("README.md", []).append("differs from the current run")
    else:
        index_path.write_text(index_text, encoding="utf-8")
        readme_path.write_text(readme_text, encoding="utf-8")

    if args.check and all_differences:
        print(json.dumps({"ok": False, "differences": all_differences}, indent=2))
        return 1

    print(json.dumps({
        "ok": True,
        "mode": "check" if args.check else "write",
        "images": len(rows),
        "output": str(EXPECTED_DIR.relative_to(REPO_ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
