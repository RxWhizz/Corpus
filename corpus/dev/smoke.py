"""One canonical smoke command for CI and for local sanity checks.

    python -m corpus.dev.smoke

Exercises the whole chain end to end, in a temporary directory, with no
private data and no network:

1. the classical measurement CLI on a demo image,
2. the committed demo baseline (``expected/``),
3. the synthetic COCO -> YOLO-seg -> audit dataset pipeline,
4. the validation harness on a small generated table,
5. the segmentation backend contract.

Each stage reports ``ok`` with a short detail line; the exit code is non-zero
if any stage fails, and ``--json`` prints the machine-readable result.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "Examples" / "demo_dataset"


def _run(command, cwd=None, timeout=900):
    result = subprocess.run(
        [sys.executable, *command],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


def stage_measurement():
    """The classical measurement CLI produces a valid record for a demo image."""
    image = DEMO_DIR / "images" / "demo_001_core_shell_spheres.png"
    if not image.exists():
        return False, f"demo image missing: {image.relative_to(REPO_ROOT)}"
    with tempfile.TemporaryDirectory() as workdir:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "measurement_modes.py"),
             "--image", str(image), "--scale", "100", "--manual-scale-px", "100",
             "--shape-preset", "spheres", "--mode", "both",
             "--au-min-radius", "5", "--au-max-radius", "40",
             "--sio2-min-radius", "30", "--sio2-max-radius", "90"],
            cwd=workdir, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return False, (result.stdout or result.stderr).strip()[:300]
        payload = json.loads(result.stdout)
        for artifact in ("processed_image.jpg", "measurements.json", "diameters.txt"):
            if not (Path(workdir) / artifact).exists():
                return False, f"missing output: {artifact}"
    if not payload.get("ok"):
        return False, str(payload.get("message"))
    if not payload.get("measurements"):
        return False, "no measurements produced for the demo image"
    if not payload.get("run_fingerprint", {}).get("image_sha256"):
        return False, "run fingerprint was not recorded"
    return True, (f"{len(payload['measurements'])} objects, "
                  f"nm_per_px={payload['nm_per_px']:g}, backend={payload['segmentation_backend']}")


def stage_demo_baseline():
    """The committed demo baseline still matches the current pipeline."""
    # The strict 1e-6 comparison runs on Linux in CI; here the bound is looser
    # so cross-platform float noise cannot fail the smoke test, while any real
    # measurement regression (orders of magnitude larger) still does.
    result = _run([str(REPO_ROOT / "scripts" / "build_demo_expected.py"),
                   "--check", "--tolerance", "1e-4"])
    if result.returncode != 0:
        return False, (result.stdout or result.stderr).strip()[:600]
    return True, "expected/ matches the current measurement output"


def stage_demo_dataset_integrity():
    """The demo images and metadata match their generator."""
    result = _run([str(REPO_ROOT / "scripts" / "build_demo_dataset.py"), "--check"])
    if result.returncode != 0:
        return False, (result.stdout or result.stderr).strip()[:600]
    return True, "demo images, metadata and annotations are unchanged"


def stage_dataset_pipeline():
    """Synthetic COCO -> YOLO-seg export -> leakage/licence audit."""
    training = REPO_ROOT / "training"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        synthetic_dir = tmp / "synthetic"
        yolo_dir = tmp / "yolo_seg"

        result = _run([str(training / "generate_synthetic_core_shell.py"),
                       "--out", str(synthetic_dir), "--count", "6", "--seed", "7",
                       "--height", "320", "--width", "320",
                       "--min-particles", "3", "--max-particles", "6"], cwd=training)
        if result.returncode != 0:
            return False, f"synthetic generation failed: {(result.stderr or result.stdout).strip()[:300]}"
        coco_path = synthetic_dir / "synthetic_core_shell_coco.json"
        if not coco_path.exists():
            return False, "synthetic COCO file was not written"

        result = _run([str(training / "prepare_yolo_seg.py"),
                       "--coco", str(coco_path), "--out", str(yolo_dir)], cwd=training)
        if result.returncode != 0:
            return False, f"YOLO export failed: {(result.stderr or result.stdout).strip()[:300]}"

        result = _run([str(training / "audit_training_dataset.py"),
                       "--dataset", str(yolo_dir)], cwd=training)
        if result.returncode != 0:
            return False, f"dataset audit failed: {(result.stderr or result.stdout).strip()[:400]}"

        manifest = yolo_dir / "manifest.csv"
        data_yaml = yolo_dir / "data.yaml"
        if not manifest.exists() or not data_yaml.exists():
            return False, "export is missing manifest.csv or data.yaml"
        images = sum(1 for _ in (yolo_dir / "labels").rglob("*.txt"))
        return True, f"{images} labelled images exported and audited"


def stage_dataset_determinism():
    """The same COCO and seed produce identical splits."""
    training = REPO_ROOT / "training"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        synthetic_dir = tmp / "synthetic"
        result = _run([str(training / "generate_synthetic_core_shell.py"),
                       "--out", str(synthetic_dir), "--count", "8", "--seed", "11",
                       "--height", "288", "--width", "288",
                       "--min-particles", "3", "--max-particles", "5"], cwd=training)
        if result.returncode != 0:
            return False, f"synthetic generation failed: {(result.stderr or result.stdout).strip()[:300]}"
        coco_path = synthetic_dir / "synthetic_core_shell_coco.json"

        manifests = []
        for index in (1, 2):
            out = tmp / f"yolo_{index}"
            result = _run([str(training / "prepare_yolo_seg.py"),
                           "--coco", str(coco_path), "--out", str(out)], cwd=training)
            if result.returncode != 0:
                return False, f"YOLO export {index} failed: {(result.stderr or result.stdout).strip()[:300]}"
            rows = (out / "manifest.csv").read_text(encoding="utf-8").splitlines()
            # Drop the absolute paths, which legitimately differ per run.
            manifests.append([",".join(row.split(",")[:4]) for row in rows])
        if manifests[0] != manifests[1]:
            return False, "repeated export produced different splits"
        return True, "repeated export produced identical splits"


def stage_validation():
    """The validation harness runs from one command and writes a report."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        table = tmp / "table.csv"
        lines = ["image_id,particle_id,reference_diameter_nm,corpus_diameter_nm,morphology"]
        for index in range(1, 13):
            reference = 90 + 3 * index
            corpus = reference + (1.0 if index % 2 else -0.8)
            lines.append(f"img_{(index - 1) // 4 + 1},p_{index},{reference},{corpus:.2f},core_shell_sphere")
        table.write_text("\n".join(lines) + "\n", encoding="utf-8")

        out = tmp / "report"
        result = _run(["-m", "corpus.validation", "--table", str(table), "--out", str(out)])
        if result.returncode != 0:
            return False, (result.stderr or result.stdout).strip()[:400]
        for name in ("report.json", "report.md", "metrics.csv", "paired_rows.csv"):
            if not (out / name).exists():
                return False, f"validation report is missing {name}"
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        metrics = report["quantities"]["diameter_nm"]["metrics"]
        figures = len(list(out.glob("*.png")))
        return True, (f"n={metrics['n']}, MAE={metrics['mae']:.3f} nm, "
                      f"{figures} figure(s)")


def stage_backends():
    """The segmentation backend contract is intact and ML-free."""
    import cv2

    from corpus.segmentation import available_backends, get_backend

    image = cv2.imread(str(DEMO_DIR / "images" / "demo_001_core_shell_spheres.png"))
    if image is None:
        return False, "could not read the demo image"
    names = available_backends()
    for name in names:
        backend = get_backend(name)
        if backend.requires_ml:
            return False, f"backend {name} requires an ML runtime in a v1.0 build"
    result = get_backend("classical").predict(image)
    if not result.contours:
        return False, "classical backend found no contours in the demo image"
    for module in ("torch", "ultralytics", "tensorflow"):
        if module in sys.modules:
            return False, f"{module} was imported by the classical path"
    return True, f"backends={names}, classical found {len(result.contours)} contours"


STAGES = (
    ("measurement", stage_measurement),
    ("demo_dataset_integrity", stage_demo_dataset_integrity),
    ("demo_baseline", stage_demo_baseline),
    ("dataset_pipeline", stage_dataset_pipeline),
    ("dataset_determinism", stage_dataset_determinism),
    ("validation", stage_validation),
    ("segmentation_backends", stage_backends),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="Print the result as JSON only.")
    parser.add_argument("--only", default="", help="Comma-separated stage names to run.")
    parser.add_argument("--list", action="store_true", help="List stage names and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for name, _ in STAGES:
            print(name)
        return 0

    wanted = {part.strip() for part in args.only.split(",") if part.strip()}
    unknown = wanted - {name for name, _ in STAGES}
    if unknown:
        print(f"Unknown stage(s): {sorted(unknown)}", file=sys.stderr)
        return 2

    results = {}
    failed = []
    for name, function in STAGES:
        if wanted and name not in wanted:
            continue
        started = time.monotonic()
        try:
            ok, detail = function()
        except Exception as error:  # a smoke test must report, never traceback
            ok, detail = False, f"{type(error).__name__}: {error}"
        elapsed = round(time.monotonic() - started, 2)
        results[name] = {"ok": ok, "detail": detail, "seconds": elapsed}
        if not ok:
            failed.append(name)
        if not args.json:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} ({elapsed}s): {detail}")

    payload = {"ok": not failed, "failed": failed, "stages": results}
    if args.json:
        print(json.dumps(payload, indent=2))
    elif failed:
        print(f"\nSmoke test FAILED: {', '.join(failed)}")
    else:
        print(f"\nSmoke test passed ({len(results)} stages).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
