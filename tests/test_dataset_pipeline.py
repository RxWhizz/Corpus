"""Dataset pipeline hardening (Workstream H1).

The guarantees under test: COCO stays the master format, YOLO is derived,
splits are deterministic, source groups never cross splits, and missing
provenance is surfaced rather than swallowed.
"""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from audit_training_dataset import audit_dataset
from common_training import MANIFEST_FIELDS, file_sha256, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"


def build_dataset(root, rows, class_names=("Au_core", "SiO2_outer")):
    """Materialise a minimal YOLO-seg dataset from manifest-style rows."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
        + "\n",
        encoding="utf-8",
    )
    manifest = []
    for row in rows:
        split = row["split"]
        stem = row["image_id"]
        image_path = root / "images" / split / f"{stem}.png"
        label_path = root / "labels" / split / f"{stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), np.full((64, 64), 200, dtype=np.uint8))
        label_path.write_text("0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n", encoding="utf-8")
        manifest.append({
            "image_id": stem,
            "source_id": row.get("source_id", "group_a"),
            "split": split,
            "dataset_layer": row.get("dataset_layer", "synthetic_core_shell"),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "file_sha256": row.get("file_sha256", file_sha256(image_path)),
            "labels": 1,
            "au_core_labels": 1,
            "sio2_outer_labels": 0,
            "annotation_review": row.get("annotation_review", "ready"),
            "skipped_review_labels": row.get("skipped_review_labels", 0),
            "nm_per_px": row.get("nm_per_px", "0.5"),
            "calibration_state": row.get("calibration_state", "confirmed"),
            "license": row.get("license", "CC0-1.0"),
            "license_status": row.get("license_status", "accepted"),
            "doi": row.get("doi", "10.0000/demo"),
            "source_url": row.get("source_url", "https://example.invalid/demo"),
        })
    write_manifest(root / "manifest.csv", manifest)
    return root


class TestManifestContract:
    def test_manifest_declares_every_provenance_column(self):
        for field in ("file_sha256", "license", "license_status", "doi", "source_url",
                      "split", "source_id", "annotation_review", "calibration_state"):
            assert field in MANIFEST_FIELDS

    def test_manifest_declares_class_counts(self):
        for field in ("labels", "au_core_labels", "sio2_outer_labels"):
            assert field in MANIFEST_FIELDS

    def test_file_sha256_is_content_addressed(self, tmp_path):
        path = tmp_path / "a.bin"
        path.write_bytes(b"corpus")
        first = file_sha256(path)
        path.write_bytes(b"corpuz")
        assert file_sha256(path) != first


class TestLeakageAudit:
    def test_a_clean_dataset_passes(self, tmp_path):
        root = build_dataset(tmp_path / "clean", [
            {"image_id": "a1", "source_id": "group_a", "split": "train"},
            {"image_id": "a2", "source_id": "group_a", "split": "train"},
            {"image_id": "b1", "source_id": "group_b", "split": "val"},
        ])
        result = audit_dataset(root)
        assert result["ok"], result["errors"]

    def test_a_source_group_across_two_splits_fails(self, tmp_path):
        root = build_dataset(tmp_path / "leaky", [
            {"image_id": "a1", "source_id": "group_a", "split": "train"},
            {"image_id": "a2", "source_id": "group_a", "split": "val"},
        ])
        result = audit_dataset(root)
        assert not result["ok"]
        assert any("multiple splits" in error for error in result["errors"])

    def test_the_offending_group_is_named(self, tmp_path):
        root = build_dataset(tmp_path / "leaky", [
            {"image_id": "a1", "source_id": "micrograph_17", "split": "train"},
            {"image_id": "a2", "source_id": "micrograph_17", "split": "test"},
        ])
        result = audit_dataset(root)
        assert any("micrograph_17" in error for error in result["errors"])

    def test_tiles_from_one_micrograph_cannot_cross_splits(self, tmp_path):
        # Tiles share a source_id, which is exactly what prevents the leak.
        root = build_dataset(tmp_path / "tiles", [
            {"image_id": f"tile_{index}", "source_id": "micrograph_1",
             "split": "train" if index < 3 else "val"}
            for index in range(4)
        ])
        assert not audit_dataset(root)["ok"]


class TestProvenanceAudit:
    def test_a_missing_manifest_is_an_error(self, tmp_path):
        root = build_dataset(tmp_path / "nomanifest", [
            {"image_id": "a1", "source_id": "group_a", "split": "train"},
        ])
        (root / "manifest.csv").unlink()
        result = audit_dataset(root)
        assert not result["ok"]
        assert any("manifest" in error.lower() for error in result["errors"])

    def test_missing_licence_is_surfaced(self, tmp_path):
        root = build_dataset(tmp_path / "nolicense", [
            {"image_id": "a1", "source_id": "group_a", "split": "train",
             "license": "", "license_status": ""},
        ])
        result = audit_dataset(root)
        assert result["provenance"]["missing_license"] == 1
        assert any("license" in warning for warning in result["warnings"])

    def test_missing_source_is_surfaced(self, tmp_path):
        root = build_dataset(tmp_path / "nosource", [
            {"image_id": "a1", "source_id": "group_a", "split": "train",
             "doi": "", "source_url": ""},
        ])
        assert audit_dataset(root)["provenance"]["missing_source"] == 1

    def test_missing_checksum_is_surfaced(self, tmp_path):
        root = build_dataset(tmp_path / "nosum", [
            {"image_id": "a1", "source_id": "group_a", "split": "train", "file_sha256": ""},
        ])
        assert audit_dataset(root)["provenance"]["missing_checksum"] == 1

    def test_missing_calibration_is_surfaced(self, tmp_path):
        root = build_dataset(tmp_path / "nocal", [
            {"image_id": "a1", "source_id": "group_a", "split": "train", "nm_per_px": ""},
        ])
        assert audit_dataset(root)["provenance"]["missing_calibration"] == 1

    def test_a_public_demo_row_without_an_accepted_licence_fails(self, tmp_path):
        root = build_dataset(tmp_path / "demo", [
            {"image_id": "a1", "source_id": "group_a", "split": "train",
             "dataset_layer": "public_demo", "license_status": "missing"},
        ])
        result = audit_dataset(root)
        assert not result["ok"]
        assert any("license status" in error for error in result["errors"])

    def test_partial_annotation_review_is_counted(self, tmp_path):
        root = build_dataset(tmp_path / "partial", [
            {"image_id": "a1", "source_id": "group_a", "split": "train",
             "annotation_review": "partial", "skipped_review_labels": 2},
        ])
        assert audit_dataset(root)["provenance"]["partial_annotation_review"] == 1


class TestCocoToYoloExport:
    """Runs the real export scripts; COCO in, YOLO out, twice."""

    @staticmethod
    def generate_and_export(workdir, seed="5", count="6"):
        workdir = Path(workdir)
        synthetic = workdir / "syn"
        result = subprocess.run(
            [sys.executable, str(TRAINING_DIR / "generate_synthetic_core_shell.py"),
             "--out", str(synthetic), "--count", count, "--seed", seed,
             "--height", "256", "--width", "256", "--min-particles", "3", "--max-particles", "5"],
            cwd=TRAINING_DIR, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        coco_path = synthetic / "synthetic_core_shell_coco.json"

        exports = []
        for index in (1, 2):
            out = workdir / f"yolo_{index}"
            result = subprocess.run(
                [sys.executable, str(TRAINING_DIR / "prepare_yolo_seg.py"),
                 "--coco", str(coco_path), "--out", str(out)],
                cwd=TRAINING_DIR, capture_output=True, text=True,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            exports.append(out)
        return coco_path, exports

    @classmethod
    @pytest.fixture(scope="class")
    def export(cls):
        with tempfile.TemporaryDirectory() as workdir:
            yield cls.generate_and_export(workdir)

    def test_coco_is_the_master_format(self, export):
        coco_path, _ = export
        payload = json.loads(coco_path.read_text(encoding="utf-8"))
        assert {"images", "annotations", "categories"} <= set(payload)
        assert payload["annotations"]

    def test_export_produces_a_usable_dataset(self, export):
        _, exports = export
        assert (exports[0] / "data.yaml").exists()
        assert (exports[0] / "manifest.csv").exists()
        assert list((exports[0] / "labels").rglob("*.txt"))

    def test_repeated_export_produces_identical_splits(self, export):
        _, exports = export

        def splits(root):
            with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                return sorted((row["image_id"], row["source_id"], row["split"])
                              for row in csv.DictReader(handle))

        assert splits(exports[0]) == splits(exports[1])

    def test_repeated_export_produces_identical_image_bytes(self, export):
        _, exports = export

        def digests(root):
            with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                return sorted(row["file_sha256"] for row in csv.DictReader(handle))

        assert digests(exports[0]) == digests(exports[1])

    def test_the_export_carries_provenance(self, export):
        _, exports = export
        with (exports[0] / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        for row in rows:
            assert row["file_sha256"]
            assert row["license"]
            assert row["license_status"] == "accepted"
            assert row["nm_per_px"]
            assert row["annotation_review"] in ("ready", "partial")

    def test_the_export_passes_its_own_audit(self, export):
        _, exports = export
        result = audit_dataset(exports[0])
        assert result["ok"], result["errors"]

    def test_normalised_labels_stay_inside_the_unit_square(self, export):
        _, exports = export
        for label in (exports[0] / "labels").rglob("*.txt"):
            for line in label.read_text(encoding="utf-8").splitlines():
                values = [float(part) for part in line.split()[1:]]
                assert all(0.0 <= value <= 1.0 for value in values)


class TestSyntheticGeneratorRobustness:
    def test_a_small_canvas_does_not_crash(self, tmp_path):
        # A fine nm/px on a small frame used to raise "low >= high" from the
        # particle placement draw.
        result = subprocess.run(
            [sys.executable, str(TRAINING_DIR / "generate_synthetic_core_shell.py"),
             "--out", str(tmp_path / "syn"), "--count", "2", "--seed", "3",
             "--height", "192", "--width", "192", "--min-particles", "2", "--max-particles", "3"],
            cwd=TRAINING_DIR, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_an_unusable_canvas_raises_a_clear_error(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(TRAINING_DIR / "generate_synthetic_core_shell.py"),
             "--out", str(tmp_path / "syn"), "--count", "1", "--seed", "3",
             "--height", "8", "--width", "8", "--min-particles", "1", "--max-particles", "1"],
            cwd=TRAINING_DIR, capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "too small" in (result.stdout + result.stderr)
