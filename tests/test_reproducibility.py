"""Same image + same settings must give the same numbers.

This is the property the epic's Definition of Done rests on, so it is tested
at three levels: the fingerprint, the pure functions, and an end-to-end run of
``measurement_modes.py`` as a subprocess.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

from corpus.io import file_sha256, run_fingerprint, settings_fingerprint
from corpus.measurement import contour_measurements
from corpus.metrology import object_row, summarize_objects
from corpus.segmentation import particle_binary, to_gray

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "measurement_modes.py"

BASE_SETTINGS = {
    "mode": "both",
    "shape_preset": "spheres",
    "scale_nm": 200.0,
    "manual_scale_px": 100.0,
    "exclude_edges": True,
    "watershed": True,
    "min_circularity": 0.0,
    "max_circularity": 1.0,
    "review_view": "overlay",
}


@pytest.fixture
def written_image(tmp_path, noisy_particles):
    path = tmp_path / "field.png"
    cv2.imwrite(str(path), noisy_particles["image"])
    return path


class TestFileChecksum:
    def test_same_bytes_give_the_same_digest(self, tmp_path):
        first = tmp_path / "a.bin"
        second = tmp_path / "b.bin"
        first.write_bytes(b"corpus" * 1000)
        second.write_bytes(b"corpus" * 1000)
        assert file_sha256(first) == file_sha256(second)

    def test_one_changed_byte_changes_the_digest(self, tmp_path):
        path = tmp_path / "a.bin"
        path.write_bytes(b"corpus")
        before = file_sha256(path)
        path.write_bytes(b"corpuz")
        assert file_sha256(path) != before

    def test_digest_is_the_documented_algorithm(self, tmp_path):
        import hashlib

        path = tmp_path / "a.bin"
        payload = b"nanoparticle"
        path.write_bytes(payload)
        assert file_sha256(path) == hashlib.sha256(payload).hexdigest()


class TestSettingsFingerprint:
    def test_identical_settings_match(self):
        assert settings_fingerprint(dict(BASE_SETTINGS)) == settings_fingerprint(dict(BASE_SETTINGS))

    def test_key_order_does_not_matter(self):
        reversed_order = dict(reversed(list(BASE_SETTINGS.items())))
        assert settings_fingerprint(reversed_order) == settings_fingerprint(BASE_SETTINGS)

    def test_a_changed_threshold_changes_the_fingerprint(self):
        changed = dict(BASE_SETTINGS, min_circularity=0.5)
        assert settings_fingerprint(changed) != settings_fingerprint(BASE_SETTINGS)

    def test_a_changed_calibration_changes_the_fingerprint(self):
        changed = dict(BASE_SETTINGS, scale_nm=500.0)
        assert settings_fingerprint(changed) != settings_fingerprint(BASE_SETTINGS)

    def test_presentation_only_changes_do_not_change_the_fingerprint(self):
        # review_view only affects the overlay, never a measured value.
        changed = dict(BASE_SETTINGS, review_view="numbered")
        assert settings_fingerprint(changed) == settings_fingerprint(BASE_SETTINGS)

    def test_run_fingerprint_binds_image_settings_and_version(self, written_image):
        fingerprint = run_fingerprint(written_image, BASE_SETTINGS, "1.0.0")
        assert set(fingerprint) == {"corpus_version", "image_sha256", "settings_sha256"}
        assert fingerprint["image_sha256"] == file_sha256(written_image)


class TestPureFunctionDeterminism:
    def test_repeated_measurement_of_one_image_is_identical(self, noisy_particles, default_filters):
        def measure():
            binary = particle_binary(to_gray(noisy_particles["image"]), default_filters)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contour_measurements(
                contours, "Au", 5, 200, 0.5, (0, 0, 255), None, [], True,
                filter_settings=default_filters, image_shape=noisy_particles["image"].shape,
            )

        assert measure() == measure()

    def test_object_rows_are_identical_across_runs(self):
        def build():
            inner = {"major_axis": 40.0, "minor_axis": 40.0, "diameter": 40.0,
                     "equivalent_diameter": 40.0, "center_x": 10.0, "center_y": 10.0,
                     "radius_px": 20.0, "flags": [], "backend": "classical"}
            outer = dict(inner, major_axis=100.0, minor_axis=100.0, diameter=100.0)
            return [object_row(index, "spheres", inner, outer) for index in range(1, 6)]

        assert build() == build()

    def test_summaries_are_order_independent_for_their_totals(self):
        inner = {"major_axis": 40.0, "minor_axis": 40.0, "diameter": 40.0,
                 "equivalent_diameter": 40.0, "center_x": 0.0, "center_y": 0.0,
                 "radius_px": 20.0, "flags": [], "backend": "classical"}
        outer = dict(inner, major_axis=100.0, minor_axis=100.0)
        rows = [object_row(index, "spheres", inner, outer) for index in range(1, 4)]
        assert summarize_objects(rows) == summarize_objects(list(reversed(rows)))


class TestEndToEndReproducibility:
    """Runs the real CLI twice and compares the JSON records."""

    @staticmethod
    def run(image_path, workdir, extra=()):
        command = [
            sys.executable, str(SCRIPT),
            "--image", str(image_path),
            "--scale", "200",
            "--manual-scale-px", "100",
            "--shape-preset", "generic",
            "--mode", "au",
            *extra,
        ]
        result = subprocess.run(command, cwd=workdir, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)

    def test_two_runs_produce_identical_measurements(self, written_image, tmp_path):
        first_dir = tmp_path / "run_1"
        second_dir = tmp_path / "run_2"
        first_dir.mkdir()
        second_dir.mkdir()

        first = self.run(written_image, first_dir)
        second = self.run(written_image, second_dir)

        assert first["ok"] and second["ok"]
        assert first["run_fingerprint"] == second["run_fingerprint"]
        assert first["nm_per_px"] == second["nm_per_px"]
        assert first["measurements"] == second["measurements"]
        assert first["class_measurements"] == second["class_measurements"]
        assert first["summary"] == second["summary"]
        assert first["object_summary"] == second["object_summary"]
        assert first["normality_report"] == second["normality_report"]

    def test_rerunning_in_place_rewrites_a_byte_identical_json(self, written_image, tmp_path):
        # measurements.json records the absolute output paths, so byte
        # equality is only meaningful for repeated runs in the same directory.
        workdir = tmp_path / "run"
        workdir.mkdir()
        self.run(written_image, workdir)
        first = file_sha256(workdir / "measurements.json")
        self.run(written_image, workdir)
        assert file_sha256(workdir / "measurements.json") == first

    def test_runs_in_different_directories_differ_only_by_those_paths(self, written_image, tmp_path):
        payloads = []
        for name in ("run_a", "run_b"):
            workdir = tmp_path / name
            workdir.mkdir()
            payloads.append(self.run(written_image, workdir))
        path_keys = {"processed_image_path", "measurements_path"}
        assert payloads[0]["processed_image_path"] != payloads[1]["processed_image_path"]
        stripped = [{k: v for k, v in payload.items() if k not in path_keys} for payload in payloads]
        assert stripped[0] == stripped[1]

    def test_a_different_setting_changes_the_fingerprint_and_the_result(self, written_image, tmp_path):
        base_dir = tmp_path / "base"
        strict_dir = tmp_path / "strict"
        base_dir.mkdir()
        strict_dir.mkdir()

        base = self.run(written_image, base_dir)
        strict = self.run(written_image, strict_dir, ["--min-circularity", "0.99"])

        assert base["run_fingerprint"]["settings_sha256"] != strict["run_fingerprint"]["settings_sha256"]
        assert len(strict["measurements"]) < len(base["measurements"])

    def test_the_run_records_its_version_and_backend(self, written_image, tmp_path):
        workdir = tmp_path / "run"
        workdir.mkdir()
        payload = self.run(written_image, workdir)
        assert payload["corpus_version"]
        assert payload["segmentation_backend"] == "classical"

    def test_compat_outputs_are_still_written(self, written_image, tmp_path):
        workdir = tmp_path / "run"
        workdir.mkdir()
        self.run(written_image, workdir)
        assert (workdir / "processed_image.jpg").exists()
        assert (workdir / "diameters.txt").exists()
        assert (workdir / "measurements.json").exists()

    def test_a_missing_image_fails_with_a_json_error(self, tmp_path):
        workdir = tmp_path / "run"
        workdir.mkdir()
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--image", str(tmp_path / "nope.png"), "--scale", "100"],
            cwd=workdir, capture_output=True, text=True,
        )
        assert result.returncode != 0
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert "not found" in payload["message"].lower()

    def test_an_image_without_a_scale_bar_fails_with_a_json_error(self, tmp_path, blank_image):
        workdir = tmp_path / "run"
        workdir.mkdir()
        image_path = tmp_path / "blank.png"
        cv2.imwrite(str(image_path), blank_image)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--image", str(image_path), "--scale", "100"],
            cwd=workdir, capture_output=True, text=True,
        )
        assert result.returncode != 0
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert "scale bar" in payload["message"].lower()
