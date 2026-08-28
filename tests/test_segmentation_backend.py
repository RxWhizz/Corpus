"""The segmentation backend contract (Workstream I1).

The point of these tests is that adding a future YOLO-seg backend cannot
change the measurement contract, and that classical measurement keeps working
with no ML dependency installed.
"""

import sys

import cv2
import pytest

from corpus.measurement import contour_measurements
from corpus.segmentation import (
    ClassicalBackend,
    ManualBackend,
    SegmentationBackend,
    SegmentationResult,
    available_backends,
    get_backend,
)


class TestNoMlDependency:
    def test_classical_measurement_imports_no_ml_runtime(self):
        # Importing the whole scientific core must not pull in torch or
        # ultralytics; the classical path has to work on a bare install.
        import corpus  # noqa: F401
        import corpus.measurement  # noqa: F401
        import corpus.metrology  # noqa: F401
        import corpus.segmentation  # noqa: F401

        for module in ("torch", "ultralytics", "tensorflow"):
            assert module not in sys.modules

    def test_shipped_backends_declare_no_ml_requirement(self):
        for name in available_backends():
            assert get_backend(name).requires_ml is False


class TestRegistry:
    def test_classical_and_manual_are_registered(self):
        assert "classical" in available_backends()
        assert "manual" in available_backends()

    def test_unknown_backend_names_are_rejected_with_the_options(self):
        with pytest.raises(ValueError, match="Unknown segmentation backend"):
            get_backend("yolo_seg_v99")

    def test_every_backend_implements_the_contract(self):
        for name in available_backends():
            backend = get_backend(name)
            assert isinstance(backend, SegmentationBackend)
            assert callable(backend.predict)
            assert set(backend.describe()) == {"name", "requires_ml", "class"}

    def test_the_abstract_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            SegmentationBackend()


class TestSegmentationResult:
    def test_records_the_backend_and_method(self):
        result = SegmentationResult(contours=[], backend="classical", method="watershed")
        provenance = result.as_provenance()
        assert provenance["backend"] == "classical"
        assert provenance["separation_method"] == "watershed"

    def test_review_is_required_by_default(self):
        # A new backend that forgets to think about review gets the safe default.
        assert SegmentationResult().review_required is True

    def test_scores_must_align_with_contours(self):
        with pytest.raises(ValueError, match="align"):
            SegmentationResult(contours=[1, 2], scores=[0.5])

    def test_provenance_is_a_copy_not_a_live_reference(self):
        result = SegmentationResult(metadata={"model": "v1"})
        provenance = result.as_provenance()
        provenance["backend_metadata"]["model"] = "mutated"
        assert result.metadata["model"] == "v1"


class TestClassicalBackend:
    def test_finds_an_isolated_particle(self, single_particle):
        result = ClassicalBackend().predict(single_particle["image"])
        assert len(result.contours) == 1
        assert result.backend == "classical"
        assert result.method == "contour"

    def test_deterministic_output_does_not_demand_review(self, single_particle):
        assert ClassicalBackend().predict(single_particle["image"]).review_required is False

    def test_watershed_is_reported_when_it_actually_splits(self, two_touching_particles):
        result = ClassicalBackend().predict(
            two_touching_particles["image"], watershed=True, min_radius_px=10, max_radius_px=100
        )
        assert result.method == "watershed"
        assert len(result.contours) == 2

    def test_watershed_falls_back_to_contour_when_nothing_splits(self, single_particle):
        result = ClassicalBackend().predict(
            single_particle["image"], watershed=True, min_radius_px=10, max_radius_px=100
        )
        assert result.method == "contour"

    def test_carrier_target_uses_the_adaptive_mask(self, core_shell_particle):
        result = ClassicalBackend(target="carriers").predict(core_shell_particle["image"])
        assert result.metadata["target"] == "carriers"
        assert result.contours

    def test_an_invalid_target_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="target must be"):
            ClassicalBackend(target="nonsense")

    def test_is_deterministic(self, noisy_particles):
        backend = ClassicalBackend()
        first = [cv2.contourArea(c) for c in backend.predict(noisy_particles["image"]).contours]
        second = [cv2.contourArea(c) for c in backend.predict(noisy_particles["image"]).contours]
        assert first == second


class TestManualBackend:
    def test_passes_human_contours_through(self, single_particle):
        source = ClassicalBackend().predict(single_particle["image"]).contours
        result = ManualBackend().predict(single_particle["image"], contours=source)
        assert result.backend == "manual"
        assert result.method == "manual"
        assert len(result.contours) == len(source)

    def test_human_input_does_not_need_further_review(self, single_particle):
        assert ManualBackend().predict(single_particle["image"], contours=[]).review_required is False


class TestBackendIsInterchangeable:
    def test_any_backend_result_feeds_the_same_measurement_contract(self, single_particle, default_filters):
        """The whole point of I1: swapping the backend changes nothing downstream."""
        image = single_particle["image"]
        classical = ClassicalBackend().predict(image)
        manual = ManualBackend().predict(image, contours=classical.contours)

        rows = []
        for result in (classical, manual):
            rows.append(contour_measurements(
                result.contours, "Au", 5, 200, 1.0, (0, 0, 255), None, [], True,
                separation_method=result.method, filter_settings=default_filters,
                image_shape=image.shape, backend=result.backend,
            ))

        assert len(rows[0]) == len(rows[1]) == 1
        assert rows[0][0]["diameter"] == pytest.approx(rows[1][0]["diameter"])
        assert rows[0][0]["backend"] == "classical"
        assert rows[1][0]["backend"] == "manual"

    def test_every_measurement_records_which_backend_produced_it(self, single_particle, default_filters):
        image = single_particle["image"]
        result = ClassicalBackend().predict(image)
        rows = contour_measurements(
            result.contours, "Au", 5, 200, 1.0, (0, 0, 255), None, [], True,
            filter_settings=default_filters, image_shape=image.shape, backend=result.backend,
        )
        assert rows and all(row["backend"] == "classical" for row in rows)
