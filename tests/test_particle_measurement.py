"""Contour geometry and the calibrated particle record."""

import math

import cv2
import numpy as np
import pytest

from corpus.measurement import (
    aspect_ratio,
    circularity,
    classify_shape,
    contour_measurements,
    equivalent_diameter_px,
    flat_measurement,
    is_duplicate,
    nearest_measurement,
    overlap_ratio,
    touches_edge,
)
from corpus.segmentation import particle_binary, to_gray


def contours_of(image, settings=None):
    binary = particle_binary(to_gray(image), settings or {})
    found, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return found


class TestGeometryPrimitives:
    def test_circularity_of_a_disc_is_high(self, single_particle):
        # A rasterised disc does not reach 1.0: the traced polygon perimeter
        # overestimates the true circumference (~264 px vs 2*pi*40 = 251 px),
        # which puts a perfect disc near 0.89. The `classify_shape` round
        # threshold of 0.55 is set with that headroom in mind.
        contour = max(contours_of(single_particle["image"]), key=cv2.contourArea)
        value = circularity(cv2.contourArea(contour), cv2.arcLength(contour, True))
        assert 0.85 <= value <= 1.05

    def test_circularity_of_degenerate_input_is_zero(self):
        assert circularity(0, 10) == 0.0
        assert circularity(10, 0) == 0.0

    def test_equivalent_diameter_inverts_disc_area(self):
        assert equivalent_diameter_px(math.pi * 25**2) == pytest.approx(50.0)

    def test_equivalent_diameter_of_nothing_is_zero(self):
        assert equivalent_diameter_px(0) == 0.0

    def test_aspect_ratio_survives_a_zero_minor_axis(self):
        assert aspect_ratio(10, 0) > 0

    @pytest.mark.parametrize(
        "circ,ar,expected",
        [(1.0, 1.0, "round"), (0.6, 1.5, "round"), (0.5, 3.0, "elongated"),
         (0.3, 2.5, "elongated"), (0.1, 1.2, None), (0.2, 5.0, None)],
    )
    def test_shape_classification_boundaries(self, circ, ar, expected):
        assert classify_shape(circ, ar) == expected

    def test_overlap_ratio_is_relative_to_the_first_box(self):
        assert overlap_ratio((0, 0, 10, 10), (0, 0, 5, 10)) == pytest.approx(0.5)
        assert overlap_ratio((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0
        assert overlap_ratio((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_touches_edge_detects_each_side(self):
        shape = (100, 100)
        assert touches_edge((0, 40, 10, 10), shape)
        assert touches_edge((40, 0, 10, 10), shape)
        assert touches_edge((92, 40, 10, 10), shape)
        assert touches_edge((40, 92, 10, 10), shape)
        assert not touches_edge((40, 40, 10, 10), shape)


class TestDeduplication:
    def test_concentric_detections_are_duplicates(self):
        first = {"center_x": 100, "center_y": 100, "radius_px": 20}
        assert is_duplicate({"center_x": 102, "center_y": 101, "radius_px": 20}, [first])

    def test_well_separated_detections_are_kept(self):
        first = {"center_x": 100, "center_y": 100, "radius_px": 20}
        assert not is_duplicate({"center_x": 200, "center_y": 200, "radius_px": 20}, [first])

    def test_nearest_measurement_picks_the_closest_within_range(self):
        candidates = [
            {"center_x": 110, "center_y": 100},
            {"center_x": 103, "center_y": 100},
        ]
        best, distance = nearest_measurement((100, 100), candidates, 20)
        assert best is candidates[1]
        assert distance == pytest.approx(3.0)

    def test_nearest_measurement_respects_the_radius(self):
        best, distance = nearest_measurement((100, 100), [{"center_x": 300, "center_y": 300}], 20)
        assert best is None and distance is None


class TestFlatMeasurement:
    def test_pixels_are_converted_with_the_calibration(self):
        row = flat_measurement("Au", 10, 20, 100, 50, math.pi * 50**2, 0, "round", 0.9, nm_per_px=0.5)
        assert row["major_axis"] == pytest.approx(50.0)
        assert row["minor_axis"] == pytest.approx(25.0)
        assert row["diameter"] == row["major_axis"]
        assert row["radius_px"] == pytest.approx(50.0)

    def test_records_its_backend_and_separation_method(self):
        row = flat_measurement("Au", 0, 0, 10, 10, 78.5, 0, "round", 0.9, 1.0,
                               separation_method="watershed", backend="classical")
        assert row["separation_method"] == "watershed"
        assert row["backend"] == "classical"

    def test_scaling_the_calibration_scales_every_length(self):
        base = flat_measurement("Au", 0, 0, 80, 40, 2500, 0, "round", 0.9, 1.0)
        doubled = flat_measurement("Au", 0, 0, 80, 40, 2500, 0, "round", 0.9, 2.0)
        for key in ("diameter", "major_axis", "minor_axis", "equivalent_diameter"):
            assert doubled[key] == pytest.approx(2 * base[key])

    def test_pixel_quantities_are_calibration_independent(self):
        base = flat_measurement("Au", 0, 0, 80, 40, 2500, 0, "round", 0.9, 1.0)
        doubled = flat_measurement("Au", 0, 0, 80, 40, 2500, 0, "round", 0.9, 2.0)
        assert base["area_px"] == doubled["area_px"]
        assert base["radius_px"] == doubled["radius_px"]


class TestContourMeasurements:
    def test_measures_an_isolated_disc_to_within_a_pixel(self, single_particle, default_filters):
        rows = contour_measurements(
            contours_of(single_particle["image"]), "Au", 5, 200, 1.0, (0, 0, 255),
            None, [], True, filter_settings=default_filters,
            image_shape=single_particle["image"].shape,
        )
        assert len(rows) == 1
        assert rows[0]["diameter"] == pytest.approx(2 * single_particle["radius_px"], abs=2.5)
        assert rows[0]["shape"] == "round"

    def test_finds_nothing_in_a_blank_frame(self, blank_image, default_filters):
        rows = contour_measurements(
            contours_of(blank_image), "Au", 5, 200, 1.0, (0, 0, 255),
            None, [], True, filter_settings=default_filters, image_shape=blank_image.shape,
        )
        assert rows == []

    def test_classifies_a_rod_as_elongated(self, elongated_particle, default_filters):
        rows = contour_measurements(
            contours_of(elongated_particle["image"]), "Au", 5, 300, 1.0, (0, 0, 255),
            None, [], True, filter_settings=default_filters,
            image_shape=elongated_particle["image"].shape,
        )
        assert len(rows) == 1
        assert rows[0]["shape"] == "elongated"
        assert rows[0]["aspect_ratio"] == pytest.approx(elongated_particle["expected_aspect_ratio"], rel=0.15)

    def test_edge_particle_is_excluded_when_asked(self, edge_cut_particle, default_filters):
        rows = contour_measurements(
            contours_of(edge_cut_particle["image"]), "Au", 5, 200, 1.0, (0, 0, 255),
            None, [], True, filter_settings=default_filters,
            image_shape=edge_cut_particle["image"].shape,
        )
        assert rows == []

    def test_edge_particle_is_kept_and_flagged_when_included(self, edge_cut_particle, default_filters):
        rows = contour_measurements(
            contours_of(edge_cut_particle["image"]), "Au", 5, 200, 1.0, (0, 0, 255),
            None, [], False, filter_settings=default_filters,
            image_shape=edge_cut_particle["image"].shape,
        )
        assert len(rows) == 1
        assert "edge" in rows[0]["flags"]

    def test_ignored_regions_suppress_the_scale_bar(self, single_particle, default_filters):
        center = single_particle["center"]
        radius = single_particle["radius_px"]
        region = (center[0] - radius, center[1] - radius, 2 * radius, 2 * radius)
        rows = contour_measurements(
            contours_of(single_particle["image"]), "Au", 5, 200, 1.0, (0, 0, 255),
            None, [region], True, filter_settings=default_filters,
            image_shape=single_particle["image"].shape,
        )
        assert rows == []

    def test_requires_an_overlay_or_an_explicit_shape(self, single_particle, default_filters):
        with pytest.raises(ValueError, match="image_shape"):
            contour_measurements(
                contours_of(single_particle["image"]), "Au", 5, 200, 1.0, (0, 0, 255),
                None, [], True, filter_settings=default_filters,
            )

    def test_draws_on_the_overlay_when_one_is_given(self, single_particle, default_filters):
        overlay = single_particle["image"].copy()
        before = overlay.copy()
        contour_measurements(
            contours_of(single_particle["image"]), "Au", 5, 200, 1.0, (0, 0, 255),
            overlay, [], True, filter_settings=default_filters,
        )
        assert not np.array_equal(before, overlay)

    def test_counts_every_particle_in_a_noisy_field(self, noisy_particles, default_filters):
        rows = contour_measurements(
            contours_of(noisy_particles["image"]), "Au", 5, 200, 1.0, (0, 0, 255),
            None, [], True, filter_settings=default_filters,
            image_shape=noisy_particles["image"].shape,
        )
        assert len(rows) == noisy_particles["count"]
