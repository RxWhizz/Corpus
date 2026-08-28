"""Watershed separation of touching particles."""

import cv2
import pytest

from corpus.measurement import contour_measurements
from corpus.segmentation import particle_binary, to_gray, watershed_split_contours


def binary_of(image, settings=None):
    return particle_binary(to_gray(image), settings or {})


class TestWatershedSplitContours:
    def test_splits_two_touching_discs(self, two_touching_particles):
        binary = binary_of(two_touching_particles["image"])
        plain, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(plain) == 1, "fixture must present as a single merged blob"

        split = watershed_split_contours(binary, 10, 100)
        assert len(split) == two_touching_particles["count"]

    def test_leaves_an_isolated_particle_alone(self, single_particle):
        binary = binary_of(single_particle["image"])
        assert len(watershed_split_contours(binary, 10, 100)) == 1

    def test_empty_mask_yields_no_contours(self, blank_image):
        assert watershed_split_contours(binary_of(blank_image), 10, 100) == []

    def test_never_returns_fewer_contours_than_the_plain_finder(self, noisy_particles):
        binary = binary_of(noisy_particles["image"])
        plain, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        split = watershed_split_contours(binary, 5, 100)
        assert len(split) >= len(plain)

    def test_is_deterministic(self, two_touching_particles):
        binary = binary_of(two_touching_particles["image"])
        first = [cv2.contourArea(c) for c in watershed_split_contours(binary, 10, 100)]
        second = [cv2.contourArea(c) for c in watershed_split_contours(binary, 10, 100)]
        assert first == second

    @pytest.mark.parametrize("factor", [0.55, 0.7, 0.85])
    def test_split_holds_at_or_above_the_default_seed_threshold(self, two_touching_particles, factor):
        binary = binary_of(two_touching_particles["image"])
        assert len(watershed_split_contours(binary, 10, 100, factor)) >= 2

    @pytest.mark.parametrize("factor", [0.28, 0.4, 0.5])
    def test_a_low_seed_threshold_merges_instead_of_splitting(self, two_touching_particles, factor):
        # A lower distance factor grows the seed region until the two cores
        # touch, so the pair is reported as one particle. This is why the
        # retry ladder in watershed_split_contours only ever *rescues* the
        # no-seed case; it cannot manufacture a split.
        binary = binary_of(two_touching_particles["image"])
        assert len(watershed_split_contours(binary, 10, 100, factor)) == 1

    def test_split_particles_measure_close_to_the_true_size(self, two_touching_particles, default_filters):
        binary = binary_of(two_touching_particles["image"])
        split = watershed_split_contours(binary, 10, 100)
        rows = contour_measurements(
            split, "Au", 5, 200, 1.0, (0, 0, 255), None, [], False,
            ["watershed_split"], "watershed", default_filters,
            image_shape=two_touching_particles["image"].shape,
        )
        assert len(rows) == 2
        expected = 2 * two_touching_particles["radius_px"]
        for row in rows:
            # Overlapping discs lose a lens-shaped sliver each, so allow 25%.
            assert row["diameter"] == pytest.approx(expected, rel=0.25)
            assert row["separation_method"] == "watershed"

    def test_split_measurements_carry_the_watershed_flag(self, two_touching_particles, default_filters):
        binary = binary_of(two_touching_particles["image"])
        rows = contour_measurements(
            watershed_split_contours(binary, 10, 100), "Au", 5, 200, 1.0, (0, 0, 255),
            None, [], False, ["watershed_split"], "watershed", default_filters,
            image_shape=two_touching_particles["image"].shape,
        )
        assert rows and all("watershed_split" in row["flags"] for row in rows)
