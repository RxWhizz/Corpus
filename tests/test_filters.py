"""Filter clamping, contrast strategies and the watershed default."""

import argparse

import numpy as np
import pytest

from corpus.measurement import (
    DEFAULT_FILTER_SETTINGS,
    clamp_filter_settings,
    contour_measurements,
    parse_bool,
    resolve_watershed,
)
from corpus.segmentation import CONTRAST_STRATEGIES, particle_binary, sio2_mask, to_gray


def namespace(**kwargs):
    base = dict(DEFAULT_FILTER_SETTINGS)
    base.update(kwargs)
    return argparse.Namespace(**base)


class TestParseBool:
    @pytest.mark.parametrize("value", [True, "true", "True", "1", "yes", "on", "anything"])
    def test_truthy(self, value):
        assert parse_bool(value) is True

    @pytest.mark.parametrize("value", [False, "false", "False", "0", "no", "off", " OFF "])
    def test_falsey(self, value):
        assert parse_bool(value) is False


class TestResolveWatershed:
    def test_auto_is_off_for_rods(self):
        assert resolve_watershed("auto", "pellets") is False

    @pytest.mark.parametrize("preset", ["generic", "spheres", "decorated"])
    def test_auto_is_on_elsewhere(self, preset):
        assert resolve_watershed("auto", preset) is True

    def test_explicit_value_overrides_the_preset(self):
        assert resolve_watershed("true", "pellets") is True
        assert resolve_watershed("false", "spheres") is False


class TestClampFilterSettings:
    def test_defaults_pass_through(self):
        settings = clamp_filter_settings(namespace())
        assert settings["min_circularity"] == 0.0
        assert settings["max_circularity"] == 1.0
        assert settings["include_holes"] is False

    def test_grey_thresholds_are_clamped_to_the_byte_range(self):
        settings = clamp_filter_settings(namespace(manual_threshold_min=-50, manual_threshold_max=900))
        assert settings["manual_threshold_min"] == 0
        assert settings["manual_threshold_max"] == 255

    def test_inverted_grey_range_is_swapped(self):
        settings = clamp_filter_settings(namespace(manual_threshold_min=200, manual_threshold_max=50))
        assert settings["manual_threshold_min"] == 50
        assert settings["manual_threshold_max"] == 200

    def test_circularity_is_clamped_to_the_unit_interval(self):
        settings = clamp_filter_settings(namespace(min_circularity=-1, max_circularity=5))
        assert settings["min_circularity"] == 0.0
        assert settings["max_circularity"] == 1.0

    def test_inverted_circularity_is_swapped(self):
        settings = clamp_filter_settings(namespace(min_circularity=0.9, max_circularity=0.2))
        assert settings["min_circularity"] == pytest.approx(0.2)
        assert settings["max_circularity"] == pytest.approx(0.9)

    def test_elongation_has_a_floor_of_one(self):
        settings = clamp_filter_settings(namespace(min_elongation=0.1, max_elongation=0.5))
        assert settings["min_elongation"] >= 1.0
        assert settings["max_elongation"] >= 1.0

    def test_accepts_a_plain_dict(self):
        settings = clamp_filter_settings({"min_circularity": 0.3, "max_circularity": 0.8})
        assert settings["min_circularity"] == pytest.approx(0.3)
        assert settings["max_circularity"] == pytest.approx(0.8)

    def test_output_is_json_serialisable_and_complete(self):
        settings = clamp_filter_settings(namespace())
        assert set(settings) == set(DEFAULT_FILTER_SETTINGS)

    def test_clamping_is_idempotent(self):
        once = clamp_filter_settings(namespace(min_circularity=-3, max_elongation=1e9))
        twice = clamp_filter_settings(once)
        assert once == twice


class TestContrastStrategies:
    def test_dark_particles_selects_the_particle(self, single_particle):
        binary = particle_binary(to_gray(single_particle["image"]), {"contrast_strategy": "dark_particles"})
        assert binary[200, 200] == 255
        assert binary[5, 5] == 0

    def test_bright_shells_selects_the_background(self, single_particle):
        binary = particle_binary(to_gray(single_particle["image"]), {"contrast_strategy": "bright_shells"})
        assert binary[200, 200] == 0
        assert binary[5, 5] == 255

    def test_manual_gray_range_selects_only_the_window(self, core_shell_particle):
        gray = to_gray(core_shell_particle["image"])
        binary = particle_binary(gray, {
            "contrast_strategy": "manual_gray_range",
            "manual_threshold_min": 130,
            "manual_threshold_max": 170,
        })
        # the shell ring is inside the window; the dark core is not
        assert binary[200, 200 - 35] == 255
        assert binary[200, 200] == 0

    def test_every_advertised_strategy_returns_a_mask(self, single_particle):
        gray = to_gray(single_particle["image"])
        for strategy in CONTRAST_STRATEGIES:
            binary = particle_binary(gray, {"contrast_strategy": strategy})
            assert binary.shape == gray.shape
            assert binary.dtype == np.uint8

    def test_include_holes_uses_a_gentler_closing(self, core_shell_particle):
        filled = sio2_mask(core_shell_particle["image"], include_holes=False)
        holed = sio2_mask(core_shell_particle["image"], include_holes=True)
        assert filled.shape == holed.shape
        assert filled.sum() >= holed.sum()


class TestShapeFilters:
    def test_a_narrow_circularity_window_rejects_a_rod(self, elongated_particle):
        import cv2

        from corpus.segmentation import particle_binary as pb

        binary = pb(to_gray(elongated_particle["image"]), {})
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rows = contour_measurements(
            contours, "Au", 5, 300, 1.0, (0, 0, 255), None, [], True,
            filter_settings={"min_circularity": 0.9, "max_circularity": 1.0,
                             "min_elongation": 1, "max_elongation": 999},
            image_shape=elongated_particle["image"].shape,
        )
        assert rows == []

    def test_an_elongation_floor_rejects_a_disc(self, single_particle):
        import cv2

        binary = particle_binary(to_gray(single_particle["image"]), {})
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rows = contour_measurements(
            contours, "Au", 5, 300, 1.0, (0, 0, 255), None, [], True,
            filter_settings={"min_circularity": 0, "max_circularity": 1,
                             "min_elongation": 2.5, "max_elongation": 999},
            image_shape=single_particle["image"].shape,
        )
        assert rows == []

    def test_a_radius_window_rejects_an_oversized_particle(self, single_particle):
        import cv2

        binary = particle_binary(to_gray(single_particle["image"]), {})
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rows = contour_measurements(
            contours, "Au", 1, 5, 1.0, (0, 0, 255), None, [], True,
            filter_settings={}, image_shape=single_particle["image"].shape,
        )
        assert rows == []
