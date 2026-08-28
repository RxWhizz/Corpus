"""Calibration is the multiplier on every reported number, so it is pinned hard."""


import pytest

from corpus.calibration import (
    collect_scale_candidates,
    nm_per_pixel,
    parse_scale_line,
    resolve_scale,
    scale_bar_confidence,
)
from corpus.errors import CalibrationError


class TestNmPerPixel:
    def test_matches_the_defining_ratio(self):
        assert nm_per_pixel(200.0, 100.0) == pytest.approx(2.0)

    @pytest.mark.parametrize(
        "scale_nm,scale_px,expected",
        [(100, 250, 0.4), (500, 125, 4.0), (1000, 1000, 1.0), (50, 400, 0.125)],
    )
    def test_known_pairs(self, scale_nm, scale_px, expected):
        assert nm_per_pixel(scale_nm, scale_px) == pytest.approx(expected)

    def test_is_deterministic_across_calls(self):
        values = {nm_per_pixel(237.0, 91.0) for _ in range(50)}
        assert len(values) == 1

    def test_scales_linearly_with_bar_length(self):
        assert nm_per_pixel(200, 50) == pytest.approx(2 * nm_per_pixel(200, 100))

    @pytest.mark.parametrize("scale_nm", [0, -10, float("nan"), float("inf")])
    def test_rejects_bad_scale_length(self, scale_nm):
        with pytest.raises(CalibrationError):
            nm_per_pixel(scale_nm, 100)

    @pytest.mark.parametrize("scale_px", [0, -5, float("nan")])
    def test_rejects_bad_pixel_length(self, scale_px):
        with pytest.raises(CalibrationError):
            nm_per_pixel(200, scale_px)

    def test_round_trips_a_measured_length(self):
        # A feature spanning 37 px under a 100 nm / 80 px bar must read back
        # as 46.25 nm regardless of how the ratio is applied.
        ratio = nm_per_pixel(100, 80)
        assert 37 * ratio == pytest.approx(46.25)


class TestParseScaleLine:
    def test_empty_means_no_manual_line(self):
        assert parse_scale_line("") is None
        assert parse_scale_line(None) is None

    def test_length_is_euclidean(self):
        line = parse_scale_line("0,0,3,4")
        assert line["length"] == pytest.approx(5.0)

    def test_accepts_a_dict(self):
        line = parse_scale_line({"x1": 0, "y1": 0, "x2": 0, "y2": 10})
        assert line["length"] == pytest.approx(10.0)

    def test_rejects_wrong_arity(self):
        with pytest.raises(CalibrationError):
            parse_scale_line("1,2,3")

    def test_rejects_non_numeric(self):
        with pytest.raises(CalibrationError):
            parse_scale_line("a,b,c,d")

    def test_rejects_zero_length(self):
        with pytest.raises(CalibrationError):
            parse_scale_line("10,10,10,10")


class TestScaleBarConfidence:
    def test_stays_in_unit_range(self, scale_bar_image):
        bar = scale_bar_image["bar"]
        score = scale_bar_confidence(bar["x"], bar["y"], bar["width"], bar["height"], 600, 400)
        assert 0.0 <= score <= 1.0

    def test_prefers_lower_right_over_upper_left(self):
        lower_right = scale_bar_confidence(420, 360, 120, 6, 600, 400)
        upper_left = scale_bar_confidence(10, 10, 120, 6, 600, 400)
        assert lower_right > upper_left

    def test_penalises_bars_flush_with_the_bottom_edge(self):
        inside = scale_bar_confidence(420, 380, 120, 6, 600, 400)
        flush = scale_bar_confidence(420, 395, 120, 6, 600, 400)
        assert flush < inside


class TestResolveScale:
    def test_detects_a_printed_bar(self, scale_bar_image):
        candidates = collect_scale_candidates(scale_bar_image["image"])
        assert candidates, "expected at least one scale-bar candidate"
        best = candidates[0]
        assert best["width_px"] == pytest.approx(scale_bar_image["bar"]["width"], abs=3)

    def test_auto_detection_recovers_the_known_ratio(self, scale_bar_image):
        nm_per_px, selected, _, ignored = resolve_scale(
            scale_bar_image["image"], scale_bar_image["scale_nm"]
        )
        assert nm_per_px == pytest.approx(scale_bar_image["expected_nm_per_px"], rel=0.05)
        assert selected["method"] in ("bright_contour", "hough_line")
        assert ignored, "the detected bar must be masked out of particle detection"

    def test_manual_line_wins_over_detection(self, scale_bar_image):
        line = parse_scale_line("100,300,200,300")
        nm_per_px, selected, _, _ = resolve_scale(
            scale_bar_image["image"], 500.0, 0, line
        )
        assert selected["method"] == "manual_line"
        assert selected["confidence"] == 1.0
        assert nm_per_px == pytest.approx(5.0)

    def test_manual_pixel_override_is_used_verbatim(self, scale_bar_image):
        nm_per_px, selected, _, ignored = resolve_scale(
            scale_bar_image["image"], 250.0, 125.0
        )
        assert selected["method"] == "manual_override"
        assert nm_per_px == pytest.approx(2.0)
        assert ignored == []

    def test_raises_when_no_bar_and_no_override(self, blank_image):
        with pytest.raises(CalibrationError, match="scale bar"):
            resolve_scale(blank_image, 100.0)

    def test_is_reproducible_for_the_same_input(self, scale_bar_image):
        first = resolve_scale(scale_bar_image["image"], 240.0)[0]
        second = resolve_scale(scale_bar_image["image"], 240.0)[0]
        assert first == second

    def test_manual_line_masks_the_bar_region(self, scale_bar_image):
        line = parse_scale_line("420,360,540,360")
        _, _, _, ignored = resolve_scale(scale_bar_image["image"], 240.0, 0, line)
        x, y, w, h = ignored[0]
        assert x <= 420 and y <= 360
        assert w >= 120

    def test_a_longer_bar_for_the_same_nm_gives_a_finer_scale(self, scale_bar_image):
        coarse = resolve_scale(scale_bar_image["image"], 240.0, 60.0)[0]
        fine = resolve_scale(scale_bar_image["image"], 240.0, 240.0)[0]
        assert fine < coarse
        assert coarse / fine == pytest.approx(4.0)
