"""Core-shell metrology: the shell-thickness invariant and its guard rails."""

import pytest

from corpus.metrology import (
    RATIO_MAX,
    RATIO_MIN,
    core_outer_ratio,
    is_ratio_outlier,
    object_row,
    shell_thickness,
)


def measurement(major_axis, minor_axis=None, center=(100.0, 100.0), flags=None, equivalent=None):
    """Minimal particle record of the shape ``object_row`` consumes."""
    minor_axis = major_axis if minor_axis is None else minor_axis
    return {
        "class": "test",
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "diameter": major_axis,
        "equivalent_diameter": equivalent if equivalent is not None else major_axis,
        "center_x": center[0],
        "center_y": center[1],
        "radius_px": major_axis / 2,
        "flags": list(flags or []),
        "backend": "classical",
    }


class TestShellThickness:
    @pytest.mark.parametrize(
        "outer,core,expected",
        [(100.0, 60.0, 20.0), (50.0, 30.0, 10.0), (12.5, 7.5, 2.5), (80.0, 80.0, 0.0)],
    )
    def test_matches_the_defining_formula(self, outer, core, expected):
        assert shell_thickness(outer, core) == pytest.approx(expected)

    def test_holds_for_the_synthetic_particle(self, core_shell_particle):
        fixture = core_shell_particle
        assert shell_thickness(
            fixture["outer_diameter_px"], fixture["core_diameter_px"]
        ) == pytest.approx(fixture["shell_thickness_px"])

    def test_is_calibration_equivariant(self):
        # Converting to nm before or after the subtraction must agree.
        nm_per_px = 0.37
        outer_px, core_px = 140.0, 62.0
        assert shell_thickness(outer_px * nm_per_px, core_px * nm_per_px) == pytest.approx(
            shell_thickness(outer_px, core_px) * nm_per_px
        )

    def test_inverted_pair_yields_zero_not_a_negative_shell(self):
        assert shell_thickness(40.0, 90.0) == 0.0

    def test_missing_half_yields_zero(self):
        assert shell_thickness(None, 50.0) == 0.0
        assert shell_thickness(50.0, None) == 0.0


class TestRatioChecks:
    def test_ratio_is_core_over_outer(self):
        assert core_outer_ratio(40.0, 100.0) == pytest.approx(0.4)

    def test_unknown_outer_gives_zero(self):
        assert core_outer_ratio(40.0, 0) == 0.0

    @pytest.mark.parametrize("ratio", [RATIO_MIN, 0.5, RATIO_MAX])
    def test_plausible_ratios_are_not_outliers(self, ratio):
        assert not is_ratio_outlier(ratio)

    @pytest.mark.parametrize("ratio", [0.05, 0.24, 0.9, 0.99])
    def test_implausible_ratios_are_outliers(self, ratio):
        assert is_ratio_outlier(ratio)


class TestObjectRow:
    def test_paired_object_reports_shell_and_ratio(self):
        row = object_row(1, "spheres", measurement(40.0), measurement(100.0))
        assert row["pair_status"] == "paired"
        assert row["shell_thickness_estimate"] == pytest.approx(30.0)
        assert row["inner_outer_ratio"] == pytest.approx(0.4)
        assert row["review_status"] == "ready"
        assert row["confidence_score"] == 1.0

    def test_object_id_is_zero_padded_and_stable(self):
        assert object_row(7, "spheres", measurement(40.0), measurement(100.0))["object_id"] == "obj_0007"
        assert object_row(1234, "spheres", measurement(40.0), measurement(100.0))["object_id"] == "obj_1234"

    def test_missing_core_is_flagged_not_dropped(self):
        row = object_row(1, "spheres", None, measurement(100.0))
        assert row["pair_status"] == "partial"
        assert "unpaired_inner" in row["flags"]
        assert row["review_status"] == "needs_review"

    def test_missing_shell_is_flagged_not_dropped(self):
        row = object_row(1, "spheres", measurement(40.0), None)
        assert row["pair_status"] == "partial"
        assert "unpaired_outer" in row["flags"]

    def test_implausible_pairing_is_flagged_for_review(self):
        row = object_row(1, "spheres", measurement(95.0), measurement(100.0))
        assert "ratio_outlier" in row["flags"]
        assert row["review_status"] == "needs_review"

    def test_edge_flag_propagates_from_the_particle(self):
        row = object_row(1, "spheres", measurement(40.0), measurement(100.0, flags=["edge"]))
        assert "edge" in row["flags"]
        assert row["confidence_score"] == pytest.approx(0.8)
        assert row["review_status"] == "needs_review"

    def test_watershed_provenance_is_recorded_without_blocking_review(self):
        row = object_row(1, "spheres", measurement(40.0, flags=["watershed_split"]), measurement(100.0))
        assert row["separation_method"] == "watershed"
        # watershed_split describes how, not whether to trust -- stays ready
        assert row["review_status"] == "ready"

    def test_flags_are_sorted_and_deduplicated(self):
        row = object_row(1, "spheres", measurement(40.0, flags=["edge"]), measurement(100.0, flags=["edge"]))
        assert row["flags"] == sorted(set(row["flags"]))

    def test_backend_is_recorded_on_every_object(self):
        row = object_row(1, "spheres", measurement(40.0), measurement(100.0))
        assert row["backend"] == "classical"

    def test_penalties_accumulate(self):
        row = object_row(1, "spheres", None, measurement(100.0, flags=["edge"]))
        # unpaired (-0.3) and edge (-0.2)
        assert row["confidence_score"] == pytest.approx(0.5)
        assert "low_confidence" in row["flags"]

    def test_confidence_never_goes_negative(self):
        inner = measurement(95.0, flags=["edge", "low_split_confidence"])
        row = object_row(1, "spheres", inner, None)
        assert row["confidence_score"] >= 0.0
