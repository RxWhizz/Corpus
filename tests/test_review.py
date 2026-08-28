"""Confidence scoring, review gating and image-level warnings."""

import pytest

from corpus.review import (
    CONFIDENCE_PENALTIES,
    READY_THRESHOLD,
    build_warnings,
    confidence_score,
    draw_review_labels,
    draw_review_markers,
    review_status,
)


class TestConfidenceScore:
    def test_a_clean_object_scores_one(self):
        assert confidence_score([]) == 1.0

    @pytest.mark.parametrize(
        "flag,expected",
        [("unpaired_inner", 0.7), ("unpaired_outer", 0.7), ("edge", 0.8),
         ("ratio_outlier", 0.8), ("low_split_confidence", 0.8)],
    )
    def test_each_penalty_is_applied(self, flag, expected):
        assert confidence_score([flag]) == pytest.approx(expected)

    def test_both_unpaired_flags_are_penalised_once(self):
        # An object missing both halves is one problem, not two.
        assert confidence_score(["unpaired_inner", "unpaired_outer"]) == pytest.approx(0.7)

    def test_penalties_from_different_groups_accumulate(self):
        assert confidence_score(["unpaired_inner", "edge"]) == pytest.approx(0.5)

    def test_score_is_clamped_at_zero(self):
        every_flag = [flag for group, _ in CONFIDENCE_PENALTIES for flag in group]
        assert confidence_score(every_flag) >= 0.0

    def test_unknown_flags_do_not_change_the_score(self):
        assert confidence_score(["some_future_flag"]) == 1.0

    def test_order_does_not_matter(self):
        assert confidence_score(["edge", "ratio_outlier"]) == confidence_score(["ratio_outlier", "edge"])


class TestReviewStatus:
    def test_clean_and_confident_is_ready(self):
        assert review_status(1.0, []) == "ready"

    def test_low_confidence_needs_review(self):
        assert review_status(READY_THRESHOLD - 0.01, []) == "needs_review"

    def test_any_review_flag_forces_review_even_at_full_confidence(self):
        assert review_status(1.0, ["edge"]) == "needs_review"

    def test_watershed_provenance_alone_does_not_force_review(self):
        assert review_status(1.0, ["watershed_split"]) == "ready"

    def test_exactly_at_the_threshold_is_ready(self):
        assert review_status(READY_THRESHOLD, []) == "ready"


class TestWarnings:
    def test_low_contrast_is_reported(self, low_contrast_image):
        warnings = build_warnings(low_contrast_image["image"], [], {"method": "manual_line"})
        assert "Low contrast" in warnings

    def test_good_contrast_is_not_reported(self, single_particle):
        warnings = build_warnings(single_particle["image"], [], {"method": "manual_line"})
        assert "Low contrast" not in warnings

    def test_auto_detected_scale_is_flagged(self, single_particle):
        warnings = build_warnings(single_particle["image"], [], {"method": "bright_contour"})
        assert any("auto-detected" in warning for warning in warnings)

    @pytest.mark.parametrize("method", ["manual_line", "manual_override"])
    def test_manual_scale_is_not_flagged(self, single_particle, method):
        warnings = build_warnings(single_particle["image"], [], {"method": method})
        assert not any("auto-detected" in warning for warning in warnings)

    def test_too_many_edge_objects_is_flagged(self, single_particle):
        objects = [{"flags": ["edge"], "pair_status": "paired"} for _ in range(4)]
        objects += [{"flags": [], "pair_status": "paired"} for _ in range(4)]
        warnings = build_warnings(single_particle["image"], objects, {"method": "manual_line"})
        assert "Too many edge objects" in warnings

    def test_many_unpaired_objects_is_flagged(self, single_particle):
        objects = [{"flags": [], "pair_status": "partial"} for _ in range(5)]
        objects += [{"flags": [], "pair_status": "paired"} for _ in range(5)]
        warnings = build_warnings(single_particle["image"], objects, {"method": "manual_line"})
        assert "Many unpaired objects" in warnings

    def test_a_clean_run_produces_no_warnings(self, single_particle):
        objects = [{"flags": [], "pair_status": "paired"} for _ in range(8)]
        assert build_warnings(single_particle["image"], objects, {"method": "manual_line"}) == []

    def test_no_objects_does_not_trigger_a_division_by_zero(self, single_particle):
        build_warnings(single_particle["image"], [], {"method": "manual_line"})


class TestOverlayHelpers:
    def test_drawing_helpers_tolerate_a_missing_overlay(self):
        # Head-less measurement passes overlay=None; these must be no-ops.
        draw_review_labels(None, [{"object_id": "obj_0001", "center_x": 1, "center_y": 1}])
        draw_review_markers(None, [{"review_status": "needs_review", "center_x": 1, "center_y": 1}])

    def test_review_markers_only_mark_unready_objects(self, single_particle):
        import numpy as np

        ready_overlay = single_particle["image"].copy()
        draw_review_markers(ready_overlay, [{"review_status": "ready", "center_x": 200, "center_y": 200}])
        assert np.array_equal(ready_overlay, single_particle["image"])

        flagged_overlay = single_particle["image"].copy()
        draw_review_markers(flagged_overlay, [{"review_status": "needs_review", "center_x": 200, "center_y": 200}])
        assert not np.array_equal(flagged_overlay, single_particle["image"])
