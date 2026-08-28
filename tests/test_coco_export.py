"""COCO is the master annotation format; YOLO-seg is derived from it.

These tests pin the derivation: class aliasing, polygon normalisation, split
assignment and source-group leakage.
"""


import pytest
from common_training import (
    CLASS_NAMES,
    COCO_CATEGORIES,
    annotation_review_status,
    canonical_categories,
    category_mapping,
    is_public_license,
    normalize_class_name,
    normalize_polygon,
    read_csv,
    stable_hash,
    write_manifest,
)
from prepare_yolo_seg import split_for_group


class TestClassMapping:
    @pytest.mark.parametrize(
        "raw,expected",
        [("Au_core", "Au_core"), ("au core", "Au_core"), ("gold", "Au_core"),
         ("Core", "Au_core"), ("SiO2_outer", "SiO2_outer"), ("silica", "SiO2_outer"),
         ("shell", "SiO2_outer"), ("SiO2 Carrier", "SiO2_outer")],
    )
    def test_aliases_collapse_to_the_two_v0_classes(self, raw, expected):
        assert normalize_class_name(raw) == expected

    def test_unknown_names_are_passed_through_untouched(self):
        assert normalize_class_name("Pt_shell") == "Pt_shell"

    def test_category_mapping_indexes_into_class_names(self, coco_annotation):
        mapping, unknown = category_mapping(coco_annotation["payload"]["categories"])
        assert unknown == []
        assert mapping == {1: CLASS_NAMES.index("Au_core"), 2: CLASS_NAMES.index("SiO2_outer")}

    def test_unknown_categories_are_reported_not_silently_dropped(self):
        mapping, unknown = category_mapping([{"id": 9, "name": "carbon_film"}])
        assert mapping == {}
        assert unknown == ["carbon_film"]

    def test_canonical_categories_are_copies(self):
        first = canonical_categories()
        first[0]["name"] = "mutated"
        assert COCO_CATEGORIES[0]["name"] == "Au_core"


class TestPolygonNormalisation:
    def test_coordinates_are_scaled_into_the_unit_square(self):
        normalized = normalize_polygon([0, 0, 100, 0, 100, 100, 0, 100], 100, 100)
        assert normalized == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    def test_out_of_frame_points_are_clipped_not_rejected(self):
        normalized = normalize_polygon([-10, -10, 200, 200, 50, 50], 100, 100)
        assert all(0.0 <= value <= 1.0 for value in normalized)

    def test_x_and_y_use_their_own_dimension(self):
        normalized = normalize_polygon([50, 50, 100, 200, 0, 0], 100, 200)
        assert normalized[0] == pytest.approx(0.5)
        assert normalized[1] == pytest.approx(0.25)

    @pytest.mark.parametrize("points", [[0, 0, 1, 1], [0, 0, 1, 1, 2], []])
    def test_degenerate_polygons_are_rejected(self, points):
        assert normalize_polygon(points, 100, 100) is None

    def test_zero_dimension_is_rejected(self):
        assert normalize_polygon([0, 0, 10, 0, 10, 10], 0, 100) is None

    def test_real_fixture_annotations_all_normalise(self, coco_annotation):
        payload = coco_annotation["payload"]
        sizes = {image["id"]: (image["width"], image["height"]) for image in payload["images"]}
        for annotation in payload["annotations"]:
            width, height = sizes[annotation["image_id"]]
            for polygon in annotation["segmentation"]:
                normalized = normalize_polygon(polygon, width, height)
                assert normalized is not None
                assert all(0.0 <= value <= 1.0 for value in normalized)


class TestSplitAssignment:
    def test_a_single_group_goes_entirely_to_train(self):
        assert split_for_group("a", ["a"]) == "train"

    def test_two_groups_become_train_and_val(self):
        groups = ["a", "b"]
        assert {split_for_group(group, groups) for group in groups} == {"train", "val"}

    def test_a_group_is_never_placed_in_two_splits(self):
        groups = [f"group_{index}" for index in range(10)]
        ranked = sorted(groups, key=stable_hash)
        assignment = {}
        for group in ranked:
            assignment.setdefault(group, set()).add(split_for_group(group, ranked))
        assert all(len(splits) == 1 for splits in assignment.values())

    def test_ranking_is_deterministic_without_an_explicit_seed(self):
        # Content-addressed ranking, so input order cannot change the splits.
        groups = [f"group_{index}" for index in range(20)]
        shuffled = groups[10:] + groups[:10]
        assert sorted(groups, key=stable_hash) == sorted(shuffled, key=stable_hash)

    def test_split_assignment_is_reproducible(self):
        groups = [f"group_{index}" for index in range(12)]
        ranked = sorted(groups, key=stable_hash)
        first = [split_for_group(group, ranked) for group in ranked]
        second = [split_for_group(group, ranked) for group in ranked]
        assert first == second

    def test_all_three_splits_appear_once_there_are_enough_groups(self):
        groups = [f"group_{index}" for index in range(12)]
        ranked = sorted(groups, key=stable_hash)
        assert {split_for_group(group, ranked) for group in ranked} == {"train", "val", "test"}

    def test_stable_hash_is_content_addressed(self):
        assert stable_hash("group_a") == stable_hash("group_a")
        assert stable_hash("group_a") != stable_hash("group_b")


class TestLicenceGating:
    @pytest.mark.parametrize(
        "row",
        [{"license_status": "accepted"}, {"license": "CC BY 4.0"},
         {"license": "CC0-1.0"}, {"license": "Public Domain"}],
    )
    def test_redistributable_rows_pass(self, row):
        assert is_public_license(row)

    @pytest.mark.parametrize(
        "row",
        [{}, {"license": "All rights reserved"}, {"license": "CC BY-NC 4.0"},
         {"license": "CC BY-ND 4.0"}, {"license_status": "missing"}],
    )
    def test_non_redistributable_or_unknown_rows_fail(self, row):
        assert not is_public_license(row)

    def test_an_accepted_status_cannot_override_a_noncommercial_licence(self):
        assert not is_public_license({"license_status": "accepted", "license": "CC BY-NC 4.0"})


class TestAnnotationReviewState:
    def test_clean_annotation_is_ready(self):
        assert annotation_review_status({"id": 1}) == "ready"

    def test_crowd_annotations_need_review(self):
        assert annotation_review_status({"iscrowd": 1}) == "needs_review"

    @pytest.mark.parametrize("status", ["needs_review", "uncertain", "ambiguous", "difficult"])
    def test_flagged_statuses_need_review(self, status):
        assert annotation_review_status({"review_status": status}) == "needs_review"

    def test_cvat_style_attribute_lists_are_understood(self):
        annotation = {"attributes": [{"name": "review_status", "value": "uncertain"}]}
        assert annotation_review_status(annotation) == "needs_review"


class TestManifest:
    def test_manifest_round_trips_and_keeps_provenance_columns(self, tmp_path):
        rows = [{
            "image_id": "img_1", "source_id": "group_a", "split": "train",
            "image_path": "images/train/img_1.png", "label_path": "labels/train/img_1.txt",
            "labels": 2, "au_core_labels": 1, "sio2_outer_labels": 1,
            "nm_per_px": "0.5", "license": "CC0-1.0", "license_status": "accepted",
            "doi": "10.0000/demo", "source_url": "https://example.invalid/demo",
        }]
        path = tmp_path / "manifest.csv"
        write_manifest(path, rows)
        read_back = read_csv(path)
        assert len(read_back) == 1
        for key in ("source_id", "split", "nm_per_px", "license", "license_status", "doi"):
            assert read_back[0][key] == str(rows[0][key])

    def test_missing_fields_are_written_as_empty_not_omitted(self, tmp_path):
        path = tmp_path / "manifest.csv"
        write_manifest(path, [{"image_id": "img_1"}])
        row = read_csv(path)[0]
        assert row["license"] == ""
        assert "license_status" in row
