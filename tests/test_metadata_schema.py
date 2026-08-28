"""Metadata schema, provenance columns and licence gating.

A dataset without provenance is not redistributable, so the schema contract is
tested as strictly as the measurement code.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "corpus_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from common import (  # noqa: E402
    CALIBRATION_FIELDS,
    IMAGE_FIELDS,
    LICENSE_PATTERNS,
    SOURCE_FIELDS,
    is_allowed_license,
    license_status_for,
)

SCHEMA_PATH = PIPELINE_DIR / "metadata_schema.json"


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


class TestSchemaFile:
    def test_schema_file_exists_and_is_valid_json(self, schema):
        assert isinstance(schema, dict)

    def test_schema_declares_sources_and_images(self, schema):
        assert "sources" in schema
        assert "images" in schema

    def test_each_section_declares_publication_requirements(self, schema):
        for section in ("sources", "images"):
            assert schema[section]["required_for_publication"]

    def test_images_declare_separate_metrology_requirements(self, schema):
        # An image can be publishable without being measurable; the schema
        # keeps those two bars apart on purpose.
        required = schema["images"]["required_for_metrology"]
        assert "nm_per_px" in required
        assert "scale_status" in required


class TestSchemaMatchesTheCsvColumns:
    def test_every_required_source_field_is_a_real_column(self, schema):
        missing = [field for field in schema["sources"]["required_for_publication"]
                   if field not in SOURCE_FIELDS]
        assert missing == []

    def test_every_required_image_field_is_a_real_column(self, schema):
        missing = [field for field in schema["images"]["required_for_publication"]
                   if field not in IMAGE_FIELDS]
        assert missing == []

    def test_every_metrology_field_is_a_real_column(self, schema):
        missing = [field for field in schema["images"]["required_for_metrology"]
                   if field not in IMAGE_FIELDS]
        assert missing == []

    def test_documented_field_descriptions_refer_to_real_columns(self, schema):
        for section, columns in (("sources", SOURCE_FIELDS), ("images", IMAGE_FIELDS)):
            unknown = [field for field in schema[section]["fields"] if field not in columns]
            assert unknown == [], f"{section} documents columns that do not exist: {unknown}"

    def test_provenance_columns_are_present(self):
        for field in ("source_id", "source_url", "license", "file_sha256"):
            assert field in IMAGE_FIELDS

    def test_calibration_state_is_tracked_per_image(self):
        for field in ("nm_per_px", "scale_status"):
            assert field in IMAGE_FIELDS
        for field in ("nm_per_px", "scale_nm", "scale_px", "method"):
            assert field in CALIBRATION_FIELDS

    def test_review_state_is_tracked_per_image(self):
        for field in ("curation_status", "quality_status", "metadata_status"):
            assert field in IMAGE_FIELDS

    def test_split_is_recorded_on_the_image_row(self):
        # Split lives with the image so it survives re-export.
        assert "split" in IMAGE_FIELDS

    def test_field_lists_have_no_duplicates(self):
        for fields in (SOURCE_FIELDS, IMAGE_FIELDS, CALIBRATION_FIELDS):
            assert len(fields) == len(set(fields))


class TestLicenceGating:
    @pytest.mark.parametrize("text", ["CC BY 4.0", "cc-by-sa", "CC0 1.0",
                                      "Public Domain", "Creative Commons Attribution"])
    def test_redistributable_licences_are_accepted(self, text):
        assert is_allowed_license(text)
        assert license_status_for(text) == "accepted"

    @pytest.mark.parametrize("text", ["CC BY-NC 4.0", "CC BY-ND 4.0",
                                      "noncommercial use only", "No Derivatives"])
    def test_restricted_licences_are_rejected(self, text):
        assert not is_allowed_license(text)
        assert license_status_for(text) == "rejected_for_public_corpus"

    @pytest.mark.parametrize("text", ["", None])
    def test_a_missing_licence_is_missing_not_accepted(self, text):
        assert license_status_for(text) == "missing"

    def test_all_rights_reserved_is_not_silently_accepted(self):
        assert license_status_for("All rights reserved") == "rejected_for_public_corpus"

    def test_every_advertised_pattern_actually_matches(self):
        for pattern in LICENSE_PATTERNS:
            assert is_allowed_license(pattern), pattern


class TestDemoDatasetMetadata:
    """The shipped demo dataset must satisfy the schema it advertises."""

    DEMO_DIR = REPO_ROOT / "Examples" / "demo_dataset"

    def test_demo_dataset_exists(self):
        assert self.DEMO_DIR.is_dir(), "Examples/demo_dataset is part of the release"

    def test_metadata_csv_declares_provenance_for_every_image(self):
        metadata_path = self.DEMO_DIR / "metadata.csv"
        assert metadata_path.exists()
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, "demo metadata.csv must not be empty"
        for row in rows:
            for field in ("image_id", "file_path", "license", "license_status",
                          "redistribution", "provenance", "nm_per_px", "scale_nm", "scale_px"):
                assert row.get(field), f"{row.get('image_id')} is missing {field}"

    def test_every_declared_demo_image_is_present(self):
        with (self.DEMO_DIR / "metadata.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                assert (self.DEMO_DIR / row["file_path"]).exists(), row["file_path"]

    def test_every_demo_image_is_redistributable(self):
        with (self.DEMO_DIR / "metadata.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                assert row["license_status"] == "accepted", row["image_id"]
                assert row["redistribution"] == "allowed", row["image_id"]

    def test_declared_calibration_is_self_consistent(self):
        with (self.DEMO_DIR / "metadata.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                expected = float(row["scale_nm"]) / float(row["scale_px"])
                assert float(row["nm_per_px"]) == pytest.approx(expected, rel=1e-6), row["image_id"]

    def test_checksums_match_the_shipped_files(self):
        from corpus.io import file_sha256

        with (self.DEMO_DIR / "metadata.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not row.get("file_sha256"):
                    continue
                assert file_sha256(self.DEMO_DIR / row["file_path"]) == row["file_sha256"], row["image_id"]

    def test_demo_dataset_has_a_readme(self):
        assert (self.DEMO_DIR / "README.md").exists()
