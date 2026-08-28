"""Agreement metrics, the validation table, and the one-command report."""

import csv
import json

import pytest

from corpus.errors import ValidationInputError
from corpus.validation import (
    agreement_metrics,
    bland_altman,
    build_report,
    count_agreement,
    load_validation_table,
    mean_absolute_error,
    mean_bias,
    mean_relative_error,
    pair_quantity,
    pearson_r,
    r_squared_identity,
    root_mean_square_error,
    schema_columns,
    write_report,
)
from corpus.validation.__main__ import main as validation_main

PERFECT = ([10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0])
OFFSET = ([10.0, 20.0, 30.0, 40.0], [12.0, 22.0, 32.0, 42.0])


class TestErrorMetrics:
    def test_perfect_agreement_is_all_zeros(self):
        reference, corpus = PERFECT
        assert mean_absolute_error(reference, corpus) == 0.0
        assert root_mean_square_error(reference, corpus) == 0.0
        assert mean_bias(reference, corpus) == 0.0
        assert mean_relative_error(reference, corpus) == 0.0

    def test_constant_offset_shows_up_as_bias(self):
        reference, corpus = OFFSET
        assert mean_bias(reference, corpus) == pytest.approx(2.0)
        assert mean_absolute_error(reference, corpus) == pytest.approx(2.0)
        assert root_mean_square_error(reference, corpus) == pytest.approx(2.0)

    def test_bias_sign_distinguishes_over_from_under_reporting(self):
        assert mean_bias([10.0, 10.0], [12.0, 12.0]) > 0
        assert mean_bias([10.0, 10.0], [8.0, 8.0]) < 0

    def test_mae_ignores_sign_where_bias_cancels(self):
        reference = [10.0, 10.0]
        corpus = [12.0, 8.0]
        assert mean_bias(reference, corpus) == pytest.approx(0.0)
        assert mean_absolute_error(reference, corpus) == pytest.approx(2.0)

    def test_rmse_is_at_least_mae(self):
        reference = [10.0, 20.0, 30.0]
        corpus = [11.0, 25.0, 30.0]
        assert root_mean_square_error(reference, corpus) >= mean_absolute_error(reference, corpus)

    def test_relative_error_is_a_fraction_of_the_reference(self):
        assert mean_relative_error([100.0, 200.0], [110.0, 220.0]) == pytest.approx(0.1)

    def test_relative_error_rejects_a_zero_reference(self):
        with pytest.raises(ValidationInputError):
            mean_relative_error([0.0, 10.0], [1.0, 10.0])

    def test_length_mismatch_is_an_error_not_a_truncation(self):
        with pytest.raises(ValidationInputError, match="same length"):
            mean_absolute_error([1.0, 2.0], [1.0])

    def test_empty_series_is_an_error(self):
        with pytest.raises(ValidationInputError, match="empty"):
            mean_absolute_error([], [])

    def test_non_finite_values_are_rejected(self):
        with pytest.raises(ValidationInputError, match="Non-finite"):
            mean_absolute_error([1.0, float("nan")], [1.0, 2.0])


class TestCorrelationMetrics:
    def test_identity_r_squared_is_one_for_perfect_agreement(self):
        assert r_squared_identity(*PERFECT) == pytest.approx(1.0)

    def test_identity_r_squared_penalises_a_pure_offset(self):
        # This is the point of using the identity line: a free-slope fit would
        # still report r^2 = 1.0 for a constant +2 nm offset.
        assert r_squared_identity(*OFFSET) < 1.0
        assert pearson_r(*OFFSET) == pytest.approx(1.0)

    def test_identity_r_squared_can_go_negative(self):
        # Worse than predicting the reference mean.
        assert r_squared_identity([10.0, 20.0, 30.0], [30.0, 10.0, 20.0]) < 0

    def test_r_squared_undefined_for_a_constant_reference(self):
        with pytest.raises(ValidationInputError, match="identical"):
            r_squared_identity([10.0, 10.0, 10.0], [11.0, 9.0, 10.0])

    def test_pearson_undefined_without_variance(self):
        with pytest.raises(ValidationInputError, match="variance"):
            pearson_r([10.0, 10.0], [11.0, 12.0])


class TestBlandAltman:
    def test_perfect_agreement_has_zero_width_limits(self):
        stats = bland_altman(*PERFECT)
        assert stats["mean_difference"] == 0.0
        assert stats["lower_limit"] == 0.0
        assert stats["upper_limit"] == 0.0

    def test_offset_moves_both_limits(self):
        stats = bland_altman(*OFFSET)
        assert stats["mean_difference"] == pytest.approx(2.0)
        assert stats["lower_limit"] == pytest.approx(2.0)
        assert stats["upper_limit"] == pytest.approx(2.0)

    def test_limits_bracket_the_mean_difference(self):
        stats = bland_altman([10.0, 20.0, 30.0, 40.0], [11.0, 24.0, 29.0, 44.0])
        assert stats["lower_limit"] < stats["mean_difference"] < stats["upper_limit"]

    def test_means_and_differences_align_with_the_input(self):
        stats = bland_altman([10.0, 20.0], [20.0, 30.0])
        assert stats["means"] == [15.0, 25.0]
        assert stats["differences"] == [10.0, 10.0]

    def test_uses_the_sample_standard_deviation(self):
        stats = bland_altman([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        # differences 1,2,3 -> sample SD = 1.0
        assert stats["std_difference"] == pytest.approx(1.0)


class TestCountAgreement:
    def test_full_agreement_gives_unit_recall_and_precision(self):
        counts = count_agreement(10, 10, 10)
        assert counts["recall"] == 1.0
        assert counts["precision"] == 1.0
        assert counts["missed_by_corpus"] == 0
        assert counts["extra_in_corpus"] == 0

    def test_missed_particles_lower_recall(self):
        counts = count_agreement(10, 6, 6)
        assert counts["recall"] == pytest.approx(0.6)
        assert counts["missed_by_corpus"] == 4

    def test_extra_detections_lower_precision(self):
        counts = count_agreement(10, 15, 10)
        assert counts["precision"] == pytest.approx(10 / 15)
        assert counts["extra_in_corpus"] == 5

    def test_impossible_match_count_is_rejected(self):
        with pytest.raises(ValidationInputError, match="cannot exceed"):
            count_agreement(5, 5, 9)


class TestAgreementMetricsBlock:
    def test_reports_every_documented_field(self):
        metrics = agreement_metrics(*OFFSET, label="diameter_nm")
        for key in ("quantity", "n", "mae", "rmse", "mean_bias", "mean_relative_error",
                    "r_squared_identity", "pearson_r", "bland_altman",
                    "mae_percent_of_reference_mean", "bias_percent_of_reference_mean"):
            assert key in metrics

    def test_percentages_are_relative_to_the_reference_mean(self):
        metrics = agreement_metrics([100.0, 100.0, 100.0, 100.0], [110.0, 110.0, 110.0, 110.0])
        assert metrics["mae_percent_of_reference_mean"] == pytest.approx(10.0)
        assert metrics["bias_percent_of_reference_mean"] == pytest.approx(10.0)

    def test_undefined_metrics_are_reported_as_none_with_a_reason(self):
        metrics = agreement_metrics([10.0, 10.0, 10.0], [11.0, 11.0, 11.0])
        assert metrics["r_squared_identity"] is None
        assert "r_squared_identity" in metrics["undefined"]
        # the well-defined metrics still come through
        assert metrics["mae"] == pytest.approx(1.0)

    def test_is_json_serialisable(self):
        json.dumps(agreement_metrics(*OFFSET, label="diameter_nm"))


def write_table(path, rows, columns=None):
    columns = columns or ["image_id", "particle_id",
                          "reference_diameter_nm", "corpus_diameter_nm",
                          "reference_core_nm", "corpus_core_nm",
                          "reference_outer_nm", "corpus_outer_nm",
                          "reference_shell_nm", "corpus_shell_nm",
                          "morphology", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


@pytest.fixture
def validation_table(tmp_path):
    rows = []
    for index in range(1, 13):
        reference_outer = 80.0 + 4 * index
        reference_core = 30.0 + 2 * index
        rows.append({
            "image_id": f"img_{(index - 1) // 4 + 1}",
            "particle_id": f"p_{index}",
            "reference_diameter_nm": reference_outer,
            "corpus_diameter_nm": reference_outer + (1.5 if index % 2 else -1.0),
            "reference_core_nm": reference_core,
            "corpus_core_nm": reference_core + 0.5,
            "reference_outer_nm": reference_outer,
            "corpus_outer_nm": reference_outer + 1.0,
            "reference_shell_nm": (reference_outer - reference_core) / 2,
            "corpus_shell_nm": (reference_outer - reference_core) / 2 + 0.25,
            "morphology": "core_shell_sphere" if index % 2 else "core_shell_rod",
        })
    # a particle the reference found but Corpus missed
    rows.append({"image_id": "img_4", "particle_id": "p_missed",
                 "reference_diameter_nm": 120.0, "morphology": "core_shell_sphere"})
    # a particle a human flagged as ambiguous
    rows.append({"image_id": "img_4", "particle_id": "p_ambiguous",
                 "reference_diameter_nm": 95.0, "corpus_diameter_nm": 150.0,
                 "morphology": "core_shell_sphere", "status": "ambiguous"})
    return write_table(tmp_path / "table.csv", rows)


class TestValidationTable:
    def test_schema_matches_the_epic_columns(self):
        columns = schema_columns()
        for column in ("image_id", "particle_id", "reference_diameter_nm", "corpus_diameter_nm",
                       "reference_core_nm", "corpus_core_nm", "reference_outer_nm",
                       "corpus_outer_nm", "reference_shell_nm", "corpus_shell_nm"):
            assert column in columns

    def test_loads_rows_and_detected_quantities(self, validation_table):
        rows, quantities = load_validation_table(validation_table)
        assert len(rows) == 14
        assert set(quantities) == {"diameter_nm", "core_nm", "outer_nm", "shell_nm"}

    def test_blank_cells_become_none_not_zero(self, validation_table):
        rows, _ = load_validation_table(validation_table)
        missed = next(row for row in rows if row["particle_id"] == "p_missed")
        assert missed["corpus_diameter_nm"] is None
        assert missed["reference_diameter_nm"] == pytest.approx(120.0)

    def test_missing_identity_columns_are_rejected(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValidationInputError, match="missing required columns"):
            load_validation_table(path)

    def test_a_table_with_no_comparable_pair_is_rejected(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("image_id,particle_id\nimg,p\n", encoding="utf-8")
        with pytest.raises(ValidationInputError, match="no comparable quantity pairs"):
            load_validation_table(path)

    def test_unparseable_numbers_name_the_line(self, tmp_path):
        path = write_table(tmp_path / "bad.csv", [
            {"image_id": "i", "particle_id": "p",
             "reference_diameter_nm": "not_a_number", "corpus_diameter_nm": "10"},
        ])
        with pytest.raises(ValidationInputError, match=r":2:"):
            load_validation_table(path)

    def test_a_header_only_table_is_rejected(self, tmp_path):
        path = write_table(tmp_path / "empty.csv", [])
        with pytest.raises(ValidationInputError, match="no data rows"):
            load_validation_table(path)

    def test_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(ValidationInputError, match="not found"):
            load_validation_table(tmp_path / "nope.csv")

    def test_unmatched_and_flagged_rows_are_separated_not_dropped(self, validation_table):
        rows, _ = load_validation_table(validation_table)
        reference, corpus, unmatched = pair_quantity(rows, "diameter_nm")
        assert len(reference) == len(corpus) == 12
        reasons = sorted(item["reason"] for item in unmatched)
        assert reasons == ["corpus_missed_particle", "flagged_ambiguous"]

    def test_a_flagged_outlier_does_not_inflate_the_error(self, validation_table):
        rows, _ = load_validation_table(validation_table)
        reference, corpus, _ = pair_quantity(rows, "diameter_nm")
        assert mean_absolute_error(reference, corpus) < 2.0


class TestReport:
    def test_report_covers_every_quantity(self, validation_table):
        rows, quantities = load_validation_table(validation_table)
        report = build_report(rows, quantities)
        assert set(report["quantities"]) == set(quantities)
        assert report["rows"] == 14

    def test_report_records_counts_and_recall(self, validation_table):
        rows, quantities = load_validation_table(validation_table)
        report = build_report(rows, quantities)
        counts = report["quantities"]["diameter_nm"]["counts"]
        assert counts["reference_particles"] == 14
        assert counts["missed_by_corpus"] >= 1
        assert 0 < counts["recall"] < 1

    def test_report_breaks_down_by_morphology(self, validation_table):
        rows, quantities = load_validation_table(validation_table)
        report = build_report(rows, quantities)
        assert "morphology" in report["strata"]
        assert set(report["strata"]["morphology"]) == {"core_shell_sphere", "core_shell_rod"}

    def test_a_single_stratum_is_not_broken_out(self, tmp_path):
        path = write_table(tmp_path / "one.csv", [
            {"image_id": "i", "particle_id": f"p{index}",
             "reference_diameter_nm": 100 + index, "corpus_diameter_nm": 100 + index,
             "morphology": "core_shell_sphere"}
            for index in range(5)
        ])
        rows, quantities = load_validation_table(path)
        report = build_report(rows, quantities)
        assert "morphology" not in report["strata"]

    def test_targets_are_evaluated_and_pass_for_a_good_run(self, validation_table):
        rows, quantities = load_validation_table(validation_table)
        report = build_report(rows, quantities)
        assert report["targets_summary"]["all_pass"] is True

    def test_targets_fail_for_a_biased_run(self, tmp_path):
        path = write_table(tmp_path / "biased.csv", [
            {"image_id": "i", "particle_id": f"p{index}",
             "reference_diameter_nm": 100.0, "corpus_diameter_nm": 150.0}
            for index in range(6)
        ])
        rows, quantities = load_validation_table(path)
        report = build_report(rows, quantities)
        assert report["targets_summary"]["all_pass"] is False

    def test_too_few_pairs_reports_a_note_not_a_crash(self, tmp_path):
        path = write_table(tmp_path / "tiny.csv", [
            {"image_id": "i", "particle_id": "p1",
             "reference_diameter_nm": 100.0, "corpus_diameter_nm": 101.0},
        ])
        rows, quantities = load_validation_table(path)
        report = build_report(rows, quantities)
        block = report["quantities"]["diameter_nm"]
        assert block["metrics"] is None
        assert "at least 2" in block["note"]

    def test_report_is_json_serialisable(self, validation_table):
        rows, quantities = load_validation_table(validation_table)
        json.dumps(build_report(rows, quantities))

    def test_write_report_emits_every_artifact(self, validation_table, tmp_path):
        rows, quantities = load_validation_table(validation_table)
        report = build_report(rows, quantities)
        out = tmp_path / "out"
        write_report(report, out, rows=rows)
        for name in ("report.json", "metrics.csv", "paired_rows.csv", "report.md"):
            assert (out / name).exists(), name

    def test_markdown_names_the_excluded_rows(self, validation_table, tmp_path):
        rows, quantities = load_validation_table(validation_table)
        out = tmp_path / "out"
        write_report(build_report(rows, quantities), out, rows=rows)
        text = (out / "report.md").read_text(encoding="utf-8")
        assert "corpus_missed_particle" in text
        assert "flagged_ambiguous" in text
        assert "Recall" in text


class TestValidationCli:
    def test_one_command_produces_the_report(self, validation_table, tmp_path, capsys):
        out = tmp_path / "cli_out"
        code = validation_main(["--table", str(validation_table), "--out", str(out)])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert payload["rows"] == 14
        assert (out / "report.json").exists()
        assert (out / "report.md").exists()

    def test_figures_are_written_when_matplotlib_is_present(self, validation_table, tmp_path, capsys):
        pytest.importorskip("matplotlib")
        out = tmp_path / "fig_out"
        validation_main(["--table", str(validation_table), "--out", str(out)])
        payload = json.loads(capsys.readouterr().out)
        assert payload["figures"] > 0
        assert (out / "scatter_diameter_nm.png").exists()
        assert (out / "bland_altman_diameter_nm.png").exists()
        assert (out / "residuals_diameter_nm.png").exists()
        assert (out / "error_vs_size_diameter_nm.png").exists()

    def test_no_figures_flag_is_respected(self, validation_table, tmp_path, capsys):
        out = tmp_path / "nofig_out"
        validation_main(["--table", str(validation_table), "--out", str(out), "--no-figures"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["figures"] == 0
        assert not list(out.glob("*.png"))

    def test_print_schema_lists_the_columns(self, capsys):
        assert validation_main(["--print-schema", "--out", "unused"]) == 0
        assert "reference_diameter_nm" in capsys.readouterr().out

    def test_a_bad_table_exits_with_code_two(self, tmp_path, capsys):
        bad = tmp_path / "bad.csv"
        bad.write_text("nope\n1\n", encoding="utf-8")
        assert validation_main(["--table", str(bad), "--out", str(tmp_path / "o")]) == 2

    def test_fail_on_target_gates_a_biased_run(self, tmp_path, capsys):
        path = write_table(tmp_path / "biased.csv", [
            {"image_id": "i", "particle_id": f"p{index}",
             "reference_diameter_nm": 100.0, "corpus_diameter_nm": 150.0}
            for index in range(6)
        ])
        code = validation_main(["--table", str(path), "--out", str(tmp_path / "o"),
                                "--no-figures", "--fail-on-target"])
        assert code == 1

    def test_separate_reference_and_corpus_tables_are_joined(self, tmp_path, capsys):
        reference = write_table(tmp_path / "ref.csv", [
            {"image_id": "i", "particle_id": f"p{index}", "reference_diameter_nm": 100.0 + index}
            for index in range(6)
        ])
        corpus = write_table(tmp_path / "cor.csv", [
            {"image_id": "i", "particle_id": f"p{index}", "corpus_diameter_nm": 101.0 + index}
            for index in range(6)
        ])
        out = tmp_path / "joined"
        code = validation_main(["--reference", str(reference), "--corpus", str(corpus),
                                "--out", str(out), "--no-figures"])
        assert code == 0
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        assert report["quantities"]["diameter_nm"]["metrics"]["n"] == 6
        assert report["quantities"]["diameter_nm"]["metrics"]["mean_bias"] == pytest.approx(1.0)
