"""Build the validation report from a validation table.

Output is deliberately three-part: JSON for machines, CSV for spreadsheets,
and Markdown for the report committed under ``docs/validation/``. Metrics are
reported per quantity and per stratum (morphology/mode), and unmatched rows are
listed explicitly rather than filtered away.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from corpus import __version__
from corpus.errors import ValidationInputError
from corpus.validation.metrics import agreement_metrics, count_agreement
from corpus.validation.table import group_rows, pair_quantity

__all__ = ["build_report", "write_report", "PILOT_TARGETS"]

#: Engineering targets from the epic. These are release gates, not scientific
#: claims about the accuracy of the method.
PILOT_TARGETS = {
    "diameter_nm": {"max_mae_percent": 10.0, "max_abs_bias_percent": 5.0},
    "core_nm": {"max_mae_percent": 15.0, "max_abs_bias_percent": 7.5},
    "outer_nm": {"max_mae_percent": 10.0, "max_abs_bias_percent": 5.0},
    "shell_nm": {"max_mae_percent": 20.0, "max_abs_bias_percent": 10.0},
}


def _evaluate_targets(quantity, metrics):
    """Compare one metric block against its engineering target."""
    target = PILOT_TARGETS.get(quantity)
    if not target or metrics.get("mae_percent_of_reference_mean") is None:
        return None
    mae_percent = metrics["mae_percent_of_reference_mean"]
    bias_percent = abs(metrics.get("bias_percent_of_reference_mean") or 0.0)
    checks = {
        "mae_percent": {
            "value": mae_percent,
            "limit": target["max_mae_percent"],
            "pass": mae_percent <= target["max_mae_percent"],
        },
        "abs_bias_percent": {
            "value": bias_percent,
            "limit": target["max_abs_bias_percent"],
            "pass": bias_percent <= target["max_abs_bias_percent"],
        },
    }
    return {"checks": checks, "pass": all(check["pass"] for check in checks.values())}


def _quantity_block(rows, quantity):
    """Metrics, counts and unmatched rows for one quantity."""
    reference, corpus, unmatched = pair_quantity(rows, quantity)
    applicable = [
        row for row in rows
        if row.get(f"reference_{quantity}") is not None or row.get(f"corpus_{quantity}") is not None
    ]
    reference_total = sum(1 for row in applicable if row.get(f"reference_{quantity}") is not None)
    corpus_total = sum(1 for row in applicable if row.get(f"corpus_{quantity}") is not None)

    block = {
        "quantity": quantity,
        "counts": count_agreement(reference_total, corpus_total, len(reference)),
        "unmatched": unmatched,
        "unmatched_reasons": _tally(item["reason"] for item in unmatched),
    }
    if len(reference) < 2:
        block["metrics"] = None
        block["note"] = (
            f"Only {len(reference)} comparable pair(s); agreement metrics need at least 2."
        )
        block["targets"] = None
        return block

    metrics = agreement_metrics(reference, corpus, quantity)
    block["metrics"] = metrics
    block["targets"] = _evaluate_targets(quantity, metrics)
    return block


def _tally(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_report(rows, quantities, stratify_by=("morphology", "mode"), reference_name="manual"):
    """Assemble the full report structure.

    ``stratify_by`` columns are only broken out when the table actually has
    more than one value for them, so a single-morphology pilot stays readable.
    """
    if not rows:
        raise ValidationInputError("No validation rows to report on.")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_version": __version__,
        "reference": reference_name,
        "images": len(sorted({row.get("image_id", "") for row in rows})),
        "rows": len(rows),
        "quantities": {},
        "strata": {},
        "targets_summary": {},
    }

    for quantity in quantities:
        report["quantities"][quantity] = _quantity_block(rows, quantity)

    for column in stratify_by:
        grouped = group_rows(rows, column)
        if len(grouped) < 2:
            continue  # nothing to compare -- do not pad the report
        report["strata"][column] = {
            value: {quantity: _quantity_block(subset, quantity) for quantity in quantities}
            for value, subset in sorted(grouped.items())
        }

    evaluated = {
        quantity: block["targets"]["pass"]
        for quantity, block in report["quantities"].items()
        if block.get("targets")
    }
    report["targets_summary"] = {
        "evaluated": evaluated,
        "all_pass": all(evaluated.values()) if evaluated else None,
    }
    return report


def _metrics_csv_rows(report):
    rows = []

    def emit(stratum, value, quantity, block):
        metrics = block.get("metrics") or {}
        counts = block.get("counts", {})
        altman = (metrics.get("bland_altman") or {})
        rows.append({
            "stratum": stratum,
            "stratum_value": value,
            "quantity": quantity,
            "n": metrics.get("n", 0),
            "reference_mean": metrics.get("reference_mean"),
            "corpus_mean": metrics.get("corpus_mean"),
            "mae": metrics.get("mae"),
            "mae_percent_of_reference_mean": metrics.get("mae_percent_of_reference_mean"),
            "rmse": metrics.get("rmse"),
            "mean_bias": metrics.get("mean_bias"),
            "bias_percent_of_reference_mean": metrics.get("bias_percent_of_reference_mean"),
            "mean_relative_error": metrics.get("mean_relative_error"),
            "r_squared_identity": metrics.get("r_squared_identity"),
            "pearson_r": metrics.get("pearson_r"),
            "bland_altman_lower": altman.get("lower_limit"),
            "bland_altman_upper": altman.get("upper_limit"),
            "reference_particles": counts.get("reference_particles"),
            "corpus_particles": counts.get("corpus_particles"),
            "matched_particles": counts.get("matched_particles"),
            "missed_by_corpus": counts.get("missed_by_corpus"),
            "extra_in_corpus": counts.get("extra_in_corpus"),
            "recall": counts.get("recall"),
            "precision": counts.get("precision"),
            "target_pass": (block.get("targets") or {}).get("pass"),
        })

    for quantity, block in report["quantities"].items():
        emit("overall", "all", quantity, block)
    for column, values in report.get("strata", {}).items():
        for value, quantities in values.items():
            for quantity, block in quantities.items():
                emit(column, value, quantity, block)
    return rows


def _format_number(value, digits=3):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _markdown(report, figure_names=()):
    lines = [
        "# Corpus validation report",
        "",
        f"- Generated: `{report['generated_utc']}`",
        f"- Corpus version: `{report['corpus_version']}`",
        f"- Reference: **{report['reference']}**",
        f"- Images: {report['images']} | rows: {report['rows']}",
        "",
        "These are engineering agreement targets for the release gate, not a",
        "scientific accuracy claim about the method.",
        "",
        "## Overall agreement",
        "",
        "| Quantity | n | Ref mean | Corpus mean | MAE | MAE % | RMSE | Bias | Bias % | R² (identity) | r | Recall | Target |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for quantity, block in report["quantities"].items():
        metrics = block.get("metrics")
        counts = block.get("counts", {})
        target = block.get("targets")
        if not metrics:
            lines.append(f"| `{quantity}` | {counts.get('matched_particles', 0)} | " + " | ".join(["n/a"] * 10) + " | n/a |")
            continue
        lines.append(
            f"| `{quantity}` | {metrics['n']} | {_format_number(metrics['reference_mean'], 2)} | "
            f"{_format_number(metrics['corpus_mean'], 2)} | {_format_number(metrics['mae'], 2)} | "
            f"{_format_number(metrics['mae_percent_of_reference_mean'], 1)}% | {_format_number(metrics['rmse'], 2)} | "
            f"{_format_number(metrics['mean_bias'], 2)} | {_format_number(metrics['bias_percent_of_reference_mean'], 1)}% | "
            f"{_format_number(metrics['r_squared_identity'])} | {_format_number(metrics['pearson_r'])} | "
            f"{_format_number(counts.get('recall'))} | "
            f"{'PASS' if target and target['pass'] else 'FAIL' if target else 'n/a'} |"
        )

    lines += ["", "## Detection counts", "",
              "| Quantity | Reference | Corpus | Matched | Missed | Extra | Recall | Precision |",
              "|---|---|---|---|---|---|---|---|"]
    for quantity, block in report["quantities"].items():
        counts = block.get("counts", {})
        lines.append(
            f"| `{quantity}` | {counts.get('reference_particles')} | {counts.get('corpus_particles')} | "
            f"{counts.get('matched_particles')} | {counts.get('missed_by_corpus')} | "
            f"{counts.get('extra_in_corpus')} | {_format_number(counts.get('recall'))} | "
            f"{_format_number(counts.get('precision'))} |"
        )

    lines += ["", "## Excluded and unmatched rows", "",
              "Rows that could not be compared. They are listed, never dropped silently.", ""]
    any_unmatched = False
    for quantity, block in report["quantities"].items():
        reasons = block.get("unmatched_reasons") or {}
        if not reasons:
            continue
        any_unmatched = True
        lines.append(f"- `{quantity}`: " + ", ".join(f"{reason} × {count}" for reason, count in reasons.items()))
    if not any_unmatched:
        lines.append("- None.")

    for column, values in report.get("strata", {}).items():
        lines += ["", f"## Breakdown by `{column}`", "",
                  "| Value | Quantity | n | MAE | MAE % | Bias | Recall |", "|---|---|---|---|---|---|---|"]
        for value, quantities in values.items():
            for quantity, block in quantities.items():
                metrics = block.get("metrics")
                counts = block.get("counts", {})
                if not metrics:
                    lines.append(f"| {value} | `{quantity}` | {counts.get('matched_particles', 0)} | n/a | n/a | n/a | "
                                 f"{_format_number(counts.get('recall'))} |")
                    continue
                lines.append(
                    f"| {value} | `{quantity}` | {metrics['n']} | {_format_number(metrics['mae'], 2)} | "
                    f"{_format_number(metrics['mae_percent_of_reference_mean'], 1)}% | "
                    f"{_format_number(metrics['mean_bias'], 2)} | {_format_number(counts.get('recall'))} |"
                )

    if figure_names:
        lines += ["", "## Figures", ""]
        lines += [f"![{name}]({name})" for name in figure_names]

    lines += ["", "## How to reproduce", "",
              "```bash", "python -m corpus.validation --table <validation_table.csv> --out <output_dir>", "```", ""]
    return "\n".join(lines)


def write_report(report, output_dir, rows=None, figure_names=()):
    """Write ``report.json``, ``metrics.csv``, ``paired_rows.csv`` and ``report.md``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    metrics_rows = _metrics_csv_rows(report)
    if metrics_rows:
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0]))
            writer.writeheader()
            writer.writerows(metrics_rows)

    if rows:
        columns = sorted({key for row in rows for key in row})
        with (output_dir / "paired_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in columns})

    (output_dir / "report.md").write_text(_markdown(report, figure_names), encoding="utf-8")
    return output_dir
