"""One-command validation run.

    python -m corpus.validation --table docs/validation/pilot_table.csv \
                                --out docs/validation/pilot_2026_08

Writes ``report.json``, ``metrics.csv``, ``paired_rows.csv``, ``report.md`` and
the figures into the output directory. Exit code is non-zero when a configured
engineering target fails, so this can gate a release.
"""

import argparse
import json
import sys

from corpus.errors import ValidationInputError
from corpus.validation.plots import MATPLOTLIB_AVAILABLE, write_figures
from corpus.validation.report import build_report, write_report
from corpus.validation.table import (
    QUANTITIES,
    join_reference_and_corpus,
    load_validation_table,
    schema_columns,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m corpus.validation",
        description="Compare Corpus measurements against a manual/ImageJ reference.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--table", help="Merged validation table CSV (reference_* and corpus_* columns).")
    source.add_argument("--reference", help="Reference-only CSV, joined with --corpus on image_id/particle_id.")
    parser.add_argument("--corpus", help="Corpus-only CSV; required with --reference.")
    parser.add_argument("--out", help="Output directory for the report. Required unless --print-schema.")
    parser.add_argument("--reference-name", default="manual",
                        help="How the reference was produced, recorded in the report.")
    parser.add_argument("--stratify-by", default="morphology,mode",
                        help="Comma-separated columns to break metrics down by.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    parser.add_argument("--print-schema", action="store_true",
                        help="Print the expected table columns and exit.")
    parser.add_argument("--fail-on-target", action="store_true",
                        help="Exit non-zero when an engineering target fails.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.print_schema:
        print(",".join(schema_columns()))
        return 0

    if not (args.table or args.reference):
        print("one of --table or --reference is required", file=sys.stderr)
        return 2
    if args.reference and not args.corpus:
        print("--reference requires --corpus", file=sys.stderr)
        return 2
    if not args.out:
        print("--out is required", file=sys.stderr)
        return 2

    try:
        if args.table:
            rows, quantities = load_validation_table(args.table)
        else:
            reference_rows, _ = load_validation_table(args.reference)
            corpus_rows, _ = load_validation_table(args.corpus)
            quantities = [name for name, _, _ in QUANTITIES]
            rows = join_reference_and_corpus(reference_rows, corpus_rows, quantities)

        stratify_by = tuple(part.strip() for part in args.stratify_by.split(",") if part.strip())
        report = build_report(rows, quantities, stratify_by, reference_name=args.reference_name)

        figure_names = []
        if not args.no_figures:
            if MATPLOTLIB_AVAILABLE:
                figure_names = write_figures(rows, quantities, args.out)
            else:
                report["figures_skipped"] = "matplotlib is not installed"

        write_report(report, args.out, rows=rows, figure_names=figure_names)
    except ValidationInputError as error:
        print(json.dumps({"ok": False, "message": str(error)}), file=sys.stderr)
        return 2

    summary = report["targets_summary"]
    print(json.dumps({
        "ok": True,
        "out": str(args.out),
        "rows": report["rows"],
        "images": report["images"],
        "quantities": list(report["quantities"]),
        "figures": len(figure_names),
        "targets": summary,
    }, indent=2))

    if args.fail_on_target and summary.get("all_pass") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
