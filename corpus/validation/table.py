"""The validation table: schema, loading, and pairing.

One row per particle, carrying both the reference and the Corpus value, as
specified by the epic's validation-table schema. Rows where either side is
missing are retained and counted as unmatched -- never dropped -- so recall
stays visible.
"""

import csv
import math
from pathlib import Path

from corpus.errors import ValidationInputError

__all__ = [
    "QUANTITIES",
    "IDENTITY_COLUMNS",
    "OPTIONAL_COLUMNS",
    "schema_columns",
    "load_validation_table",
    "join_reference_and_corpus",
    "pair_quantity",
    "group_rows",
]

#: Quantities compared, as ``(name, reference_column, corpus_column)``.
QUANTITIES = (
    ("diameter_nm", "reference_diameter_nm", "corpus_diameter_nm"),
    ("core_nm", "reference_core_nm", "corpus_core_nm"),
    ("outer_nm", "reference_outer_nm", "corpus_outer_nm"),
    ("shell_nm", "reference_shell_nm", "corpus_shell_nm"),
)

#: Columns that identify a row.
IDENTITY_COLUMNS = ("image_id", "particle_id")

#: Recognised but not required. ``morphology`` and ``mode`` drive the
#: per-stratum breakdown; ``status`` marks a row ambiguous for a human.
OPTIONAL_COLUMNS = ("morphology", "mode", "magnification", "source", "status", "notes")


def schema_columns():
    """Full column list of the canonical validation table."""
    columns = list(IDENTITY_COLUMNS)
    for _, reference_column, corpus_column in QUANTITIES:
        columns.extend([reference_column, corpus_column])
    columns.extend(OPTIONAL_COLUMNS)
    return columns


def _to_float(value):
    """Blank/absent -> ``None``; anything unparseable is an error, not a zero."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in ("na", "n/a", "nan", "none", "null"):
        return None
    try:
        number = float(text)
    except ValueError as error:
        raise ValidationInputError(f"Could not read {value!r} as a number.") from error
    if not math.isfinite(number):
        return None
    return number


def load_validation_table(path):
    """Read a merged validation CSV into normalised row dicts."""
    path = Path(path)
    if not path.exists():
        raise ValidationInputError(f"Validation table not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationInputError(f"Validation table has no header row: {path}")
        missing = [column for column in IDENTITY_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValidationInputError(f"Validation table is missing required columns: {missing}")
        known_quantities = [
            (name, reference_column, corpus_column)
            for name, reference_column, corpus_column in QUANTITIES
            if reference_column in reader.fieldnames and corpus_column in reader.fieldnames
        ]
        if not known_quantities:
            raise ValidationInputError(
                "Validation table has no comparable quantity pairs. Expected at least one of: "
                + ", ".join(f"{r}/{c}" for _, r, c in QUANTITIES)
            )
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            row = {column: (raw.get(column) or "").strip() for column in IDENTITY_COLUMNS}
            row.update({column: (raw.get(column) or "").strip() for column in OPTIONAL_COLUMNS})
            try:
                for name, reference_column, corpus_column in known_quantities:
                    row[f"reference_{name}"] = _to_float(raw.get(reference_column))
                    row[f"corpus_{name}"] = _to_float(raw.get(corpus_column))
            except ValidationInputError as error:
                raise ValidationInputError(f"{path}:{line_number}: {error}") from None
            rows.append(row)
    if not rows:
        raise ValidationInputError(f"Validation table has a header but no data rows: {path}")
    return rows, [name for name, _, _ in known_quantities]


def join_reference_and_corpus(reference_rows, corpus_rows, quantities=None):
    """Merge two separate tables on ``(image_id, particle_id)``.

    Unmatched rows from either side are kept with ``None`` on the missing
    side, so they surface in the count agreement rather than disappearing.
    """
    quantities = quantities or [name for name, _, _ in QUANTITIES]

    def key(row):
        return (str(row.get("image_id", "")).strip(), str(row.get("particle_id", "")).strip())

    reference_index = {key(row): row for row in reference_rows}
    corpus_index = {key(row): row for row in corpus_rows}
    merged = []
    for identity in sorted(set(reference_index) | set(corpus_index)):
        reference = reference_index.get(identity, {})
        corpus = corpus_index.get(identity, {})
        row = {"image_id": identity[0], "particle_id": identity[1]}
        for column in OPTIONAL_COLUMNS:
            row[column] = str(reference.get(column, corpus.get(column, "")) or "").strip()
        for name in quantities:
            row[f"reference_{name}"] = _to_float(reference.get(name, reference.get(f"reference_{name}")))
            row[f"corpus_{name}"] = _to_float(corpus.get(name, corpus.get(f"corpus_{name}")))
        merged.append(row)
    return merged


def pair_quantity(rows, quantity):
    """Split ``rows`` for one quantity into matched pairs and unmatched rows.

    Returns ``(reference_values, corpus_values, unmatched)`` where
    ``unmatched`` lists the rows that could not be compared and why.
    """
    reference_values = []
    corpus_values = []
    unmatched = []
    for row in rows:
        reference = row.get(f"reference_{quantity}")
        corpus = row.get(f"corpus_{quantity}")
        if reference is None and corpus is None:
            continue  # this quantity simply does not apply to the row
        if reference is None:
            unmatched.append({**_identity(row), "reason": "no_reference_value"})
            continue
        if corpus is None:
            unmatched.append({**_identity(row), "reason": "corpus_missed_particle"})
            continue
        if str(row.get("status", "")).strip().lower() in ("ambiguous", "excluded", "unresolved"):
            unmatched.append({**_identity(row), "reason": f"flagged_{row['status'].strip().lower()}"})
            continue
        reference_values.append(reference)
        corpus_values.append(corpus)
    return reference_values, corpus_values, unmatched


def _identity(row):
    return {"image_id": row.get("image_id", ""), "particle_id": row.get("particle_id", "")}


def group_rows(rows, column):
    """Group rows by an optional stratum column, e.g. ``morphology``."""
    grouped = {}
    for row in rows:
        key = str(row.get(column, "") or "unspecified").strip() or "unspecified"
        grouped.setdefault(key, []).append(row)
    return grouped
