"""Result serialisation: the JSON record and the legacy text output."""

import json
from pathlib import Path

__all__ = ["OUTPUT_IMAGE", "DIAMETERS_TXT", "MEASUREMENTS_JSON",
           "write_compat_files", "write_measurements_json", "measurement_rows"]

OUTPUT_IMAGE = "processed_image.jpg"
DIAMETERS_TXT = "diameters.txt"
MEASUREMENTS_JSON = "measurements.json"


def write_compat_files(nm_per_px, class_measurements, preferred_class, path=DIAMETERS_TXT):
    """Write ``diameters.txt``: nm/px on line 1, then one diameter per line.

    Kept for scripts written against the original Corpus output.
    """
    preferred = [row for row in class_measurements if row["class"] == preferred_class]
    rows = preferred or class_measurements
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{nm_per_px}\n")
        for row in rows:
            handle.write(f"{row['diameter']}\n")
    return path


def write_measurements_json(payload, path=MEASUREMENTS_JSON):
    """Write the full run record, sorted so two identical runs diff clean."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def measurement_rows(payload):
    """Flatten a ``measurements.json`` payload into CSV-ready object rows.

    This is the shape the validation harness compares against a reference
    table, so it is defined once here rather than in each consumer.
    """
    rows = []
    for row in payload.get("measurements", []):
        rows.append({
            "object_id": row.get("object_id", ""),
            "preset": row.get("preset", ""),
            "center_x": row.get("center_x", 0),
            "center_y": row.get("center_y", 0),
            "core_nm": row.get("inner_major_axis", 0),
            "outer_nm": row.get("outer_major_axis", 0),
            "diameter_nm": row.get("outer_major_axis") or row.get("inner_major_axis", 0),
            "shell_nm": row.get("shell_thickness_estimate", 0),
            "pair_status": row.get("pair_status", ""),
            "review_status": row.get("review_status", ""),
            "confidence_score": row.get("confidence_score", 0),
            "backend": row.get("backend", ""),
            "flags": "|".join(row.get("flags", [])),
        })
    return rows
