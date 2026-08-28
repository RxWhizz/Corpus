"""Distribution summaries reported next to every measurement run.

Corpus never claims a distribution is normal. With few particles it reports
``insufficient_n``; with enough it reports a *hint* from skewness and excess
kurtosis, which is a prompt to look at the histogram, not a test result.
"""

import math

__all__ = [
    "MIN_N_FOR_NORMALITY",
    "normality_stats",
    "build_normality_report",
    "summarize_class_measurements",
    "summarize_objects",
    "summarize_decorated",
]

#: Below this count no distribution hint is attempted.
MIN_N_FOR_NORMALITY = 8


def normality_stats(values):
    """Descriptive stats plus a cautious ``normality_hint``.

    Non-positive and missing values are dropped: a diameter of zero means "not
    measured", not "a particle of size zero".
    """
    values = [float(value) for value in values if value and value > 0]
    n = len(values)
    if n == 0:
        return {"n": 0, "normality_hint": "no data"}
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / n
    std = math.sqrt(variance)
    if n < MIN_N_FOR_NORMALITY or std <= 1e-9:
        return {"n": n, "mean": mean, "std": std, "normality_hint": "insufficient_n"}
    skewness = sum(((value - mean) / std) ** 3 for value in values) / n
    kurtosis_excess = sum(((value - mean) / std) ** 4 for value in values) / n - 3
    hint = "roughly_normal" if abs(skewness) < 1 and abs(kurtosis_excess) < 2 else "check_distribution"
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "kurtosis_excess": kurtosis_excess,
        "normality_hint": hint,
    }


def build_normality_report(objects, class_measurements):
    """Per-class and per-quantity distribution hints for ``measurements.json``."""
    report = {}
    for class_name in sorted({row["class"] for row in class_measurements}):
        report[f"{class_name}_diameter"] = normality_stats(
            [row["diameter"] for row in class_measurements if row["class"] == class_name]
        )
    report["inner_major_axis"] = normality_stats([row.get("inner_major_axis") for row in objects])
    report["outer_major_axis"] = normality_stats([row.get("outer_major_axis") for row in objects])
    report["shell_thickness"] = normality_stats([row.get("shell_thickness_estimate") for row in objects])
    return report


def summarize_class_measurements(class_measurements):
    """Count and diameter range per detected class."""
    summary = {}
    for class_name in sorted({row["class"] for row in class_measurements}):
        values = [row["diameter"] for row in class_measurements if row["class"] == class_name]
        summary[class_name] = {
            "count": len(values),
            "mean_diameter": sum(values) / len(values) if values else 0,
            "min_diameter": min(values) if values else 0,
            "max_diameter": max(values) if values else 0,
        }
    return summary


def summarize_objects(objects):
    """Object-level counts, including how many still need review."""
    paired = [row for row in objects if row["pair_status"] == "paired"]
    ready = [row for row in objects if row["review_status"] == "ready"]
    watershed_splits = [row for row in objects if "watershed_split" in row.get("flags", [])]
    inner_values = [row["inner_major_axis"] for row in objects if row.get("inner_major_axis")]
    outer_values = [row["outer_major_axis"] for row in objects if row.get("outer_major_axis")]
    shell_values = [row["shell_thickness_estimate"] for row in objects if row.get("shell_thickness_estimate")]
    return {
        "objects": len(objects),
        "paired": len(paired),
        "ready": len(ready),
        "needs_review": len(objects) - len(ready),
        "watershed_splits": len(watershed_splits),
        "mean_inner_major_axis": sum(inner_values) / len(inner_values) if inner_values else 0,
        "mean_outer_major_axis": sum(outer_values) / len(outer_values) if outer_values else 0,
        "mean_shell_thickness": sum(shell_values) / len(shell_values) if shell_values else 0,
    }


def summarize_decorated(objects):
    """Decoration counts and surface density for the decorated preset."""
    carriers = [row for row in objects if row.get("preset") == "decorated" and row.get("outer_major_axis")]
    if not carriers:
        return None
    counts = [row.get("decoration_count", 0) for row in carriers]
    densities = [row.get("decoration_density_per_1000_nm2", 0) for row in carriers]
    mean_decoration_diameters = [
        row.get("mean_decoration_diameter", 0) for row in carriers if row.get("mean_decoration_diameter", 0) > 0
    ]
    return {
        "carriers": len(carriers),
        "total_decorations_on_carriers": sum(counts),
        "mean_decorations_per_carrier": sum(counts) / len(counts),
        "mean_decoration_density_per_1000_nm2": sum(densities) / len(densities) if densities else 0,
        "mean_decoration_diameter": (
            sum(mean_decoration_diameters) / len(mean_decoration_diameters) if mean_decoration_diameters else 0
        ),
    }
