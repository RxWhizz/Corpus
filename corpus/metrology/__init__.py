"""Metrology: core-shell derivations and distribution summaries."""

from corpus.metrology.core_shell import (
    RATIO_MAX,
    RATIO_MIN,
    core_outer_ratio,
    is_ratio_outlier,
    object_row,
    shell_thickness,
)
from corpus.metrology.statistics import (
    MIN_N_FOR_NORMALITY,
    build_normality_report,
    normality_stats,
    summarize_class_measurements,
    summarize_decorated,
    summarize_objects,
)

__all__ = [
    "RATIO_MAX",
    "RATIO_MIN",
    "core_outer_ratio",
    "is_ratio_outlier",
    "object_row",
    "shell_thickness",
    "MIN_N_FOR_NORMALITY",
    "build_normality_report",
    "normality_stats",
    "summarize_class_measurements",
    "summarize_decorated",
    "summarize_objects",
]
