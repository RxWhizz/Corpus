"""Confidence scoring, review gating and image-level warnings.

The scores here are deliberately simple and additive: a reviewer must be able
to read a flag list and reconstruct why Corpus asked for a second look.
"""

import cv2
import numpy as np

__all__ = [
    "CONFIDENCE_PENALTIES",
    "READY_THRESHOLD",
    "LOW_CONTRAST_STD",
    "confidence_score",
    "review_status",
    "build_warnings",
]

#: Penalty applied once per flag group present on an object.
CONFIDENCE_PENALTIES = (
    (("unpaired_inner", "unpaired_outer"), 0.3),
    (("edge",), 0.2),
    (("ratio_outlier",), 0.2),
    (("low_split_confidence",), 0.2),
)

#: Objects at or above this confidence, with no review flags, are ``ready``.
READY_THRESHOLD = 0.7

#: Grey-level standard deviation below which an image is flagged low contrast.
LOW_CONTRAST_STD = 18

#: Flags that describe how a particle was found, not whether it is trustworthy.
NON_REVIEW_FLAGS = frozenset({"watershed_split"})


def confidence_score(flags):
    """Additive confidence in ``[0, 1]`` derived from ``flags``."""
    present = set(flags)
    score = 1.0
    for group, penalty in CONFIDENCE_PENALTIES:
        if present.intersection(group):
            score -= penalty
    return max(0.0, round(score, 3))


def review_status(score, flags):
    """``"ready"`` only when confidence is sufficient *and* no flag remains."""
    review_flags = set(flags) - NON_REVIEW_FLAGS
    if score >= READY_THRESHOLD and not review_flags:
        return "ready"
    return "needs_review"


def build_warnings(image, objects, selected_scale):
    """Image-level caveats surfaced next to the results table."""
    warnings = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if float(np.std(gray)) < LOW_CONTRAST_STD:
        warnings.append("Low contrast")
    if selected_scale.get("method") not in ("manual_line", "manual_override"):
        warnings.append("Scale was auto-detected; manual scale is recommended")
    if objects:
        edge_count = sum(1 for row in objects if "edge" in row.get("flags", []))
        unpaired_count = sum(1 for row in objects if row.get("pair_status") == "partial")
        if edge_count / len(objects) > 0.25:
            warnings.append("Too many edge objects")
        if unpaired_count / len(objects) > 0.35:
            warnings.append("Many unpaired objects")
    return warnings
