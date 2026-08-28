"""Core-shell derivations.

The defining invariant of this module, and the one the test suite pins:

    t_shell = (D_outer - D_core) / 2

Everything else (ratio checks, pairing status, confidence) exists to tell a
reviewer when that subtraction should not be trusted.
"""

from corpus.review.scoring import confidence_score, review_status

__all__ = [
    "RATIO_MIN",
    "RATIO_MAX",
    "shell_thickness",
    "core_outer_ratio",
    "is_ratio_outlier",
    "object_row",
]

#: Plausible core/outer diameter ratio for a real core-shell particle. Outside
#: this band the pairing is more likely a mis-association than a thin/thick shell.
RATIO_MIN = 0.25
RATIO_MAX = 0.85


def shell_thickness(outer_diameter, core_diameter):
    """Shell thickness from concentric diameters: ``(D_outer - D_core) / 2``.

    Returns ``0`` when the core is reported larger than the outer shell, which
    means the pairing is wrong rather than the shell being negative.
    """
    if outer_diameter is None or core_diameter is None:
        return 0.0
    if outer_diameter < core_diameter:
        return 0.0
    return (float(outer_diameter) - float(core_diameter)) / 2


def core_outer_ratio(core_diameter, outer_diameter):
    """``D_core / D_outer``; ``0`` when the outer diameter is unknown."""
    if not outer_diameter:
        return 0.0
    return float(core_diameter) / float(outer_diameter)


def is_ratio_outlier(ratio):
    """True when the core/outer ratio falls outside the plausible band."""
    return ratio < RATIO_MIN or ratio > RATIO_MAX


def object_row(index, preset, inner=None, outer=None, extra_flags=None):
    """Assemble one reviewable object from an optional inner/outer pair.

    An object is emitted even when only one half was found; it is marked
    ``partial`` and flagged, never dropped, so recall stays auditable.
    """
    flags = list(extra_flags or [])
    for measurement in (inner, outer):
        if measurement:
            flags.extend(measurement.get("flags", []))
    if not inner:
        flags.append("unpaired_inner")
    if not outer:
        flags.append("unpaired_outer")

    inner_major = inner.get("major_axis", 0) if inner else 0
    outer_major = outer.get("major_axis", 0) if outer else 0
    inner_minor = inner.get("minor_axis", 0) if inner else 0
    outer_minor = outer.get("minor_axis", 0) if outer else 0

    paired = bool(inner and outer)
    shell = shell_thickness(outer_major, inner_major) if paired else 0
    ratio = core_outer_ratio(inner_major, outer_major) if paired else 0
    center_x = (outer or inner or {}).get("center_x", 0)
    center_y = (outer or inner or {}).get("center_y", 0)

    if paired and is_ratio_outlier(ratio):
        flags.append("ratio_outlier")

    confidence = confidence_score(flags)
    if confidence < 0.7 and "low_confidence" not in flags:
        flags.append("low_confidence")

    backend = (outer or inner or {}).get("backend", "classical")

    return {
        "object_id": f"obj_{index:04d}",
        "preset": preset,
        "class": "core_shell_object",
        "center_x": center_x,
        "center_y": center_y,
        "inner_major_axis": inner_major,
        "inner_minor_axis": inner_minor,
        "outer_major_axis": outer_major,
        "outer_minor_axis": outer_minor,
        "equivalent_diameter": (
            outer.get("equivalent_diameter", 0) if outer
            else inner.get("equivalent_diameter", 0) if inner
            else 0
        ),
        "shell_thickness_estimate": shell,
        "inner_outer_ratio": ratio,
        "pair_status": "paired" if paired else "partial",
        "review_status": review_status(confidence, flags),
        "confidence_score": confidence,
        "separation_method": "watershed" if "watershed_split" in flags else "contour/hough",
        "backend": backend,
        "flags": sorted(set(flags)),
        "inner": inner,
        "outer": outer,
    }
