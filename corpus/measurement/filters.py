"""Particle filter settings and their normalisation.

Filters arrive from the GUI as free text, so they are clamped into valid
ranges here -- once -- and the clamped dict is what gets written to
``measurements.json``. That makes a run reproducible from its own output.
"""

__all__ = ["DEFAULT_FILTER_SETTINGS", "clamp_filter_settings", "parse_bool", "resolve_watershed"]

DEFAULT_FILTER_SETTINGS = {
    "ui_mode": "easy",
    "contrast_strategy": "dark_particles",
    "manual_threshold_min": 0,
    "manual_threshold_max": 255,
    "min_circularity": 0.0,
    "max_circularity": 1.0,
    "min_elongation": 1.0,
    "max_elongation": 999.0,
    "include_holes": False,
    "review_view": "overlay",
}

_FALSEY = ("0", "false", "no", "off")


def parse_bool(value):
    """Permissive boolean parse for CLI/GUI strings."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in _FALSEY


def resolve_watershed(value, shape_preset):
    """``"auto"`` enables watershed everywhere except the rod/pellet preset.

    Elongated particles are the case where watershed tends to over-split, so
    ``pellets`` opts out unless the user asks for it explicitly.
    """
    if str(value).strip().lower() == "auto":
        return shape_preset != "pellets"
    return parse_bool(value)


def clamp_filter_settings(source):
    """Clamp and order filter settings from an argparse namespace or a dict.

    Grey thresholds are clamped to ``[0, 255]``, circularity to ``[0, 1]`` and
    elongation to ``>= 1``; inverted min/max pairs are swapped rather than
    rejected, so a mis-typed range still yields a usable run.
    """
    get = source.get if isinstance(source, dict) else (lambda key, default=None: getattr(source, key, default))

    threshold_min = max(0, min(255, get("manual_threshold_min", 0)))
    threshold_max = max(0, min(255, get("manual_threshold_max", 255)))
    if threshold_min > threshold_max:
        threshold_min, threshold_max = threshold_max, threshold_min

    min_circularity = max(0.0, min(1.0, get("min_circularity", 0.0)))
    max_circularity = max(0.0, min(1.0, get("max_circularity", 1.0)))
    if min_circularity > max_circularity:
        min_circularity, max_circularity = max_circularity, min_circularity

    min_elongation = max(1.0, get("min_elongation", 1.0))
    max_elongation = max(1.0, get("max_elongation", 999.0))
    if min_elongation > max_elongation:
        min_elongation, max_elongation = max_elongation, min_elongation

    return {
        "ui_mode": get("ui_mode", "easy"),
        "contrast_strategy": get("contrast_strategy", "dark_particles"),
        "manual_threshold_min": threshold_min,
        "manual_threshold_max": threshold_max,
        "min_circularity": min_circularity,
        "max_circularity": max_circularity,
        "min_elongation": min_elongation,
        "max_elongation": max_elongation,
        "include_holes": parse_bool(get("include_holes", False)),
        "review_view": get("review_view", "overlay"),
    }
