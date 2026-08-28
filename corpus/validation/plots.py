"""Publication-ready validation figures.

Matplotlib is imported with the ``Agg`` backend so this works head-less in CI
and over SSH. Figures are optional: if matplotlib is unavailable the report is
still written, with the omission recorded.
"""

from pathlib import Path

from corpus.validation.metrics import bland_altman
from corpus.validation.table import pair_quantity

__all__ = ["MATPLOTLIB_AVAILABLE", "write_figures"]

try:  # pragma: no cover - exercised by whichever branch the environment has
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

FIGURE_DPI = 200


def _finish(figure, axis_or_axes, path):
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path.name


def _scatter(reference, corpus, quantity, output_dir):
    """Corpus vs reference with the identity line."""
    figure, axis = plt.subplots(figsize=(4.6, 4.4))
    axis.scatter(reference, corpus, s=22, alpha=0.75, edgecolor="none")
    low = min(min(reference), min(corpus))
    high = max(max(reference), max(corpus))
    pad = 0.05 * (high - low or 1)
    axis.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--", linewidth=1,
              color="0.4", label="identity")
    axis.set_xlabel(f"Reference {quantity} (nm)")
    axis.set_ylabel(f"Corpus {quantity} (nm)")
    axis.set_title(f"Corpus vs reference — {quantity}")
    axis.legend(frameon=False, fontsize=8)
    axis.set_aspect("equal", adjustable="box")
    return _finish(figure, axis, output_dir / f"scatter_{quantity}.png")


def _residuals(reference, corpus, quantity, output_dir):
    """Distribution of ``corpus - reference``."""
    differences = [cor - ref for ref, cor in zip(reference, corpus, strict=True)]
    figure, axis = plt.subplots(figsize=(4.6, 3.4))
    bins = max(6, min(30, len(differences) // 3 or 6))
    axis.hist(differences, bins=bins, edgecolor="white", linewidth=0.6)
    axis.axvline(0, linestyle="--", linewidth=1, color="0.3")
    axis.set_xlabel(f"Corpus − reference {quantity} (nm)")
    axis.set_ylabel("Particles")
    axis.set_title(f"Error distribution — {quantity}")
    return _finish(figure, axis, output_dir / f"residuals_{quantity}.png")


def _bland_altman(reference, corpus, quantity, output_dir):
    """Bland-Altman plot with mean difference and limits of agreement."""
    stats = bland_altman(reference, corpus)
    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    axis.scatter(stats["means"], stats["differences"], s=22, alpha=0.75, edgecolor="none")
    for value, label, style in (
        (stats["mean_difference"], f"bias {stats['mean_difference']:.2f}", "-"),
        (stats["upper_limit"], f"+{stats['limit_multiplier']}·SD", "--"),
        (stats["lower_limit"], f"−{stats['limit_multiplier']}·SD", "--"),
    ):
        axis.axhline(value, linestyle=style, linewidth=1, color="0.35")
        axis.annotate(label, xy=(0.99, value), xycoords=("axes fraction", "data"),
                      ha="right", va="bottom", fontsize=7, color="0.25")
    axis.set_xlabel(f"Mean of reference and Corpus {quantity} (nm)")
    axis.set_ylabel("Difference (nm)")
    axis.set_title(f"Bland–Altman — {quantity}")
    return _finish(figure, axis, output_dir / f"bland_altman_{quantity}.png")


def _error_vs_size(reference, corpus, quantity, output_dir):
    """Absolute error against reference size, to expose size-dependent bias."""
    errors = [abs(cor - ref) for ref, cor in zip(reference, corpus, strict=True)]
    figure, axis = plt.subplots(figsize=(4.8, 3.4))
    axis.scatter(reference, errors, s=22, alpha=0.75, edgecolor="none")
    axis.set_xlabel(f"Reference {quantity} (nm)")
    axis.set_ylabel("Absolute error (nm)")
    axis.set_title(f"Error vs particle size — {quantity}")
    return _finish(figure, axis, output_dir / f"error_vs_size_{quantity}.png")


def write_figures(rows, quantities, output_dir):
    """Write every figure that has enough data. Returns the file names."""
    if not MATPLOTLIB_AVAILABLE:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for quantity in quantities:
        reference, corpus, _ = pair_quantity(rows, quantity)
        if len(reference) < 3:
            continue  # a two-point scatter is not a figure
        names.append(_scatter(reference, corpus, quantity, output_dir))
        names.append(_residuals(reference, corpus, quantity, output_dir))
        names.append(_bland_altman(reference, corpus, quantity, output_dir))
        names.append(_error_vs_size(reference, corpus, quantity, output_dir))
    return names
