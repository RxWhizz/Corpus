"""Agreement metrics for Corpus vs. a manual/ImageJ reference.

These are *agreement* statistics, not goodness-of-fit of a free regression.
Where a choice exists, the identity line ``corpus = reference`` is the model,
because that is the claim being tested.

Every function returns plain floats/dicts so results serialise straight to
JSON and can be diffed between pilot runs.
"""

import math

from corpus.errors import ValidationInputError

__all__ = [
    "paired_values",
    "mean_absolute_error",
    "root_mean_square_error",
    "mean_bias",
    "mean_relative_error",
    "r_squared_identity",
    "pearson_r",
    "bland_altman",
    "agreement_metrics",
    "count_agreement",
]


def paired_values(reference, corpus):
    """Validate and return two equal-length numeric sequences.

    Raises :class:`ValidationInputError` on length mismatch or non-finite
    values -- a silent drop here would flatter the results.
    """
    reference = list(reference)
    corpus = list(corpus)
    if len(reference) != len(corpus):
        raise ValidationInputError(
            f"Reference and Corpus series must be the same length ({len(reference)} vs {len(corpus)})."
        )
    if not reference:
        raise ValidationInputError("Cannot compute agreement metrics on an empty series.")
    for index, (ref, cor) in enumerate(zip(reference, corpus, strict=True)):
        if not (math.isfinite(ref) and math.isfinite(cor)):
            raise ValidationInputError(f"Non-finite value at paired index {index}: ({ref}, {cor}).")
    return reference, corpus


def mean_absolute_error(reference, corpus):
    """Mean |corpus - reference|, in the unit of the inputs."""
    reference, corpus = paired_values(reference, corpus)
    return sum(abs(cor - ref) for ref, cor in zip(reference, corpus, strict=True)) / len(reference)


def root_mean_square_error(reference, corpus):
    """RMSE; penalises the large individual errors MAE averages away."""
    reference, corpus = paired_values(reference, corpus)
    return math.sqrt(sum((cor - ref) ** 2 for ref, cor in zip(reference, corpus, strict=True)) / len(reference))


def mean_bias(reference, corpus):
    """Signed mean of ``corpus - reference``. Positive = Corpus over-reports."""
    reference, corpus = paired_values(reference, corpus)
    return sum(cor - ref for ref, cor in zip(reference, corpus, strict=True)) / len(reference)


def mean_relative_error(reference, corpus):
    """Mean |error| / reference, as a fraction. Reference values must be > 0."""
    reference, corpus = paired_values(reference, corpus)
    if any(ref <= 0 for ref in reference):
        raise ValidationInputError("Relative error needs strictly positive reference values.")
    return sum(abs(cor - ref) / ref for ref, cor in zip(reference, corpus, strict=True)) / len(reference)


def r_squared_identity(reference, corpus):
    """R^2 of the *identity* line, i.e. ``1 - SS_res/SS_tot``.

    This can go negative, which is informative: it means Corpus tracks the
    reference worse than just predicting the reference mean. A free-slope
    regression R^2 would hide exactly the systematic offset we care about.
    """
    reference, corpus = paired_values(reference, corpus)
    mean_reference = sum(reference) / len(reference)
    ss_total = sum((ref - mean_reference) ** 2 for ref in reference)
    ss_residual = sum((cor - ref) ** 2 for ref, cor in zip(reference, corpus, strict=True))
    if ss_total == 0:
        raise ValidationInputError("R^2 is undefined when every reference value is identical.")
    return 1 - ss_residual / ss_total


def pearson_r(reference, corpus):
    """Pearson correlation -- reported alongside R^2, never instead of it.

    A high ``r`` with a bad bias still means Corpus disagrees with the
    reference, so both numbers are always shown together.
    """
    reference, corpus = paired_values(reference, corpus)
    n = len(reference)
    mean_reference = sum(reference) / n
    mean_corpus = sum(corpus) / n
    covariance = sum((ref - mean_reference) * (cor - mean_corpus) for ref, cor in zip(reference, corpus, strict=True))
    reference_spread = math.sqrt(sum((ref - mean_reference) ** 2 for ref in reference))
    corpus_spread = math.sqrt(sum((cor - mean_corpus) ** 2 for cor in corpus))
    if reference_spread == 0 or corpus_spread == 0:
        raise ValidationInputError("Pearson r is undefined when one series has no variance.")
    return covariance / (reference_spread * corpus_spread)


def bland_altman(reference, corpus, limit_multiplier=1.96):
    """Bland-Altman agreement: mean difference and limits of agreement.

    Returns the per-pair ``means`` and ``differences`` too, so the plotting
    layer never has to recompute them.
    """
    reference, corpus = paired_values(reference, corpus)
    differences = [cor - ref for ref, cor in zip(reference, corpus, strict=True)]
    means = [(ref + cor) / 2 for ref, cor in zip(reference, corpus, strict=True)]
    n = len(differences)
    mean_difference = sum(differences) / n
    if n > 1:
        variance = sum((value - mean_difference) ** 2 for value in differences) / (n - 1)
    else:
        variance = 0.0
    std_difference = math.sqrt(variance)
    return {
        "n": n,
        "mean_difference": mean_difference,
        "std_difference": std_difference,
        "lower_limit": mean_difference - limit_multiplier * std_difference,
        "upper_limit": mean_difference + limit_multiplier * std_difference,
        "limit_multiplier": limit_multiplier,
        "means": means,
        "differences": differences,
    }


def agreement_metrics(reference, corpus, label=""):
    """The full metric block reported for one quantity."""
    reference, corpus = paired_values(reference, corpus)
    n = len(reference)
    mean_reference = sum(reference) / n
    altman = bland_altman(reference, corpus)
    metrics = {
        "quantity": label,
        "n": n,
        "reference_mean": mean_reference,
        "corpus_mean": sum(corpus) / n,
        "mae": mean_absolute_error(reference, corpus),
        "rmse": root_mean_square_error(reference, corpus),
        "mean_bias": mean_bias(reference, corpus),
        "bland_altman": {key: value for key, value in altman.items() if key not in ("means", "differences")},
    }
    metrics["mae_percent_of_reference_mean"] = (
        100 * metrics["mae"] / mean_reference if mean_reference else None
    )
    metrics["bias_percent_of_reference_mean"] = (
        100 * metrics["mean_bias"] / mean_reference if mean_reference else None
    )
    # These three are undefined for some inputs; report None rather than
    # failing the whole run, and say so in the report.
    for key, function in (
        ("mean_relative_error", mean_relative_error),
        ("r_squared_identity", r_squared_identity),
        ("pearson_r", pearson_r),
    ):
        try:
            metrics[key] = function(reference, corpus)
        except ValidationInputError as error:
            metrics[key] = None
            metrics.setdefault("undefined", {})[key] = str(error)
    return metrics


def count_agreement(reference_count, corpus_count, matched_count):
    """Detection counts, reported explicitly so recall is never implied.

    ``recall`` is matched/reference; ``precision`` is matched/corpus. Extra
    Corpus detections lower precision instead of vanishing.
    """
    if matched_count > min(reference_count, corpus_count):
        raise ValidationInputError(
            f"matched_count ({matched_count}) cannot exceed reference ({reference_count}) "
            f"or corpus ({corpus_count}) counts."
        )
    return {
        "reference_particles": reference_count,
        "corpus_particles": corpus_count,
        "matched_particles": matched_count,
        "missed_by_corpus": reference_count - matched_count,
        "extra_in_corpus": corpus_count - matched_count,
        "recall": matched_count / reference_count if reference_count else None,
        "precision": matched_count / corpus_count if corpus_count else None,
        "count_ratio": corpus_count / reference_count if reference_count else None,
    }
