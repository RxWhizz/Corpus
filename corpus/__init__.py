"""Corpus scientific core.

This package holds the measurement logic that must be testable without
Electron, without a display, and without any machine-learning dependency:

- :mod:`corpus.calibration` -- scale-bar detection and nm/pixel resolution
- :mod:`corpus.segmentation` -- binarisation, watershed separation, backends
- :mod:`corpus.measurement` -- contour geometry and per-particle records
- :mod:`corpus.metrology` -- core-shell derivations and distribution summaries
- :mod:`corpus.review` -- confidence scoring, review status, overlays
- :mod:`corpus.io` -- image reading and legacy export files
- :mod:`corpus.validation` -- Corpus vs. reference agreement metrics

``measurement_modes.py`` at the repository root remains the CLI contract used
by the Electron main process and re-exports these names for compatibility.
"""

from corpus.errors import (
    CalibrationError,
    CorpusError,
    ImageReadError,
    ValidationInputError,
)

__all__ = [
    "CorpusError",
    "CalibrationError",
    "ImageReadError",
    "ValidationInputError",
    "__version__",
]

__version__ = "1.0.0"
