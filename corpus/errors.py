"""Exception types for the Corpus scientific core.

Library code raises these instead of printing JSON and calling ``SystemExit``,
so the same functions can be used from the Electron CLI entry point, from
``pytest``, and from batch validation scripts.
"""


class CorpusError(Exception):
    """Base class for every recoverable Corpus error."""


class CalibrationError(CorpusError):
    """Raised when a pixel-to-nanometre scale cannot be established."""


class ImageReadError(CorpusError):
    """Raised when an image cannot be decoded by OpenCV or Pillow."""


class ValidationInputError(CorpusError):
    """Raised when a reference/Corpus validation table is malformed."""
