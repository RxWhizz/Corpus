"""Segmentation: contrast strategies, watershed separation, backend contract."""

from corpus.segmentation.backends import (
    ClassicalBackend,
    ManualBackend,
    SegmentationBackend,
    SegmentationResult,
    available_backends,
    get_backend,
)
from corpus.segmentation.binarize import (
    CONTRAST_STRATEGIES,
    particle_binary,
    sio2_mask,
    to_gray,
)
from corpus.segmentation.watershed import watershed_split_contours

__all__ = [
    "CONTRAST_STRATEGIES",
    "ClassicalBackend",
    "ManualBackend",
    "SegmentationBackend",
    "SegmentationResult",
    "available_backends",
    "get_backend",
    "particle_binary",
    "sio2_mask",
    "to_gray",
    "watershed_split_contours",
]
