"""The segmentation backend contract (Workstream I1).

Corpus v1.0 ships exactly one implementation, :class:`ClassicalBackend`, which
has no machine-learning dependency. The point of this module is that a future
YOLO-seg or SAM backend can be dropped in *without touching the measurement
contract*: everything downstream consumes :class:`SegmentationResult`.

The hybrid rule is enforced by data, not by convention -- every result carries
the ``backend`` that produced it and a ``review_required`` flag, so a
measurement can always be traced back to how its mask was obtained::

    AI proposes -> Corpus displays -> researcher reviews -> accepted measurement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2

from corpus.segmentation.binarize import particle_binary, sio2_mask, to_gray
from corpus.segmentation.watershed import watershed_split_contours

__all__ = [
    "SegmentationResult",
    "SegmentationBackend",
    "ClassicalBackend",
    "ManualBackend",
    "available_backends",
    "get_backend",
]


@dataclass
class SegmentationResult:
    """Masks proposed by a backend, plus the provenance of that proposal.

    Attributes
    ----------
    contours:
        OpenCV contours in pixel coordinates.
    backend:
        Identifier of the producing backend, recorded on every measurement.
    method:
        How contours were separated (``"contour"``, ``"watershed"``, ...).
    review_required:
        ``True`` whenever a human must confirm before the numbers are trusted.
        Any non-classical backend should leave this ``True``.
    scores:
        Optional per-contour confidence, aligned with ``contours``.
    metadata:
        Free-form backend detail (model name, weights checksum, thresholds).
    """

    contours: list = field(default_factory=list)
    backend: str = "unknown"
    method: str = "contour"
    review_required: bool = True
    scores: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.scores and len(self.scores) != len(self.contours):
            raise ValueError("SegmentationResult.scores must align with contours.")

    def as_provenance(self):
        """Dict merged into ``measurements.json`` for traceability."""
        return {
            "backend": self.backend,
            "separation_method": self.method,
            "review_required": self.review_required,
            "backend_metadata": dict(self.metadata),
        }


class SegmentationBackend(ABC):
    """Stable interface every segmentation source must implement."""

    #: Short identifier recorded on each measurement.
    name = "abstract"

    #: Set to ``True`` by backends that import a machine-learning runtime.
    requires_ml = False

    @abstractmethod
    def predict(self, image, **options):
        """Return a :class:`SegmentationResult` for a BGR ``image``."""

    def describe(self):
        """Human-readable capability summary for the UI and for reports."""
        return {"name": self.name, "requires_ml": self.requires_ml, "class": type(self).__name__}


class ClassicalBackend(SegmentationBackend):
    """Threshold + morphology + optional watershed. No ML dependencies.

    This is the v1.0 default and the reference every future backend is
    compared against in the validation harness.
    """

    name = "classical"
    requires_ml = False

    def __init__(self, target="particles", default_threshold=80):
        if target not in ("particles", "carriers"):
            raise ValueError("target must be 'particles' or 'carriers'.")
        self.target = target
        self.default_threshold = default_threshold

    def predict(self, image, settings=None, watershed=False, watershed_factor=0.55,
                min_radius_px=1, max_radius_px=10_000, **_options):
        settings = settings or {}
        if self.target == "carriers":
            binary = sio2_mask(image, settings.get("include_holes", False))
        else:
            binary = particle_binary(to_gray(image), settings, self.default_threshold)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        method = "contour"
        if watershed:
            split = watershed_split_contours(binary, min_radius_px, max_radius_px, watershed_factor)
            if len(split) > len(contours):
                contours = split
                method = "watershed"

        return SegmentationResult(
            contours=list(contours),
            backend=self.name,
            method=method,
            # Classical output is deterministic and inspectable, so it is not
            # gated on review the way a model proposal is.
            review_required=False,
            metadata={
                "target": self.target,
                "contrast_strategy": settings.get("contrast_strategy", "dark_particles"),
                "watershed": watershed,
            },
        )


class ManualBackend(SegmentationBackend):
    """Contours supplied by a human (imported annotation or GUI drawing)."""

    name = "manual"
    requires_ml = False

    def predict(self, image, contours=None, **_options):
        return SegmentationResult(
            contours=list(contours or []),
            backend=self.name,
            method="manual",
            review_required=False,
            metadata={"source": "human_annotation"},
        )


#: Backends available in this build. A YOLOSegBackend registers here once the
#: dataset in Workstream H reaches a trainable size.
_REGISTRY = {
    ClassicalBackend.name: ClassicalBackend,
    ManualBackend.name: ManualBackend,
}


def available_backends():
    """Names of registered backends."""
    return sorted(_REGISTRY)


def get_backend(name, **kwargs):
    """Instantiate a registered backend by name."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown segmentation backend {name!r}. Available: {available_backends()}") from None
    return factory(**kwargs)
