"""Run provenance: checksums and a fingerprint for reproducibility.

Two runs that share a ``run_fingerprint`` were given the same image bytes and
the same settings, so they must produce the same measurements. That turns
"Corpus is reproducible" from a claim into something a test can assert.
"""

import hashlib
import json
from pathlib import Path

__all__ = ["file_sha256", "settings_fingerprint", "run_fingerprint"]

#: Settings that actually change the numbers. Presentation-only options such as
#: ``review_view`` are deliberately excluded so a display change is not
#: mistaken for a different measurement.
FINGERPRINT_KEYS = (
    "mode",
    "shape_preset",
    "scale_nm",
    "manual_scale_px",
    "manual_scale_line",
    "exclude_edges",
    "watershed",
    "watershed_min_distance_factor",
    "au_min_radius",
    "au_max_radius",
    "sio2_min_radius",
    "sio2_max_radius",
    "contrast_strategy",
    "manual_threshold_min",
    "manual_threshold_max",
    "min_circularity",
    "max_circularity",
    "min_elongation",
    "max_elongation",
    "include_holes",
)


def file_sha256(path, chunk_size=1 << 20):
    """SHA-256 of a file, streamed so large micrographs stay cheap."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def settings_fingerprint(settings):
    """SHA-256 over the measurement-relevant settings only."""
    canonical = {key: settings.get(key) for key in FINGERPRINT_KEYS if key in settings}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_fingerprint(image_path, settings, corpus_version):
    """Combined image + settings + version identity for one measurement run."""
    return {
        "corpus_version": corpus_version,
        "image_sha256": file_sha256(image_path),
        "settings_sha256": settings_fingerprint(settings),
    }
