"""Synthetic fixtures for the Corpus test suite.

Every fixture is generated in-process from a seeded RNG. That keeps the suite
runnable in CI with no private TEM images, and it means each fixture carries
exact ground truth we can assert against instead of a golden file nobody can
re-derive.

Convention: images are 8-bit BGR with a light background and dark particles,
matching the ``dark_particles`` contrast strategy that Corpus defaults to.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# The repo root holds the `corpus` package and `measurement_modes.py`;
# `training/` uses flat sibling imports (`from common_training import ...`),
# so it needs its own entry.
REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "training"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

BACKGROUND = 235
PARTICLE_GRAY = 40
SHELL_GRAY = 150


def _canvas(height=400, width=400, background=BACKGROUND):
    return np.full((height, width, 3), background, dtype=np.uint8)


def _disc(image, center, radius, gray):
    cv2.circle(image, center, radius, (gray, gray, gray), -1)


@pytest.fixture
def blank_image():
    """Featureless frame -- nothing should ever be detected here."""
    return _canvas()


@pytest.fixture
def single_particle():
    """One isolated dark disc well away from every edge."""
    image = _canvas()
    center, radius = (200, 200), 40
    _disc(image, center, radius, PARTICLE_GRAY)
    return {"image": image, "center": center, "radius_px": radius, "count": 1}


@pytest.fixture
def two_touching_particles():
    """Two equal discs overlapping slightly -- the watershed test case.

    Centres are 2*r - 8 apart, so a plain contour finder sees one blob.
    """
    image = _canvas()
    radius = 38
    gap = 2 * radius - 8
    left = (200 - gap // 2, 200)
    right = (200 + gap // 2, 200)
    _disc(image, left, radius, PARTICLE_GRAY)
    _disc(image, right, radius, PARTICLE_GRAY)
    return {"image": image, "centers": [left, right], "radius_px": radius, "count": 2}


@pytest.fixture
def core_shell_particle():
    """Concentric dark core inside a mid-grey shell, with exact ground truth."""
    image = _canvas()
    center = (200, 200)
    core_radius, outer_radius = 20, 50
    _disc(image, center, outer_radius, SHELL_GRAY)
    _disc(image, center, core_radius, PARTICLE_GRAY)
    return {
        "image": image,
        "center": center,
        "core_radius_px": core_radius,
        "outer_radius_px": outer_radius,
        "core_diameter_px": 2 * core_radius,
        "outer_diameter_px": 2 * outer_radius,
        "shell_thickness_px": outer_radius - core_radius,
    }


@pytest.fixture
def elongated_particle():
    """A rod with a 3:1 aspect ratio -- must classify as ``elongated``."""
    image = _canvas()
    center, axes, angle = (200, 200), (75, 25), 0
    cv2.ellipse(image, center, axes, angle, 0, 360, (PARTICLE_GRAY,) * 3, -1)
    return {
        "image": image,
        "center": center,
        "major_px": 2 * axes[0],
        "minor_px": 2 * axes[1],
        "expected_aspect_ratio": axes[0] / axes[1],
    }


@pytest.fixture
def scale_bar_image():
    """Frame with a printed white scale bar of known pixel length.

    The bar sits low and right where TEM software prints it, so the
    confidence heuristic should rank it first.
    """
    image = _canvas(height=400, width=600, background=90)
    bar = {"x": 420, "y": 360, "width": 120, "height": 6}
    cv2.rectangle(
        image,
        (bar["x"], bar["y"]),
        (bar["x"] + bar["width"], bar["y"] + bar["height"]),
        (255, 255, 255),
        -1,
    )
    _disc(image, (150, 150), 40, PARTICLE_GRAY)
    return {"image": image, "bar": bar, "scale_nm": 240.0, "expected_nm_per_px": 240.0 / 120}


@pytest.fixture
def edge_cut_particle():
    """A particle clipped by the left frame edge -- must be flagged ``edge``."""
    image = _canvas()
    center, radius = (10, 200), 45
    _disc(image, center, radius, PARTICLE_GRAY)
    return {"image": image, "center": center, "radius_px": radius}


@pytest.fixture
def low_contrast_image():
    """Particle only ~6 grey levels below background -> ``Low contrast``."""
    image = _canvas(background=128)
    _disc(image, (200, 200), 40, 122)
    return {"image": image}


@pytest.fixture
def noisy_particles():
    """Several discs plus Gaussian noise, from a fixed seed."""
    rng = np.random.default_rng(20260828)
    image = _canvas(height=300, width=300)
    centers = [(70, 70), (220, 80), (90, 220), (230, 230)]
    for center in centers:
        _disc(image, center, 25, PARTICLE_GRAY)
    noise = rng.normal(0, 5, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return {"image": image, "centers": centers, "radius_px": 25, "count": len(centers)}


@pytest.fixture
def coco_annotation(tmp_path):
    """Minimal two-image, two-class COCO file with polygon segmentation.

    ``image_2`` deliberately shares ``source_id`` with ``image_1`` so leakage
    checks have something to catch.
    """

    def polygon(cx, cy, r, points=12):
        coords = []
        for index in range(points):
            theta = 2 * math.pi * index / points
            coords.extend([cx + r * math.cos(theta), cy + r * math.sin(theta)])
        return [round(value, 2) for value in coords]

    payload = {
        "images": [
            {"id": 1, "file_name": "image_1.png", "width": 200, "height": 200,
             "source_id": "group_a", "nm_per_px": 0.5,
             "metadata": {"license": "CC0-1.0", "license_status": "accepted", "source_id": "group_a"}},
            {"id": 2, "file_name": "image_2.png", "width": 200, "height": 200,
             "source_id": "group_a", "nm_per_px": 0.5,
             "metadata": {"license": "CC0-1.0", "license_status": "accepted", "source_id": "group_a"}},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [polygon(100, 100, 20)],
             "area": 1256.0, "bbox": [80, 80, 40, 40], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "segmentation": [polygon(100, 100, 50)],
             "area": 7854.0, "bbox": [50, 50, 100, 100], "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "segmentation": [polygon(60, 60, 15)],
             "area": 706.0, "bbox": [45, 45, 30, 30], "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "Au_core"},
            {"id": 2, "name": "SiO2_outer"},
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {"path": path, "payload": payload}


@pytest.fixture
def default_filters():
    """Filter settings as produced by ``clamp_filter_settings`` defaults."""
    from corpus.measurement import DEFAULT_FILTER_SETTINGS

    return dict(DEFAULT_FILTER_SETTINGS)
