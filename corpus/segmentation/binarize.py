"""Contrast strategies that turn a grey TEM frame into a particle mask."""

import cv2
import numpy as np

__all__ = ["CONTRAST_STRATEGIES", "particle_binary", "sio2_mask", "to_gray"]

#: Strategies exposed in the Segmentation Assist panel.
CONTRAST_STRATEGIES = ("dark_particles", "bright_shells", "manual_gray_range")

BRIGHT_SHELL_THRESHOLD = 185


def to_gray(image):
    """Single-channel view of ``image``; a grey input is returned unchanged."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def particle_binary(gray, settings, default_threshold=80):
    """Binary particle mask for the configured contrast strategy.

    ``dark_particles`` (the default) keeps everything darker than
    ``default_threshold``; ``bright_shells`` keeps bright halos;
    ``manual_gray_range`` keeps an explicit grey window.
    """
    strategy = settings.get("contrast_strategy", "dark_particles")
    if strategy == "manual_gray_range":
        lo = int(settings.get("manual_threshold_min", 0))
        hi = int(settings.get("manual_threshold_max", default_threshold))
        return cv2.inRange(gray, lo, hi)
    if strategy == "bright_shells":
        _, binary = cv2.threshold(gray, BRIGHT_SHELL_THRESHOLD, 255, cv2.THRESH_BINARY)
        return binary
    _, binary = cv2.threshold(gray, default_threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def sio2_mask(image, include_holes=False):
    """Adaptive mask for large low-contrast carriers such as SiO2 spheres.

    ``include_holes`` uses a smaller closing kernel so internal voids survive
    instead of being filled in.
    """
    gray = cv2.GaussianBlur(to_gray(image), (7, 7), 0)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 3)
    kernel_size = 3 if include_holes else 9
    return cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
