"""Image reading that tolerates the formats microscopes actually emit."""

import cv2
import numpy as np

from corpus.errors import ImageReadError

__all__ = ["read_image", "read_image_or_raise"]


def read_image(image_path):
    """BGR image, or ``None`` when neither OpenCV nor Pillow can decode it.

    OpenCV is tried first; Pillow covers the 16-bit and palletised TIFFs that
    ``cv2.imread`` returns ``None`` for.
    """
    image = cv2.imread(str(image_path))
    if image is not None:
        return image

    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def read_image_or_raise(image_path):
    """Like :func:`read_image` but raises :class:`ImageReadError`."""
    image = read_image(image_path)
    if image is None:
        raise ImageReadError(f"Could not read image: {image_path}")
    return image
