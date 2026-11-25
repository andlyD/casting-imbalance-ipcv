"""
preprocessing.py
----------------
Image loading and the image-processing variants compared in the study.

Pipeline variants (selected with ``mode``):
    "resize"          grayscale -> resize
    "gaussian"        grayscale -> resize -> 3x3 Gaussian blur   (default)
    "clahe"           grayscale -> resize -> CLAHE contrast enhancement
    "gaussian_clahe"  grayscale -> resize -> Gaussian blur -> CLAHE
"""

from functools import lru_cache
from typing import Iterable

import cv2
import numpy as np

from .config import CFG, Config

VALID_MODES = ("resize", "gaussian", "clahe", "gaussian_clahe")


@lru_cache(maxsize=4096)
def load_grayscale(path: str) -> np.ndarray:
    """Read an image from disk as 8-bit grayscale (cached in memory).

    The casting images are stored as 3-channel JPEGs whose channels are
    identical, so converting to one channel loses no information.
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"Could not read image: {path}")
    return image


class ImagePreprocessor:
    """Applies one of the image-processing variants to a grayscale image."""

    def __init__(self, size: int, mode: str = "gaussian", config: Config = CFG):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        self.size = size
        self.mode = mode
        self.cfg = config
        self._clahe = cv2.createCLAHE(clipLimit=config.clahe_clip,
                                      tileGridSize=(config.clahe_grid,
                                                    config.clahe_grid))

    def process(self, image: np.ndarray) -> np.ndarray:
        """Return a uint8 image of shape (size, size)."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # INTER_AREA is the recommended interpolation when shrinking images
        # because it averages pixels and limits aliasing of fine defects.
        out = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.mode in ("gaussian", "gaussian_clahe"):
            k = self.cfg.gaussian_kernel
            out = cv2.GaussianBlur(out, (k, k), 0)
        if self.mode in ("clahe", "gaussian_clahe"):
            out = self._clahe.apply(out)
        return out

    def process_path(self, path: str) -> np.ndarray:
        return self.process(load_grayscale(path))

    def process_many(self, paths: Iterable[str]) -> np.ndarray:
        return np.stack([self.process_path(p) for p in paths])
