"""
features.py
-----------
Handcrafted feature extraction for the classical machine-learning pathway.

Three complementary descriptors are concatenated into one feature vector:

    HOG   (Histogram of Oriented Gradients)  - edge / contour shape, sensitive to
          burrs, broken edges and irregular blade outlines.
    LBP   (uniform Local Binary Patterns, 4x4 spatial grid) - local micro-texture,
          sensitive to pinholes, porosity and surface roughness.
    GLCM  (Grey-Level Co-occurrence Matrix statistics) - global texture
          statistics (contrast, homogeneity, energy, ...), as in Tutorial 3.
"""

from typing import Dict, Iterable, List, Sequence

import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from .config import CFG, Config
from .preprocessing import ImagePreprocessor

GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity",
                   "energy", "correlation", "ASM")
ALL_GROUPS = ("hog", "lbp", "glcm")


class HandcraftedFeatureExtractor:
    """Converts a preprocessed image into a fixed-length 1-D feature vector."""

    def __init__(self, preprocessor: ImagePreprocessor,
                 groups: Sequence[str] = ALL_GROUPS, config: Config = CFG):
        unknown = set(groups) - set(ALL_GROUPS)
        if unknown:
            raise ValueError(f"Unknown feature groups: {unknown}")
        self.pre = preprocessor
        self.groups = tuple(groups)
        self.cfg = config

    # ----------------------------------------------------------- descriptors
    def hog_features(self, img: np.ndarray) -> np.ndarray:
        c = self.cfg
        return hog(img, orientations=c.hog_orientations,
                   pixels_per_cell=(c.hog_pixels_per_cell,) * 2,
                   cells_per_block=(c.hog_cells_per_block,) * 2,
                   block_norm="L2-Hys", feature_vector=True)

    def lbp_features(self, img: np.ndarray) -> np.ndarray:
        """Uniform LBP histograms computed in a grid of equal tiles.

        A uniform LBP with P neighbours has P + 2 possible codes, so each
        tile contributes a (P + 2)-bin normalised histogram.
        """
        c = self.cfg
        lbp = local_binary_pattern(img, c.lbp_points, c.lbp_radius, "uniform")
        n_bins = c.lbp_points + 2
        tile = img.shape[0] // c.lbp_grid
        hist_list = []
        for i in range(c.lbp_grid):
            for j in range(c.lbp_grid):
                block = lbp[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]
                hist, _ = np.histogram(block, bins=n_bins, range=(0, n_bins))
                hist_list.append(hist / max(hist.sum(), 1))
        return np.concatenate(hist_list)

    def glcm_features(self, img: np.ndarray) -> np.ndarray:
        """GLCM properties averaged over 4 angles for each distance.

        Grey levels are quantised from 256 to 32 to make the matrix less
        sparse and the statistics more stable.
        """
        c = self.cfg
        quantised = (img // (256 // c.glcm_levels)).astype(np.uint8)
        glcm = graycomatrix(quantised, distances=list(c.glcm_distances),
                            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=c.glcm_levels, symmetric=True, normed=True)
        return np.concatenate([graycoprops(glcm, p).mean(axis=1)
                               for p in GLCM_PROPERTIES])

    # -------------------------------------------------------------- public
    def extract_groups(self, path: str) -> Dict[str, np.ndarray]:
        img = self.pre.process_path(path)
        return {"hog": self.hog_features(img),
                "lbp": self.lbp_features(img),
                "glcm": self.glcm_features(img)}

    def extract(self, path: str) -> np.ndarray:
        parts = self.extract_groups(path)
        return np.concatenate([parts[g] for g in self.groups]).astype(np.float32)

    def extract_many(self, paths: Iterable[str]) -> np.ndarray:
        return np.stack([self.extract(p) for p in paths])


class FeatureBank:
    """Pre-computes features for every image once and serves row subsets.

    Feature extraction is deterministic, so computing it once per image and
    preprocessing mode avoids repeating ~1,300 extractions for every one of the
    hundreds of experimental runs.
    """

    def __init__(self, paths: List[str], preprocessor: ImagePreprocessor,
                 config: Config = CFG):
        extractor = HandcraftedFeatureExtractor(preprocessor, ALL_GROUPS, config)
        self.paths = list(paths)
        self._row = {p: i for i, p in enumerate(self.paths)}
        parts = [extractor.extract_groups(p) for p in self.paths]
        self.blocks = {g: np.stack([d[g] for d in parts]).astype(np.float32)
                       for g in ALL_GROUPS}

    def get(self, paths: Iterable[str], groups: Sequence[str] = ALL_GROUPS) -> np.ndarray:
        rows = [self._row[p] for p in paths]
        return np.hstack([self.blocks[g][rows] for g in groups])

    def dimensions(self) -> Dict[str, int]:
        return {g: b.shape[1] for g, b in self.blocks.items()}
