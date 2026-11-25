"""
config.py
---------
Central configuration for the casting-defect class-imbalance study.

Every tunable value used anywhere in the project is defined here so that the
experiment can be reproduced exactly by changing a single file. The values are
grouped in a frozen dataclass to prevent accidental modification at run time.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_data_dir(root: Path = PROJECT_ROOT) -> Path:
    """Locate the folder that holds the unaugmented def_front/ and ok_front/.

    Order of preference:
      1. the CASTING_DATA_DIR environment variable, if set;
      2. data/casting_512x512 (the documented layout);
      3. any folder under data/ that contains both class folders, e.g. the
         nested data/casting_512x512/casting_512x512 created by the Kaggle zip.
    The pre-augmented casting_data/train|test folders are deliberately skipped.
    """
    env = os.environ.get("CASTING_DATA_DIR")
    if env:
        return Path(env)
    default = root / "data" / "casting_512x512"
    if (default / "ok_front").is_dir() and (default / "def_front").is_dir():
        return default
    data_root = root / "data"
    if data_root.is_dir():
        candidates = [p.parent for p in data_root.rglob("ok_front")
                      if (p.parent / "def_front").is_dir()
                      and "casting_data" not in p.parts]
        candidates.sort(key=lambda d: ("512" not in str(d), len(d.parts)))
        if candidates:
            return candidates[0]
    return default


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = find_data_dir()
    output_dir: Path = project_root / "outputs"
    model_dir: Path = project_root / "models"
    # Local copy of the ImageNet MobileNetV2 weights (no classification top).
    # If the file is missing, Keras downloads "imagenet" weights automatically.
    mobilenet_weights: Path = project_root / "weights" / \
        "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"

    # ------------------------------------------------------------ class labels
    # Defective is the POSITIVE class (label 1) because missing a defect is the
    # costly error in quality inspection.
    class_folders: Tuple[str, str] = ("ok_front", "def_front")
    class_names: Tuple[str, str] = ("OK", "Defective")

    # ----------------------------------------------------------------- split
    global_seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    # ------------------------------------------------------ imbalance design
    # Imbalance ratio = number of OK images / number of Defective images in the
    # TRAINING set. "natural" keeps the original training split unchanged.
    imbalance_ratios: Tuple = ("natural", 3, 10, 20)
    repetitions: int = 5            # random minority sub-samples per ratio
    cnn_repetitions: int = 3        # fewer for the from-scratch CNN (CPU cost)

    # ------------------------------------------------------- preprocessing
    classical_size: int = 128       # handcrafted-feature working resolution
    cnn_size: int = 128             # from-scratch CNN input resolution
    mobilenet_size: int = 224       # MobileNetV2 native resolution
    gaussian_kernel: int = 3
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    # ---------------------------------------------------- handcrafted features
    hog_orientations: int = 9
    hog_pixels_per_cell: int = 8
    hog_cells_per_block: int = 2
    lbp_points: int = 8
    lbp_radius: int = 1
    lbp_grid: int = 4               # 4 x 4 spatial grid of LBP histograms
    glcm_levels: int = 32
    glcm_distances: Tuple[int, ...] = (1, 2, 4)
    pca_variance: float = 0.95

    # ------------------------------------------------- classical models
    svm_c_grid: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    svm_gamma_grid: Tuple = ("scale", 1e-4, 1e-5)
    rf_estimators: int = 300

    # ------------------------------------------------------ neural models
    batch_size: int = 32
    max_epochs: int = 40
    patience: int = 6
    learning_rate: float = 1e-3
    n_augmented_copies: int = 10    # augmented copies per image (MobileNetV2)
    cnn_max_epochs: int = 35        # the from-scratch CNN learns more slowly
    cnn_patience: int = 8
    cnn_learning_rate: float = 5e-4

    # --------------------------------------------------- handling methods
    classical_methods: Tuple[str, ...] = (
        "none", "class_weight", "random_oversampling", "smote",
        "random_undersampling", "threshold_moving")
    mobilenet_methods: Tuple[str, ...] = (
        "none", "class_weight", "random_oversampling", "smote",
        "random_undersampling", "augmentation_oversampling",
        "threshold_moving")
    cnn_methods: Tuple[str, ...] = (
        "none", "class_weight", "augmentation_oversampling",
        "threshold_moving")

    def ensure_dirs(self) -> None:
        """Create output folders if they do not already exist."""
        for d in (self.output_dir, self.model_dir,
                  self.output_dir / "figures", self.output_dir / "tables"):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()
