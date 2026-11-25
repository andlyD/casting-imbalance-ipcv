"""
imbalance.py
------------
Class-imbalance handling strategies (Strategy design pattern).

Every strategy exposes the same interface so the experiment runner can swap
them without knowing how each one works:

    resample(X, y, rng, ids)  -> (X_new, y_new)   data-level methods
    class_weight(y)           -> dict or None     algorithm-level methods
    tunes_threshold           -> bool             post-processing method

All strategies are applied to the TRAINING data only.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import balanced_accuracy_score


def balanced_class_weights(y: np.ndarray) -> Dict[int, float]:
    """w_c = N / (K * N_c): the inverse-frequency weighting used by
    scikit-learn's class_weight="balanced"."""
    classes, counts = np.unique(y, return_counts=True)
    n, k = len(y), len(classes)
    return {int(c): n / (k * cnt) for c, cnt in zip(classes, counts)}


class ImbalanceStrategy(ABC):
    """Abstract base class for all imbalance-handling strategies."""

    name: str = "abstract"
    family: str = "none"           # data-level / algorithm-level / post-hoc
    tunes_threshold: bool = False

    def resample(self, X: np.ndarray, y: np.ndarray,
                 rng: np.random.RandomState,
                 ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        return X, y

    def class_weight(self, y: np.ndarray) -> Optional[Dict[int, float]]:
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NoHandling(ImbalanceStrategy):
    """Baseline: the model sees the imbalanced data unchanged."""
    name, family = "none", "baseline"


class ClassWeighting(ImbalanceStrategy):
    """Cost-sensitive learning: errors on the minority class cost more."""
    name, family = "class_weight", "algorithm-level"

    def class_weight(self, y):
        return balanced_class_weights(y)


class RandomOverSampling(ImbalanceStrategy):
    """Duplicate randomly chosen minority samples until classes are equal."""
    name, family = "random_oversampling", "data-level"

    def resample(self, X, y, rng, ids=None):
        return RandomOverSampler(random_state=rng.randint(1_000_000)).fit_resample(X, y)


class SMOTEOversampling(ImbalanceStrategy):
    """SMOTE: synthesise new minority vectors by interpolating between a
    minority sample and one of its k nearest minority neighbours."""
    name, family = "smote", "data-level"

    def __init__(self, k_neighbors: int = 5):
        self.k = k_neighbors

    def resample(self, X, y, rng, ids=None):
        n_minority = np.bincount(y).min()
        k = max(1, min(self.k, n_minority - 1))   # guard for tiny minorities
        return SMOTE(k_neighbors=k, random_state=rng.randint(1_000_000)).fit_resample(X, y)


class RandomUnderSampling(ImbalanceStrategy):
    """Discard randomly chosen majority samples until classes are equal."""
    name, family = "random_undersampling", "data-level"

    def resample(self, X, y, rng, ids=None):
        return RandomUnderSampler(random_state=rng.randint(1_000_000)).fit_resample(X, y)


class AugmentationOversampling(ImbalanceStrategy):
    """Oversample the minority class with *augmented* copies instead of exact
    duplicates.

    ``augmented_lookup(id, rng)`` must return one augmented version of the
    sample with identifier ``id`` (e.g. the MobileNetV2 embedding of a rotated
    and brightness-shifted image). For the end-to-end CNN the lookup is not
    needed because random augmentation is applied on the fly to every batch,
    so duplicating indices already yields different augmented images.
    """
    name, family = "augmentation_oversampling", "data-level"

    def __init__(self, augmented_lookup: Optional[Callable] = None):
        self.lookup = augmented_lookup

    def resample(self, X, y, rng, ids=None):
        classes, counts = np.unique(y, return_counts=True)
        minority = classes[np.argmin(counts)]
        deficit = counts.max() - counts.min()
        minority_rows = np.where(y == minority)[0]
        picks = rng.choice(minority_rows, deficit, replace=True)
        if self.lookup is None:
            X_extra = X[picks]
        else:
            if ids is None:
                raise ValueError("ids are required when a lookup is used")
            X_extra = np.stack([self.lookup(ids[i], rng) for i in picks])
        X_new = np.concatenate([X, X_extra])
        y_new = np.concatenate([y, np.full(deficit, minority, dtype=y.dtype)])
        return X_new, y_new


class ThresholdMoving(ImbalanceStrategy):
    """Post-hoc correction: train normally, then move the decision threshold
    to the value that maximises balanced accuracy on the validation set."""
    name, family, tunes_threshold = "threshold_moving", "post-hoc", True


class ThresholdOptimizer:
    """Grid search for the decision threshold on held-out validation scores."""

    def __init__(self, grid: Optional[np.ndarray] = None):
        self.grid = grid if grid is not None else np.round(np.arange(0.01, 1.0, 0.01), 2)

    def fit(self, y_val: np.ndarray, prob_val: np.ndarray) -> float:
        scores = np.array([balanced_accuracy_score(y_val, (prob_val >= t).astype(int))
                           for t in self.grid])
        best = np.flatnonzero(scores == scores.max())
        # Among equally good thresholds choose the one closest to 0.5 so the
        # adjustment is as small as possible.
        return float(self.grid[best[np.argmin(np.abs(self.grid[best] - 0.5))]])


def make_strategy(name: str, augmented_lookup: Optional[Callable] = None) -> ImbalanceStrategy:
    """Factory that maps a configuration string to a strategy object."""
    registry = {
        "none": NoHandling,
        "class_weight": ClassWeighting,
        "random_oversampling": RandomOverSampling,
        "smote": SMOTEOversampling,
        "random_undersampling": RandomUnderSampling,
        "threshold_moving": ThresholdMoving,
    }
    if name == "augmentation_oversampling":
        return AugmentationOversampling(augmented_lookup)
    if name not in registry:
        raise KeyError(f"Unknown strategy '{name}'")
    return registry[name]()
