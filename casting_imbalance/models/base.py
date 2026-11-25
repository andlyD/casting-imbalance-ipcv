"""
base.py
-------
Abstract interface shared by every classifier in the project.

Using one interface (fit / predict_proba / save / load) lets the experiment
runner, the evaluator and the GUI application treat an SVM, a Random Forest,
a MobileNetV2 head and a from-scratch CNN polymorphically.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class BaseClassifier(ABC):
    """Binary classifier returning P(Defective) for each input sample."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            class_weight: Optional[Dict[int, float]] = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "BaseClassifier":
        """Train the model. ``class_weight`` implements cost-sensitive learning."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return a 1-D array of scores in [0, 1]; higher = more likely defective."""

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
