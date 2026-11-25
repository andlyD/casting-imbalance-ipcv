"""
classical.py
------------
Classical machine-learning classifiers operating on handcrafted features.

    FeatureTransformer  - StandardScaler followed by PCA (95 % variance).
    SVMClassifier       - RBF-kernel Support Vector Machine (primary model).
    RandomForestModel   - Random Forest (secondary classical comparison).
    LogisticRegressionModel - L2-regularised linear model (simple baseline).
    KNNModel            - k-Nearest Neighbours (instance-based baseline).

The transformer is fitted on the *original* (not resampled) training subset so
that synthetic or duplicated samples do not distort the learned feature space;
resampling is then performed in the reduced PCA space.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import CFG, Config
from .base import BaseClassifier


class FeatureTransformer:
    """Standardise each feature, then project onto principal components."""

    def __init__(self, variance: float = CFG.pca_variance, seed: int = 42):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance, svd_solver="full", random_state=seed)

    def fit(self, X: np.ndarray) -> "FeatureTransformer":
        self.pca.fit(self.scaler.fit_transform(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)


class SVMClassifier(BaseClassifier):
    """RBF Support Vector Machine.

    The SVM returns a signed distance to the separating hyperplane. To share
    one threshold convention with the other models, the margin d is mapped
    through the logistic function s = 1 / (1 + exp(-d)); this is monotonic, so
    ROC-AUC is unchanged and the default threshold 0.5 equals the SVM's own
    decision boundary d = 0. The value is a score, not a calibrated probability.
    """

    name = "SVM"

    def __init__(self, C: float = 10.0, gamma="scale", seed: int = 42):
        self.C, self.gamma, self.seed = C, gamma, seed
        self.model: Optional[SVC] = None

    def fit(self, X, y, class_weight=None, X_val=None, y_val=None):
        self.model = SVC(C=self.C, kernel="rbf", gamma=self.gamma,
                         class_weight=class_weight, random_state=self.seed)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        margin = self.model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-margin))


class RandomForestModel(BaseClassifier):
    """Random Forest of fully grown decision trees; outputs vote fractions."""

    name = "RandomForest"

    def __init__(self, n_estimators: int = CFG.rf_estimators, seed: int = 42):
        self.n_estimators, self.seed = n_estimators, seed
        self.model: Optional[RandomForestClassifier] = None

    def fit(self, X, y, class_weight=None, X_val=None, y_val=None):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, max_features="sqrt",
            class_weight=class_weight, n_jobs=-1, random_state=self.seed)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class LogisticRegressionModel(BaseClassifier):
    """L2-regularised logistic regression; a linear decision boundary."""

    name = "LogReg"

    def __init__(self, C: float = 1.0, seed: int = 42):
        self.C, self.seed = C, seed
        self.model: Optional[LogisticRegression] = None

    def fit(self, X, y, class_weight=None, X_val=None, y_val=None):
        self.model = LogisticRegression(C=self.C, class_weight=class_weight,
                                        max_iter=5000, random_state=self.seed)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class KNNModel(BaseClassifier):
    """Distance-weighted k-Nearest Neighbours.

    scikit-learn's k-NN has no ``class_weight`` argument, so cost-sensitive
    learning is applied at prediction time: the neighbour-vote fractions are
    multiplied by the class weights and renormalised. With weights
    N / (2 * N_c) this is equivalent to dividing each vote by its class prior.
    """

    name = "KNN"

    def __init__(self, k: int = 5, seed: int = 42):
        self.k, self.seed = k, seed
        self.model: Optional[KNeighborsClassifier] = None
        self.class_weight: Optional[Dict[int, float]] = None

    def fit(self, X, y, class_weight=None, X_val=None, y_val=None):
        k = min(self.k, len(y))
        self.model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        self.model.fit(X, y)
        self.class_weight = class_weight
        return self

    def predict_proba(self, X):
        votes = self.model.predict_proba(X)
        if votes.shape[1] == 1:                  # only one class seen in training
            return np.full(len(X), float(self.model.classes_[0]))
        if self.class_weight:
            w = np.array([self.class_weight[0], self.class_weight[1]])
            votes = votes * w
            votes = votes / votes.sum(axis=1, keepdims=True)
        return votes[:, 1]
