"""
test_core.py
------------
Unit tests for the components whose correctness the study depends on.
Run from the project root with:  python -m pytest -q   (or python tests/test_core.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from casting_imbalance.data import ImbalanceScenarioBuilder
from casting_imbalance.evaluation import Evaluator
from casting_imbalance.imbalance import (ThresholdOptimizer, balanced_class_weights,
                                         make_strategy)
from casting_imbalance.preprocessing import ImagePreprocessor


def _toy_train(n_ok=363, n_def=547):
    return pd.DataFrame({"path": [f"p{i}" for i in range(n_ok + n_def)],
                         "filename": [f"f{i:04d}" for i in range(n_ok + n_def)],
                         "label": [0] * n_ok + [1] * n_def})


def test_scenario_ratio_and_ok_kept():
    builder = ImbalanceScenarioBuilder(_toy_train())
    for ratio, expected in ((3, 121), (10, 36), (20, 18)):
        d = builder.build(ratio, seed=1000)
        assert (d.label == 0).sum() == 363          # every OK image kept
        assert (d.label == 1).sum() == expected     # round(363 / ratio)


def test_scenario_seed_changes_subset_but_is_reproducible():
    builder = ImbalanceScenarioBuilder(_toy_train())
    a = set(builder.build(10, 1000).path)
    assert a == set(builder.build(10, 1000).path)
    assert a != set(builder.build(10, 1001).path)


def test_balanced_weights_equalise_class_mass():
    y = np.array([0] * 90 + [1] * 10)
    w = balanced_class_weights(y)
    assert np.isclose(w[0] * 90, w[1] * 10)


def test_resamplers_balance_classes():
    rng = np.random.RandomState(0)
    X, y = rng.rand(110, 5), np.array([0] * 100 + [1] * 10)
    for name in ("random_oversampling", "smote", "random_undersampling",
                 "augmentation_oversampling"):
        Xr, yr = make_strategy(name).resample(X, y, np.random.RandomState(1))
        assert np.bincount(yr)[0] == np.bincount(yr)[1], name


def test_threshold_optimizer_recovers_shifted_boundary():
    y = np.array([0] * 50 + [1] * 50)
    prob = np.concatenate([np.full(50, 0.05), np.full(50, 0.20)])  # all below 0.5
    t = ThresholdOptimizer().fit(y, prob)
    assert 0.05 < t <= 0.20
    assert Evaluator.compute(y, prob, t)["balanced_accuracy"] == 1.0


def test_evaluator_majority_classifier_has_chance_balanced_accuracy():
    y = np.array([0] * 95 + [1] * 5)
    m = Evaluator.compute(y, np.zeros(100))
    assert m["accuracy"] == 0.95 and m["balanced_accuracy"] == 0.5 and m["recall_defective"] == 0


def test_preprocessor_output_shape_and_type():
    img = (np.random.RandomState(0).rand(512, 512) * 255).astype(np.uint8)
    for mode in ("resize", "gaussian", "clahe", "gaussian_clahe"):
        out = ImagePreprocessor(128, mode).process(img)
        assert out.shape == (128, 128) and out.dtype == np.uint8


def test_knn_class_weight_reweights_votes():
    from casting_imbalance.models.classical import KNNModel, LogisticRegressionModel
    rs = np.random.RandomState(0)
    X = np.vstack([rs.normal(0, 1, (90, 2)), rs.normal(1.5, 1, (10, 2))])
    y = np.array([0] * 90 + [1] * 10)
    X_new = rs.normal(0.75, 1, (50, 2))          # unseen points (no zero distances)
    plain = KNNModel(k=9).fit(X, y).predict_proba(X_new)
    weighted = KNNModel(k=9).fit(X, y, class_weight={0: 100 / 180, 1: 100 / 20}).predict_proba(X_new)
    assert np.all(weighted >= plain - 1e-12) and weighted.mean() > plain.mean()
    p = LogisticRegressionModel().fit(X, y).predict_proba(X)
    assert p.shape == (100,) and 0 <= p.min() and p.max() <= 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
