"""
experiment.py
-------------
Orchestrates the full factorial experiment:

    imbalance ratio  x  repetition (minority sub-sample)  x  handling method

for each model family. Every run is evaluated on the SAME untouched test set,
and the decision threshold is fixed at 0.5 unless the method is threshold
moving, in which case it is tuned on the untouched validation set.
"""

import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CFG, Config
from .data import ImbalanceScenarioBuilder
from .evaluation import Evaluator
from .imbalance import ThresholdOptimizer, make_strategy
from .models.base import BaseClassifier
from .models.classical import FeatureTransformer

# Type of a function that turns a list of image paths into a model-ready array.
ArrayGetter = Callable[[Sequence[str]], np.ndarray]


class ExperimentRunner:
    """Runs and records experiments for any BaseClassifier implementation."""

    def __init__(self, split_df: pd.DataFrame, config: Config = CFG):
        self.cfg = config
        self.train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
        self.val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
        self.test_df = split_df[split_df["split"] == "test"].reset_index(drop=True)
        self.builder = ImbalanceScenarioBuilder(self.train_df)
        self.rows: List[dict] = []
        self.probabilities: Dict[tuple, np.ndarray] = {}
        self.histories: Dict[tuple, dict] = {}     # neural learning curves
        self.y_test = self.test_df["label"].to_numpy()
        self.y_val = self.val_df["label"].to_numpy()

    # ------------------------------------------------------------------ core
    def run(self, tag: str, model_factory: Callable[[int], BaseClassifier],
            get_array: ArrayGetter, methods: Sequence[str],
            ratios: Iterable = None, repetitions: Optional[int] = None,
            use_pca: bool = False, augmented_lookup: Optional[Callable] = None,
            verbose: bool = True) -> pd.DataFrame:
        """Run every (ratio, repetition, method) combination for one model.

        Parameters
        ----------
        tag            label of this experiment block (e.g. "main", "ablation")
        model_factory  function seed -> fresh, untrained BaseClassifier
        get_array      function list_of_paths -> feature/image array
        use_pca        True for classical models (scaler + PCA before resampling)
        augmented_lookup  function (path, rng) -> augmented sample, required by
                          augmentation_oversampling for pre-computed features
        """
        ratios = ratios if ratios is not None else self.cfg.imbalance_ratios
        repetitions = repetitions or self.cfg.repetitions
        X_val_raw = get_array(self.val_df["path"].tolist())
        X_test_raw = get_array(self.test_df["path"].tolist())
        new_rows = []

        for ratio in ratios:
            for rep in range(repetitions):
                seed = 1000 + rep
                subset = self.builder.build(ratio, seed)
                ids = subset["path"].to_numpy()
                X_tr = get_array(ids.tolist())
                y_tr = subset["label"].to_numpy()
                X_val, X_test = X_val_raw, X_test_raw
                if use_pca:
                    # Feature space learned from real training images only.
                    transformer = FeatureTransformer(seed=seed).fit(X_tr)
                    X_tr, X_val, X_test = (transformer.transform(a)
                                           for a in (X_tr, X_val_raw, X_test_raw))
                baseline_cache = None
                # "none" must run first because threshold moving reuses it.
                ordered = sorted(methods, key=lambda m: m != "none")
                for method in ordered:
                    t0 = time.time()
                    rng = np.random.RandomState(seed)
                    strategy = make_strategy(method, augmented_lookup)
                    if strategy.tunes_threshold and baseline_cache is not None:
                        prob_val, prob_test, n_fit = baseline_cache
                    else:
                        X_fit, y_fit = strategy.resample(X_tr, y_tr, rng, ids)
                        model = model_factory(seed)
                        model.fit(X_fit, y_fit, class_weight=strategy.class_weight(y_fit),
                                  X_val=X_val, y_val=self.y_val)
                        prob_val = model.predict_proba(X_val)
                        prob_test = model.predict_proba(X_test)
                        n_fit = len(y_fit)
                        if getattr(model, "history", None):
                            self.histories[(tag, model.name, str(ratio), rep, method)] = model.history
                        if method == "none":
                            baseline_cache = (prob_val, prob_test, n_fit)
                    threshold = (ThresholdOptimizer().fit(self.y_val, prob_val)
                                 if strategy.tunes_threshold else 0.5)
                    test_m = Evaluator.compute(self.y_test, prob_test, threshold)
                    val_m = Evaluator.compute(self.y_val, prob_val, threshold)
                    stats = ImbalanceScenarioBuilder.describe(subset)
                    row = {"tag": tag, "model": model_factory(seed).name,
                           "ratio": str(ratio), "rep": rep, "seed": seed,
                           "method": method, "family": strategy.family,
                           **stats, "n_fit": n_fit,
                           "seconds": round(time.time() - t0, 2),
                           **{f"val_{k}": v for k, v in val_m.items()
                              if k in ("balanced_accuracy", "roc_auc", "f1_defective")},
                           **test_m}
                    new_rows.append(row)
                    self.probabilities[(tag, row["model"], str(ratio), rep, method)] = prob_test
                    if verbose:
                        print(f"[{tag}] {row['model']:12s} ratio={str(ratio):7s} rep={rep} "
                              f"{method:26s} BA={test_m['balanced_accuracy']:.3f} "
                              f"Rec_def={test_m['recall_defective']:.3f} "
                              f"F1={test_m['f1_defective']:.3f} t={row['seconds']}s",
                              flush=True)
        self.rows.extend(new_rows)
        return pd.DataFrame(new_rows)

    def results(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def summarise(results: pd.DataFrame,
              metrics: Sequence[str] = ("balanced_accuracy", "recall_defective",
                                        "recall_ok", "precision_defective",
                                        "f1_defective", "g_mean", "mcc",
                                        "roc_auc", "pr_auc", "accuracy"),
              by: Sequence[str] = ("tag", "model", "ratio", "method")) -> pd.DataFrame:
    """Mean and standard deviation of each metric across repetitions."""
    grouped = results.groupby(list(by), sort=False)[list(metrics)]
    mean, std = grouped.mean(), grouped.std(ddof=1).fillna(0.0)
    out = mean.copy()
    for m in metrics:
        out[f"{m}_sd"] = std[m]
    return out.reset_index()
