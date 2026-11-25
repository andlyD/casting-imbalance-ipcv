"""
run_classical.py
----------------
Step 2 of the pipeline: classical machine-learning experiments.

    1. Extract HOG + LBP + GLCM features for every image (once).
    2. Tune the SVM (C, gamma) on the validation set under the natural
       distribution, then FIX these values for all imbalance experiments.
    3. Main experiment: SVM and Random Forest x 4 imbalance ratios x
       5 repetitions x 6 handling methods.
    4. Ablation A: effect of the image-processing variant.
    5. Ablation B: effect of the feature group.

Run from the project root:  python scripts/run_classical.py
"""

import json
import pickle
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from casting_imbalance.config import CFG
from casting_imbalance.data import CastingDataset, DataSplitter
from casting_imbalance.evaluation import Evaluator
from casting_imbalance.experiment import ExperimentRunner
from casting_imbalance.features import FeatureBank
from casting_imbalance.models.classical import (FeatureTransformer,
                                                RandomForestModel, SVMClassifier)
from casting_imbalance.preprocessing import VALID_MODES, ImagePreprocessor


def tune_svm(bank: FeatureBank, runner: ExperimentRunner) -> dict:
    """Grid search over C and gamma using validation balanced accuracy."""
    tr, va = runner.train_df, runner.val_df
    transformer = FeatureTransformer().fit(bank.get(tr["path"]))
    X_tr = transformer.transform(bank.get(tr["path"]))
    X_va = transformer.transform(bank.get(va["path"]))
    rows = []
    for C, gamma in product(CFG.svm_c_grid, CFG.svm_gamma_grid):
        model = SVMClassifier(C=C, gamma=gamma).fit(X_tr, tr["label"].to_numpy())
        m = Evaluator.compute(runner.y_val, model.predict_proba(X_va))
        rows.append({"C": C, "gamma": str(gamma), "val_balanced_accuracy": m["balanced_accuracy"],
                     "val_roc_auc": m["roc_auc"], "val_f1_defective": m["f1_defective"]})
    table = pd.DataFrame(rows).sort_values(["val_balanced_accuracy", "val_roc_auc"],
                                           ascending=False)
    table.to_csv(CFG.output_dir / "tables" / "svm_tuning.csv", index=False)
    best = table.iloc[0]
    gamma = best["gamma"] if best["gamma"] == "scale" else float(best["gamma"])
    params = {"C": float(best["C"]), "gamma": gamma,
              "pca_components": transformer.n_components}
    print("SVM tuning (top 5):\n", table.head(), "\nselected:", params)
    return params


def main() -> None:
    CFG.ensure_dirs()
    split_df = DataSplitter().load_or_create(CastingDataset().index)
    runner = ExperimentRunner(split_df)
    paths = split_df["path"].tolist()

    print("Extracting features for every preprocessing variant ...")
    banks = {mode: FeatureBank(paths, ImagePreprocessor(CFG.classical_size, mode))
             for mode in VALID_MODES}
    print("Feature dimensions:", banks["gaussian"].dimensions())
    with open(CFG.output_dir / "tables" / "feature_dimensions.json", "w") as f:
        json.dump(banks["gaussian"].dimensions(), f)

    svm_params = tune_svm(banks["gaussian"], runner)
    with open(CFG.output_dir / "tables" / "svm_params.json", "w") as f:
        json.dump(svm_params, f)
    svm = lambda seed: SVMClassifier(C=svm_params["C"], gamma=svm_params["gamma"], seed=seed)
    rf = lambda seed: RandomForestModel(seed=seed)
    default_bank = banks["gaussian"]

    # ---- main experiment --------------------------------------------------
    for factory in (svm, rf):
        runner.run("main", factory, default_bank.get, CFG.classical_methods, use_pca=True)

    # ---- ablation A: image-processing variant ----------------------------
    for mode in VALID_MODES:
        runner.run(f"pre_{mode}", svm, banks[mode].get, ("none", "class_weight"),
                   ratios=("natural", 10), use_pca=True, verbose=False)

    # ---- ablation B: feature group ----------------------------------------
    for groups in (("hog",), ("lbp",), ("glcm",), ("hog", "lbp"), ("hog", "lbp", "glcm")):
        getter = lambda p, g=groups: default_bank.get(p, g)
        runner.run("feat_" + "+".join(groups), svm, getter, ("none", "class_weight"),
                   ratios=("natural", 10), use_pca=True, verbose=False)

    results = runner.results()
    results.to_csv(CFG.output_dir / "tables" / "results_classical.csv", index=False)
    with open(CFG.output_dir / "probs_classical.pkl", "wb") as f:
        pickle.dump({"probs": runner.probabilities, "y_test": runner.y_test,
                     "test_paths": runner.test_df["path"].tolist()}, f)
    print("Saved", len(results), "classical runs.")


if __name__ == "__main__":
    main()
