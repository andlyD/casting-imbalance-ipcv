"""
run_more_classical.py
---------------------
Step 2b: two extra classical baselines on the same HOG + LBP + GLCM features.

    * Logistic Regression - tune C on validation (natural distribution).
    * k-Nearest Neighbours - tune k on validation (natural distribution).

Both are then run through exactly the same protocol as the SVM and Random
Forest (4 ratios x 5 repetitions x 6 handling methods), so the four classical
models can be compared on equal terms.

Run from the project root after run_classical.py:
    python scripts/run_more_classical.py
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from casting_imbalance.config import CFG
from casting_imbalance.data import CastingDataset, DataSplitter
from casting_imbalance.evaluation import Evaluator
from casting_imbalance.experiment import ExperimentRunner
from casting_imbalance.features import FeatureBank
from casting_imbalance.models.classical import (FeatureTransformer, KNNModel,
                                                LogisticRegressionModel)
from casting_imbalance.preprocessing import ImagePreprocessor

LR_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
KNN_K_GRID = (1, 3, 5, 7, 9, 15)


def tune(bank, runner, name, factory, grid):
    tr, va = runner.train_df, runner.val_df
    tf = FeatureTransformer().fit(bank.get(tr["path"]))
    X_tr, X_va = tf.transform(bank.get(tr["path"])), tf.transform(bank.get(va["path"]))
    rows = []
    for value in grid:
        model = factory(value).fit(X_tr, tr["label"].to_numpy())
        m = Evaluator.compute(runner.y_val, model.predict_proba(X_va))
        rows.append({"param": value, "val_balanced_accuracy": m["balanced_accuracy"],
                     "val_roc_auc": m["roc_auc"]})
    table = pd.DataFrame(rows).sort_values(["val_balanced_accuracy", "val_roc_auc"],
                                           ascending=False)
    table.to_csv(CFG.output_dir / "tables" / f"{name}_tuning.csv", index=False)
    print(name, "tuning:\n", table, "\n")
    return table.iloc[0]["param"]


def main() -> None:
    CFG.ensure_dirs()
    split_df = DataSplitter().load_or_create(CastingDataset().index)
    runner = ExperimentRunner(split_df)
    print("Extracting features ...")
    bank = FeatureBank(split_df["path"].tolist(),
                       ImagePreprocessor(CFG.classical_size, "gaussian"))

    best_c = float(tune(bank, runner, "logreg", lambda c: LogisticRegressionModel(C=c), LR_C_GRID))
    best_k = int(tune(bank, runner, "knn", lambda k: KNNModel(k=int(k)), KNN_K_GRID))
    with open(CFG.output_dir / "tables" / "extra_params.json", "w") as f:
        json.dump({"logreg_C": best_c, "knn_k": best_k}, f)

    runner.run("main", lambda s: LogisticRegressionModel(C=best_c, seed=s), bank.get,
               CFG.classical_methods, use_pca=True)
    runner.run("main", lambda s: KNNModel(k=best_k, seed=s), bank.get,
               CFG.classical_methods, use_pca=True)

    # ---- ablation: neighbourhood size for k-NN --------------------------
    # With k = 1 the prediction is a hard 0/1 vote, so class weights, duplicated
    # samples and threshold moving cannot change it. Larger k gives graded
    # scores; this ablation shows how much that matters under imbalance.
    for k in (1, 5, 9, 15):
        runner.run(f"knnk_{k}", lambda s, k=k: KNNModel(k=k, seed=s), bank.get,
                   ("none", "class_weight", "smote", "threshold_moving"),
                   ratios=("natural", 10, 20), use_pca=True, verbose=False)

    results = runner.results()
    results.to_csv(CFG.output_dir / "tables" / "results_classical_extra.csv", index=False)
    with open(CFG.output_dir / "probs_classical_extra.pkl", "wb") as f:
        pickle.dump({"probs": runner.probabilities, "y_test": runner.y_test,
                     "test_paths": runner.test_df["path"].tolist()}, f)
    print("Saved", len(results), "extra classical runs.")


if __name__ == "__main__":
    main()
