"""
run_svm_c_ablation.py
---------------------
Ablation C: why does class weighting not change the tuned SVM?

Cost-sensitive SVMs re-weight the penalty C for margin violations of each
class (C_minority > C_majority). If the training data are separated with
almost no margin violations (large C), the weights have nothing to act on.
This script repeats the 10:1 and 20:1 scenarios for several C values with
and without class weighting.

Run from the project root:  python scripts/run_svm_c_ablation.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casting_imbalance.config import CFG
from casting_imbalance.data import CastingDataset, DataSplitter
from casting_imbalance.experiment import ExperimentRunner
from casting_imbalance.features import FeatureBank
from casting_imbalance.models.classical import SVMClassifier
from casting_imbalance.preprocessing import ImagePreprocessor


def main() -> None:
    split_df = DataSplitter().load_or_create(CastingDataset().index)
    runner = ExperimentRunner(split_df)
    bank = FeatureBank(split_df["path"].tolist(),
                       ImagePreprocessor(CFG.classical_size, "gaussian"))
    with open(CFG.output_dir / "tables" / "svm_params.json") as f:
        gamma = json.load(f)["gamma"]
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        runner.run(f"svmC_{C}", lambda seed, C=C: SVMClassifier(C=C, gamma=gamma, seed=seed),
                   bank.get, ("none", "class_weight"), ratios=(10, 20),
                   use_pca=True, verbose=False)
    runner.results().to_csv(CFG.output_dir / "tables" / "results_svm_c.csv", index=False)
    print(runner.results().groupby(["tag", "ratio", "method"])[
        ["balanced_accuracy", "recall_defective", "recall_ok", "roc_auc"]].mean().round(3))


if __name__ == "__main__":
    main()
