"""
run_cnn.py
----------
Step 4 of the pipeline: end-to-end CNN trained from scratch.

The CNN is the most expensive model on a CPU, so it is run with three
repetitions per imbalanced ratio (two for the natural distribution) and the four methods most relevant to
end-to-end training: none, class weighting, augmentation-based oversampling
and threshold moving.

Run from the project root:  python scripts/run_cnn.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from casting_imbalance.config import CFG
from casting_imbalance.data import CastingDataset, DataSplitter
from casting_imbalance.experiment import ExperimentRunner
from casting_imbalance.models.neural import SimpleCNN
from casting_imbalance.preprocessing import ImagePreprocessor


def main() -> None:
    CFG.ensure_dirs()
    split_df = DataSplitter().load_or_create(CastingDataset().index)
    runner = ExperimentRunner(split_df)
    pre = ImagePreprocessor(CFG.cnn_size, "gaussian")
    images = {p: pre.process_path(p) for p in split_df["path"]}
    get_images = lambda paths: np.stack([images[p] for p in paths])

    factory = lambda seed: SimpleCNN(seed=seed)
    # Under the natural distribution the training set is identical in every
    # repetition (only the weight initialisation changes), so two repetitions
    # suffice there; the imbalanced ratios use three different sub-samples.
    runner.run("main", factory, get_images, CFG.cnn_methods,
               ratios=("natural",), repetitions=2)
    runner.run("main", factory, get_images, CFG.cnn_methods,
               ratios=(3, 10, 20), repetitions=CFG.cnn_repetitions)

    results = runner.results()
    results.to_csv(CFG.output_dir / "tables" / "results_cnn.csv", index=False)
    with open(CFG.output_dir / "probs_cnn.pkl", "wb") as f:
        pickle.dump({"probs": runner.probabilities, "histories": runner.histories,
                     "y_test": runner.y_test,
                     "test_paths": runner.test_df["path"].tolist()}, f)
    print("Saved", len(results), "CNN runs.")


if __name__ == "__main__":
    main()
