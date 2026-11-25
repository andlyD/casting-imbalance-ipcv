"""
run_mobilenet.py
----------------
Step 3 of the pipeline: transfer-learning experiments with MobileNetV2.

    1. Pass every image through the frozen ImageNet MobileNetV2 once per
       preprocessing variant and cache the 1,280-D embeddings.
    2. Pre-compute embeddings of 10 randomly augmented copies of every
       TRAINING image (used only by augmentation-based oversampling).
    3. Main experiment: 4 ratios x 5 repetitions x 7 handling methods.
    4. Ablation: image-processing variant (natural and 10:1, none / class weight).

Run from the project root:  python scripts/run_mobilenet.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from casting_imbalance.config import CFG
from casting_imbalance.data import CastingDataset, DataSplitter
from casting_imbalance.experiment import ExperimentRunner
from casting_imbalance.models.neural import ImageAugmenter, MobileNetV2Embedder, TransferHead
from casting_imbalance.preprocessing import VALID_MODES, ImagePreprocessor, load_grayscale


class EmbeddingStore:
    """Path -> embedding lookup with an on-disk cache."""

    def __init__(self, paths, embedder, preprocessor, cache_file: Path):
        if cache_file.exists():
            data = np.load(cache_file, allow_pickle=True)
            self.paths, self.emb = list(data["paths"]), data["emb"]
        else:
            images = preprocessor.process_many(paths)
            self.paths, self.emb = list(paths), embedder.embed(images)
            np.savez_compressed(cache_file, paths=np.array(self.paths), emb=self.emb)
        self.row = {p: i for i, p in enumerate(self.paths)}

    def get(self, paths):
        return self.emb[[self.row[p] for p in paths]]


class AugmentedEmbeddingStore:
    """Embeddings of K augmented copies of each training image."""

    def __init__(self, paths, embedder, preprocessor, k: int, cache_file: Path):
        if cache_file.exists():
            data = np.load(cache_file, allow_pickle=True)
            paths, emb = list(data["paths"]), data["emb"]
        else:
            rng = np.random.RandomState(CFG.global_seed)
            augmenter = ImageAugmenter()
            emb_list = []
            for p in paths:   # augment the full-resolution image, then preprocess
                raw = load_grayscale(p)
                copies = np.stack([preprocessor.process(augmenter(raw, rng)) for _ in range(k)])
                emb_list.append(embedder.embed(copies))
            emb = np.stack(emb_list)                       # (N, K, 1280)
            np.savez_compressed(cache_file, paths=np.array(paths), emb=emb)
        self.row = {p: i for i, p in enumerate(paths)}
        self.emb = emb

    def lookup(self, path, rng):
        copies = self.emb[self.row[path]]
        return copies[rng.randint(len(copies))]


def main() -> None:
    CFG.ensure_dirs()
    split_df = DataSplitter().load_or_create(CastingDataset().index)
    runner = ExperimentRunner(split_df)
    paths = split_df["path"].tolist()
    embedder = MobileNetV2Embedder()
    cache = CFG.output_dir / "cache"
    cache.mkdir(exist_ok=True)

    stores = {}
    for mode in VALID_MODES:
        print("Embedding images, preprocessing =", mode, flush=True)
        stores[mode] = EmbeddingStore(paths, embedder,
                                      ImagePreprocessor(CFG.mobilenet_size, mode),
                                      cache / f"mobilenet_{mode}.npz")
    print("Embedding augmented training copies ...", flush=True)
    aug_store = AugmentedEmbeddingStore(
        runner.train_df["path"].tolist(), embedder,
        ImagePreprocessor(CFG.mobilenet_size, "gaussian"),
        CFG.n_augmented_copies, cache / "mobilenet_aug_gaussian.npz")

    head = lambda seed: TransferHead(seed=seed)
    runner.run("main", head, stores["gaussian"].get, CFG.mobilenet_methods,
               augmented_lookup=aug_store.lookup)
    for mode in VALID_MODES:
        runner.run(f"pre_{mode}", head, stores[mode].get, ("none", "class_weight"),
                   ratios=("natural", 10), verbose=False)

    results = runner.results()
    results.to_csv(CFG.output_dir / "tables" / "results_mobilenet.csv", index=False)
    with open(CFG.output_dir / "probs_mobilenet.pkl", "wb") as f:
        pickle.dump({"probs": runner.probabilities, "histories": runner.histories,
                     "y_test": runner.y_test,
                     "test_paths": runner.test_df["path"].tolist()}, f)
    print("Saved", len(results), "MobileNetV2 runs.")


if __name__ == "__main__":
    main()
