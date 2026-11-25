"""
run_eda.py
----------
Step 1 of the pipeline: exploratory data analysis and data-integrity checks.

Produces the dataset figures used in the report and a JSON summary:
    * class distribution of the full dataset and of each split
    * sample images of both classes
    * the four image-processing variants
    * HOG / LBP visualisations of a defective casting
    * training-set composition of every imbalance scenario
    * augmentation examples
    * mean-intensity comparison between classes (shortcut check)
    * exact (SHA-256) and near-duplicate (RMS pixel difference) checks

Run from the project root:  python scripts/run_eda.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from skimage.feature import hog, local_binary_pattern

from casting_imbalance.config import CFG
from casting_imbalance.data import CastingDataset, DataSplitter, ImbalanceScenarioBuilder
from casting_imbalance.models.neural import ImageAugmenter
from casting_imbalance.preprocessing import VALID_MODES, ImagePreprocessor, load_grayscale
from casting_imbalance.plotting import COLORS, save_figure, style_axes

FIG = CFG.output_dir / "figures"


def main() -> None:
    CFG.ensure_dirs()
    dataset = CastingDataset()
    split_df = DataSplitter().load_or_create(dataset.index)
    summary = {}

    # ---------------------------------------------------- class distribution
    counts = split_df.groupby(["split", "class_name"]).size().unstack()[list(CFG.class_names)]
    counts = counts.loc[["train", "val", "test"]]
    summary["split_counts"] = counts.to_dict()
    total = dataset.class_counts()
    summary["total_counts"] = total.to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    bars = ax.bar(list(CFG.class_names), [total[c] for c in CFG.class_names],
                  color=[COLORS["ok"], COLORS["defective"]], width=0.55)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 10, f"{int(b.get_height())}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("(a) Full dataset (1,300 images)")
    ax.set_ylabel("Number of images")
    style_axes(ax)
    ax = axes[1]
    x = np.arange(3)
    for i, cls in enumerate(CFG.class_names):
        vals = counts[cls].to_numpy()
        b = ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, label=cls,
                   color=COLORS["ok"] if cls == "OK" else COLORS["defective"])
        for rect in b:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 6,
                    f"{int(rect.get_height())}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ["Train (70%)", "Validation (15%)", "Test (15%)"])
    ax.set_title("(b) Stratified split")
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, FIG / "fig_class_distribution.png")

    # ---------------------------------------------------------- sample grid
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.7))
    rng = np.random.RandomState(7)
    for row, label in enumerate([0, 1]):
        paths = split_df[split_df["label"] == label]["path"].to_numpy()
        for col, p in enumerate(rng.choice(paths, 4, replace=False)):
            axes[row, col].imshow(load_grayscale(p), cmap="gray")
            axes[row, col].set_title(f"{CFG.class_names[label]}: {Path(p).name}", fontsize=7.5)
            axes[row, col].axis("off")
    save_figure(fig, FIG / "fig_samples.png")

    # ------------------------------------------------- preprocessing variants
    example = split_df[split_df["filename"] == "cast_def_0_0.jpeg"]["path"].iloc[0]
    raw = load_grayscale(example)
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.6))
    axes[0].imshow(raw, cmap="gray"); axes[0].set_title("Original 512x512", fontsize=8.5)
    names = {"resize": "Resize 128", "gaussian": "Resize + Gaussian (default)",
             "clahe": "Resize + CLAHE", "gaussian_clahe": "Gaussian + CLAHE"}
    for ax, mode in zip(axes[1:], VALID_MODES):
        ax.imshow(ImagePreprocessor(CFG.classical_size, mode).process(raw), cmap="gray")
        ax.set_title(names[mode], fontsize=8.5)
    for ax in axes:
        ax.axis("off")
    save_figure(fig, FIG / "fig_preprocessing.png")

    # ----------------------------------------------------- HOG / LBP visual
    img = ImagePreprocessor(CFG.classical_size, "gaussian").process(raw)
    _, hog_img = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                     block_norm="L2-Hys", visualize=True)
    lbp = local_binary_pattern(img, 8, 1, "uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Preprocessed 128x128", fontsize=8.5)
    axes[1].imshow(hog_img ** 0.5, cmap="magma"); axes[1].set_title("HOG gradient cells", fontsize=8.5)
    axes[2].imshow(lbp, cmap="viridis"); axes[2].set_title("Uniform LBP codes", fontsize=8.5)
    for ax in axes[:3]:
        ax.axis("off")
    axes[3].bar(range(10), hist / hist.sum(), color=COLORS["accent"])
    axes[3].set_title("Global LBP histogram", fontsize=8.5)
    axes[3].set_xlabel("LBP code"); axes[3].set_ylabel("Frequency")
    style_axes(axes[3])
    save_figure(fig, FIG / "fig_features.png")

    # ------------------------------------------------- imbalance scenarios
    builder = ImbalanceScenarioBuilder(split_df[split_df["split"] == "train"])
    rows = []
    for ratio in CFG.imbalance_ratios:
        d = ImbalanceScenarioBuilder.describe(builder.build(ratio, 1000))
        rows.append({"ratio": "Natural" if ratio == "natural" else f"{ratio}:1", **d})
    scen = pd.DataFrame(rows)
    scen.to_csv(CFG.output_dir / "tables" / "scenarios.csv", index=False)
    summary["scenarios"] = rows
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(scen["ratio"], scen["n_ok"], color=COLORS["ok"], label="OK")
    ax.barh(scen["ratio"], scen["n_defective"], left=scen["n_ok"],
            color=COLORS["defective"], label="Defective")
    for i, r in scen.iterrows():
        ax.text(r["n_ok"] + r["n_defective"] + 8, i,
                f"{r['n_ok']} OK : {r['n_defective']} Def  (minority {r['minority_share']:.1%})",
                va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1250)
    ax.set_xlabel("Training images")
    ax.set_ylabel("OK : Defective ratio")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    save_figure(fig, FIG / "fig_scenarios.png")

    # --------------------------------------------------- augmentation demo
    aug = ImageAugmenter()
    rng = np.random.RandomState(3)
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.4))
    axes[0].imshow(raw, cmap="gray"); axes[0].set_title("Original", fontsize=8.5)
    for i, ax in enumerate(axes[1:]):
        ax.imshow(aug(raw, rng), cmap="gray"); ax.set_title(f"Augmented copy {i + 1}", fontsize=8.5)
    for ax in axes:
        ax.axis("off")
    save_figure(fig, FIG / "fig_augmentation.png")

    # --------------------------------------------- intensity shortcut check
    means = np.array([load_grayscale(p).mean() for p in split_df["path"]])
    split_df = split_df.assign(mean_intensity=means)
    ok_m = split_df[split_df.label == 0]["mean_intensity"]
    def_m = split_df[split_df.label == 1]["mean_intensity"]
    t = stats.mannwhitneyu(ok_m, def_m)
    summary["intensity"] = {"ok_mean": ok_m.mean(), "ok_sd": ok_m.std(),
                            "def_mean": def_m.mean(), "def_sd": def_m.std(),
                            "mannwhitney_p": float(t.pvalue)}

    # ------------------------------------------------- near-duplicate check
    # Every image is reduced to 64x64 and compared with every other image using
    # the root-mean-square (RMS) grey-level difference. Exact duplicates give 0;
    # re-saved or slightly shifted copies give a few grey levels.
    X = np.stack([cv2.resize(load_grayscale(p), (64, 64), interpolation=cv2.INTER_AREA)
                  .astype(np.float32).ravel() for p in split_df["path"]])
    sq = (X ** 2).sum(1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0) / X.shape[1])
    np.fill_diagonal(dist, np.inf)
    nearest, nearest_d = dist.argmin(1), dist.min(1)
    close = {tuple(sorted((i, int(nearest[i])))) for i in np.where(nearest_d < 5.0)[0]}
    dup_df = pd.DataFrame([(split_df["filename"][a], split_df["filename"][b],
                            split_df["split"][a], split_df["split"][b],
                            round(float(dist[a, b]), 2)) for a, b in close],
                          columns=["file_a", "file_b", "split_a", "split_b", "rms_difference"])
    dup_df.to_csv(CFG.output_dir / "tables" / "near_duplicates.csv", index=False)
    summary["exact_duplicates_sha256"] = 0 if len(set(
        __import__("hashlib").sha256(open(p, "rb").read()).hexdigest()
        for p in split_df["path"])) == len(split_df) else "check"
    summary["nearest_neighbour_rms"] = {"min": float(nearest_d.min()),
                                        "median": float(np.median(nearest_d))}
    summary["pairs_rms_below_5"] = len(dup_df)
    summary["cross_split_pairs_rms_below_5"] = int((dup_df["split_a"] != dup_df["split_b"]).sum())

    with open(CFG.output_dir / "tables" / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
