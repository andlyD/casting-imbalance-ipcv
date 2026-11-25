"""
analyse_results.py
------------------
Step 5 of the pipeline: statistics, summary tables and result figures.

Reads every results_*.csv written by the experiment scripts and produces:
    tables/summary_main.csv        mean and SD of every metric
    tables/statistics.csv          Friedman and Wilcoxon-Holm tests
    tables/best_per_model.csv      best handling method per model and ratio
    figures/fig_*.png              all result figures used in the report

Run from the project root:  python scripts/analyse_results.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from casting_imbalance.config import CFG
from casting_imbalance.experiment import summarise
from casting_imbalance.plotting import (METHOD_COLORS, METHOD_LABELS, MODEL_COLORS,
                                        MODEL_LABELS, MODEL_SHORT, save_figure, style_axes)

TAB, FIG = CFG.output_dir / "tables", CFG.output_dir / "figures"
RATIO_ORDER = ["natural", "3", "10", "20"]
RATIO_LABEL = {"natural": "Natural\n(0.66:1)", "3": "3:1", "10": "10:1", "20": "20:1"}
MODEL_ORDER = ["SVM", "LogReg", "KNN", "RandomForest", "MobileNetV2", "CNN"]


def load_results() -> pd.DataFrame:
    frames = [pd.read_csv(TAB / f"results_{n}.csv")
              for n in ("classical", "classical_extra", "mobilenet", "cnn") if (TAB / f"results_{n}.csv").exists()]
    df = pd.concat(frames, ignore_index=True)
    df["ratio"] = df["ratio"].astype(str)
    return df


def load_probs() -> dict:
    probs, y_test = {}, None
    for n in ("classical", "classical_extra", "mobilenet", "cnn"):
        f = CFG.output_dir / f"probs_{n}.pkl"
        if f.exists():
            d = pickle.load(open(f, "rb"))
            probs.update(d["probs"])
            y_test = d["y_test"]
            if "histories" in d:
                probs.setdefault("_histories", {}).update(d["histories"])
    return probs, y_test


# ------------------------------------------------------------------ figures
def fig_baseline_degradation(summ: pd.DataFrame) -> None:
    base = summ[(summ.tag == "main") & (summ.method == "none")]
    metrics = [("balanced_accuracy", "(a) Balanced accuracy"),
               ("recall_defective", "(b) Defective recall (sensitivity)"),
               ("roc_auc", "(c) ROC-AUC (threshold-free)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharex=True)
    x = np.arange(len(RATIO_ORDER))
    for ax, (m, title) in zip(axes, metrics):
        for model in MODEL_ORDER:
            d = base[base.model == model].set_index("ratio").reindex(RATIO_ORDER)
            if d[m].isna().all():
                continue
            ax.errorbar(x, d[m], yerr=d[f"{m}_sd"], marker="o", ms=4, capsize=3,
                        color=MODEL_COLORS[model], label=MODEL_LABELS[model], lw=1.6)
        ax.set_xticks(x, [RATIO_LABEL[r] for r in RATIO_ORDER])
        ax.set_title(title)
        ax.set_ylim(-0.03 if m == "recall_defective" else 0.3, 1.03)
        ax.set_xlabel("Training ratio OK : Defective")
        style_axes(ax)
    axes[0].set_ylabel("Score (mean ± SD)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.13))
    save_figure(fig, FIG / "fig_baseline_degradation.png")


def fig_heatmaps(summ: pd.DataFrame, metric: str, fname: str, title: str) -> None:
    main = summ[summ.tag == "main"]
    models = [m for m in MODEL_ORDER if m in main.model.unique()]
    ncol = 3
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 3.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(models):]:
        ax.axis("off")
    for idx, (ax, model) in enumerate(zip(axes, models)):
        d = main[main.model == model]
        methods = list(METHOD_LABELS)          # same rows in every panel
        grid = d.pivot(index="method", columns="ratio", values=metric).reindex(
            index=methods, columns=RATIO_ORDER)
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#E6E6E6")
        im = ax.imshow(np.ma.masked_invalid(grid.to_numpy(dtype=float)), cmap=cmap,
                       vmin=0.5, vmax=1.0, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid.iat[i, j]
                ax.text(j, i, "n/a" if pd.isna(v) else f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="#7F7F7F" if pd.isna(v) else "black")
        ax.set_xticks(range(len(RATIO_ORDER)), ["Nat.", "3:1", "10:1", "20:1"])
        ax.set_yticks(range(len(methods)),
                      [METHOD_LABELS[m] for m in methods] if idx % ncol == 0 else [""] * len(methods))
        ax.set_title(MODEL_LABELS[model], fontsize=9)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label=title)
    fig.savefig(FIG / fname, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_severe_methods(summ: pd.DataFrame, ratio: str = "20") -> None:
    main = summ[(summ.tag == "main") & (summ.ratio == ratio)]
    models = [m for m in MODEL_ORDER if m in main.model.unique()]
    nrow = int(np.ceil(len(models) / 2))
    fig, axes = plt.subplots(nrow, 2, figsize=(11, 2.3 * nrow), sharex=False)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(models):]:
        ax.axis("off")
    for ax, model in zip(axes, models):
        d = main[main.model == model].set_index("method")
        methods = [m for m in METHOD_LABELS if m in d.index]
        x = np.arange(len(methods))
        ax.bar(x - 0.2, d.loc[methods, "recall_defective"], 0.38,
               yerr=d.loc[methods, "recall_defective_sd"], capsize=2,
               color="#D55E00", label="Defective recall")
        ax.bar(x + 0.2, d.loc[methods, "recall_ok"], 0.38,
               yerr=d.loc[methods, "recall_ok_sd"], capsize=2,
               color="#56B4E9", label="OK recall")
        ax.plot(x, d.loc[methods, "balanced_accuracy"], "kD", ms=5, label="Balanced accuracy")
        short = {"none": "None", "class_weight": "Class\nweight", "random_oversampling": "ROS",
                 "smote": "SMOTE", "random_undersampling": "RUS",
                 "augmentation_oversampling": "Aug.\nover", "threshold_moving": "Thresh.\nmoving"}
        ax.set_xticks(x, [short[m] for m in methods], fontsize=7.5)
        ax.set_ylim(0, 1.08)
        ax.set_title(MODEL_LABELS[model], fontsize=9, loc="left")
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, 1.03))
    save_figure(fig, FIG / f"fig_methods_ratio{ratio}.png")


def fig_roc_pr(probs: dict, y: np.ndarray) -> None:
    """(a) ROC on natural data, (b) ROC at 20:1, (c) precision-recall at 20:1.
    Repetition 0 with no handling, so the curves show the raw ranking ability."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    panels = ((axes[0], "natural", "roc"), (axes[1], "20", "roc"), (axes[2], "20", "pr"))
    for ax, ratio, kind in panels:
        for model in MODEL_ORDER:
            key = ("main", model, ratio, 0, "none")
            if key not in probs:
                continue
            p = probs[key]
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, color=MODEL_COLORS[model], lw=1.4, label=MODEL_SHORT[model])
            else:
                prec, rec, _ = precision_recall_curve(y, p)
                ax.plot(rec, prec, color=MODEL_COLORS[model], lw=1.4, label=MODEL_SHORT[model])
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
            ax.set(xlabel="False positive rate (good parts rejected)",
                   ylabel="True positive rate (defects caught)")
        else:
            ax.axhline(y.mean(), ls=":", color="grey", lw=0.8)
            ax.set(xlabel="Defective recall", ylabel="Defective precision")
        style_axes(ax)
    axes[0].set_title("(a) ROC, natural training data")
    axes[1].set_title("(b) ROC, 20:1 training data")
    axes[2].set_title("(c) Precision-recall, 20:1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.06))
    save_figure(fig, FIG / "fig_roc_pr.png")


def fig_model_comparison(summ: pd.DataFrame, best: pd.DataFrame) -> None:
    """The headline comparison of the six models.

    (a) Balanced accuracy on natural training data (no handling).
    (b) Dumbbell at 20:1: baseline (open circle) to best handling method
        (filled circle), so the gain each model gets from handling is visible.
    """
    main = summ[(summ.tag == "main")]
    models = [m for m in MODEL_ORDER if m in main.model.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    ypos = np.arange(len(models))[::-1]
    nat = main[(main.ratio == "natural") & (main.method == "none")].set_index("model")
    ax = axes[0]
    vals = [nat.loc[m, "balanced_accuracy"] for m in models]
    sds = [nat.loc[m, "balanced_accuracy_sd"] for m in models]
    ax.barh(ypos, vals, xerr=sds, capsize=2, color=[MODEL_COLORS[m] for m in models], height=0.6)
    for y_, v, sd in zip(ypos, vals, sds):
        ax.text(v + (sd if sd == sd else 0) + 0.004, y_, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_xlim(0.8, 1.02)
    ax.set_yticks(ypos, [MODEL_SHORT[m] for m in models])
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("(a) Natural training data, no handling", loc="left")
    ax = axes[1]
    sev = main[main.ratio == "20"].set_index(["model", "method"])
    for y_, m in zip(ypos, models):
        b = sev.loc[(m, "none"), "balanced_accuracy"]
        bm = best[(best.model == m) & (best.ratio == "20")].iloc[0]
        ax.plot([b, bm.balanced_accuracy], [y_, y_], color="#BBBBBB", lw=2.2, zorder=1)
        ax.scatter([b], [y_], s=42, facecolor="white", edgecolor=MODEL_COLORS[m], lw=1.6, zorder=2)
        ax.scatter([bm.balanced_accuracy], [y_], s=46, color=MODEL_COLORS[m], zorder=3)
        ax.text(max(b, bm.balanced_accuracy) + 0.012, y_,
                f"{bm.balanced_accuracy:.3f}  ({METHOD_LABELS[bm.method]})", va="center",
                fontsize=7.2)
    ax.set_xlim(0.45, 1.25)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("(b) 20:1 training data: baseline (o) to best method (●)", loc="left")
    for a in axes:
        a.grid(axis="x", color="#E5E5E5", lw=0.6)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.set_axisbelow(True)
    save_figure(fig, FIG / "fig_model_comparison.png")


def fig_knn_k(raw: pd.DataFrame) -> None:
    d = raw[raw.tag.str.startswith("knnk_")]
    if d.empty:
        return
    d = d.assign(k=d.tag.str.replace("knnk_", "").astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    for ax, ratio in zip(axes, ("10", "20")):
        g = d[d.ratio == ratio].groupby(["k", "method"])["balanced_accuracy"].mean().unstack()
        for method in ("none", "class_weight", "smote", "threshold_moving"):
            ax.plot(g.index, g[method], marker="o", ms=4, color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method])
        ax.set_xticks(g.index)
        ax.set(xlabel="Number of neighbours k", title=f"Training ratio {ratio}:1")
        style_axes(ax)
    axes[0].set_ylabel("Balanced accuracy")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    save_figure(fig, FIG / "fig_knn_k.png")


def fig_confusions(raw: pd.DataFrame, probs: dict, y: np.ndarray, best: pd.DataFrame,
                   ratio: str = "20") -> None:
    models = [m for m in MODEL_ORDER if ("main", m, ratio, 0, "none") in probs]
    fig, axes = plt.subplots(2, len(models), figsize=(2.15 * len(models), 5.0))
    for j, model in enumerate(models):
        best_m = best[(best.model == model) & (best.ratio == ratio)]["method"].iloc[0]
        for i, method in enumerate(("none", best_m)):
            row = raw[(raw.tag == "main") & (raw.model == model) & (raw.ratio == ratio)
                      & (raw.rep == 0) & (raw.method == method)].iloc[0]
            cm = confusion_matrix(y, (probs[("main", model, ratio, 0, method)]
                                      >= row["threshold"]).astype(int), labels=[0, 1])
            ax = axes[i, j]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=len(y))
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, cm[a, b], ha="center", va="center", fontsize=10,
                            color="white" if cm[a, b] > len(y) / 2 else "black")
            ax.set_xticks([0, 1], ["OK", "Def"])
            ax.set_yticks([0, 1], ["OK", "Def"])
            if j == 0:
                ax.set_ylabel("True class")
            ax.set_xlabel("Predicted class")
            ax.set_title(f"{MODEL_SHORT[model]}\n{METHOD_LABELS[method]}", fontsize=7.5)
    save_figure(fig, FIG / f"fig_confusion_ratio{ratio}.png")


def fig_ablation_preprocessing(raw: pd.DataFrame) -> None:
    pre = raw[raw.tag.str.startswith("pre_") & (raw.method == "none")]
    if pre.empty:
        return
    modes = ["resize", "gaussian", "clahe", "gaussian_clahe"]
    names = ["Resize", "Gaussian\n(default)", "CLAHE", "Gaussian\n+ CLAHE"]
    models = [m for m in ("SVM", "MobileNetV2") if m in pre.model.unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        for k, (ratio, col) in enumerate((("natural", "#0072B2"), ("10", "#E69F00"))):
            vals, sds = [], []
            for mode in modes:
                d = pre[(pre.model == model) & (pre.tag == f"pre_{mode}") & (pre.ratio == ratio)]
                vals.append(d.balanced_accuracy.mean()); sds.append(d.balanced_accuracy.std())
            x = np.arange(len(modes)) + (k - 0.5) * 0.38
            bars = ax.bar(x, vals, 0.36, yerr=sds, capsize=2, color=col,
                          label="Natural" if ratio == "natural" else "10:1")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center",
                        fontsize=6.8)
        ax.set_xticks(range(len(modes)), names)
        ax.set_ylim(0.5, 1.05)
        ax.set_title(MODEL_LABELS[model], fontsize=9)
        style_axes(ax)
    axes[0].set_ylabel("Balanced accuracy (no handling)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, title="Training ratio", fontsize=7.5,
               ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    save_figure(fig, FIG / "fig_ablation_preprocessing.png")


def fig_ablation_features(raw: pd.DataFrame) -> None:
    feat = raw[raw.tag.str.startswith("feat_")]
    if feat.empty:
        return
    groups = ["hog", "lbp", "glcm", "hog+lbp", "hog+lbp+glcm"]
    labels = ["HOG", "LBP", "GLCM", "HOG+LBP", "HOG+LBP\n+GLCM"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    combos = [("natural", "none", "#0072B2", "Natural, none"),
              ("10", "none", "#E69F00", "10:1, none"),
              ("10", "class_weight", "#009E73", "10:1, class weighting")]
    for k, (ratio, method, col, lab) in enumerate(combos):
        vals = [feat[(feat.tag == f"feat_{g}") & (feat.ratio == ratio)
                     & (feat.method == method)].balanced_accuracy.mean() for g in groups]
        x = np.arange(len(groups)) + (k - 1) * 0.27
        bars = ax.bar(x, vals, 0.25, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(groups)), labels)
    ax.set_ylim(0.4, 1.06)
    ax.set_ylabel("Balanced accuracy")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    style_axes(ax)
    save_figure(fig, FIG / "fig_ablation_features.png")


def fig_svm_c() -> None:
    f = TAB / "results_svm_c.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d["C"] = d.tag.str.replace("svmC_", "").astype(float)
    d["ratio"] = d["ratio"].astype(str)
    g = d.groupby(["C", "ratio", "method"])["balanced_accuracy"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for ratio, ls in (("10", "-"), ("20", "--")):
        for method in ("none", "class_weight"):
            s = g[(g.ratio == ratio) & (g.method == method)].sort_values("C")
            ax.errorbar(s.C, s["mean"], yerr=s["std"], ls=ls, marker="o", ms=4, capsize=2,
                        color=METHOD_COLORS[method],
                        label=f"{METHOD_LABELS[method]}, {ratio}:1")
    ax.set_xscale("log")
    ax.axvline(10, color="grey", lw=0.8, ls=":")
    ax.text(10.5, 0.52, "tuned C", fontsize=7, color="grey")
    ax.set(xlabel="SVM penalty C (log scale)", ylabel="Balanced accuracy", ylim=(0.45, 0.85))
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    style_axes(ax)
    save_figure(fig, FIG / "fig_svm_c_ablation.png")


def fig_thresholds(raw: pd.DataFrame) -> None:
    t = raw[(raw.tag == "main") & (raw.method == "threshold_moving")]
    models = [m for m in MODEL_ORDER if m in t.model.unique()]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    width = 0.84 / len(models)
    for k, model in enumerate(models):
        g = t[t.model == model].groupby("ratio")["threshold"].agg(["mean", "std"]).reindex(RATIO_ORDER)
        x = np.arange(len(RATIO_ORDER)) + (k - (len(models) - 1) / 2) * width
        ax.bar(x, g["mean"], width * 0.9, yerr=g["std"], capsize=2,
               color=MODEL_COLORS[model], label=MODEL_SHORT[model])
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.text(3.45, 0.52, "default 0.5", fontsize=7, ha="right")
    ax.set_xticks(range(len(RATIO_ORDER)), [RATIO_LABEL[r] for r in RATIO_ORDER])
    ax.set_ylabel("Tuned decision threshold")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    style_axes(ax)
    save_figure(fig, FIG / "fig_thresholds.png")


def fig_learning_curves(probs: dict) -> None:
    hist = probs.get("_histories", {})
    keys = [k for k in (("main", "CNN", "natural", 0, "none"), ("main", "CNN", "20", 0, "none"),
                        ("main", "MobileNetV2", "natural", 0, "none"),
                        ("main", "MobileNetV2", "20", 0, "none")) if k in hist]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        h = hist[k]
        ax.plot(h["auc"], color="#0072B2", label="Training AUC")
        ax.plot(h["val_auc"], color="#D55E00", label="Validation AUC")
        # Epoch whose weights were restored: lowest validation loss for the
        # CNN, highest validation AUC for the MobileNetV2 head.
        best = int(np.argmin(h["val_loss"])) if k[1] == "CNN" else int(np.argmax(h["val_auc"]))
        ax.axvline(best, color="grey", ls=":", lw=0.8)
        ax.set_title(f"{k[1]}, {'natural' if k[2] == 'natural' else k[2] + ':1'}", fontsize=8.5)
        ax.set_xlabel("Epoch")
        style_axes(ax)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(frameon=False, fontsize=7)
    save_figure(fig, FIG / "fig_learning_curves.png")


# --------------------------------------------------------------- statistics
def statistics(raw: pd.DataFrame) -> pd.DataFrame:
    """Friedman test across methods (blocks = imbalanced ratio x repetition)
    and Wilcoxon signed-rank tests of each method against the baseline with
    Holm correction, on balanced accuracy."""
    rows = []
    imb = raw[(raw.tag == "main") & (raw.ratio != "natural")]
    for model in MODEL_ORDER:
        d = imb[imb.model == model]
        if d.empty:
            continue
        table = d.pivot_table(index=["ratio", "rep"], columns="method",
                              values="balanced_accuracy")
        fr = stats.friedmanchisquare(*[table[c] for c in table.columns])
        ranks = table.rank(axis=1, ascending=False).mean()
        tests = []
        for m in table.columns:
            if m == "none":
                continue
            diff = table[m] - table["none"]
            p = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(table[m], table["none"]).pvalue
            tests.append((m, diff.mean(), p))
        tests.sort(key=lambda t: t[2])
        k = len(tests)
        running = 0.0
        for i, (m, dmean, p) in enumerate(tests):          # Holm step-down
            running = max(running, min(1.0, p * (k - i)))
            rows.append({"model": model, "method": m, "n_blocks": len(table),
                         "friedman_chi2": fr.statistic, "friedman_p": fr.pvalue,
                         "mean_rank": ranks[m], "rank_none": ranks["none"],
                         "mean_diff_vs_none": dmean, "wilcoxon_p": p, "holm_p": running})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "statistics.csv", index=False)
    return out


def main() -> None:
    raw = load_results()
    probs, y = load_probs()
    summ = summarise(raw)
    summ.to_csv(TAB / "summary_main.csv", index=False)

    # Best method per model and ratio by mean test balanced accuracy.
    main_s = summ[summ.tag == "main"]
    best = (main_s.sort_values("balanced_accuracy", ascending=False)
            .groupby(["model", "ratio"]).head(1)
            [["model", "ratio", "method", "balanced_accuracy", "balanced_accuracy_sd",
              "recall_defective", "recall_ok", "f1_defective", "mcc", "roc_auc"]])
    best.to_csv(TAB / "best_per_model.csv", index=False)

    # Overall ranking of model + method across the three imbalanced ratios.
    overall = (main_s[main_s.ratio != "natural"].groupby(["model", "method"])
               [["balanced_accuracy", "recall_defective", "recall_ok", "mcc", "f1_defective"]]
               .mean().sort_values("balanced_accuracy", ascending=False).reset_index())
    overall.to_csv(TAB / "overall_ranking_imbalanced.csv", index=False)

    fig_baseline_degradation(summ)
    fig_heatmaps(summ, "balanced_accuracy", "fig_heatmap_ba.png", "Balanced accuracy")
    fig_heatmaps(summ, "recall_defective", "fig_heatmap_recall.png", "Defective recall")
    for r in ("10", "20"):
        fig_severe_methods(summ, r)
    fig_roc_pr(probs, y)
    fig_model_comparison(summ, best)
    fig_knn_k(raw)
    fig_confusions(raw, probs, y, best, "20")
    fig_ablation_preprocessing(raw)
    fig_ablation_features(raw)
    fig_svm_c()
    fig_thresholds(raw)
    fig_learning_curves(probs)
    st = statistics(raw)
    pd.set_option("display.width", 220)
    print(best.round(3).to_string(index=False))
    print(overall.head(12).round(3).to_string(index=False))
    print(st.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
