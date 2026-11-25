"""
plotting.py
-----------
Shared figure style so every chart in the report looks consistent.
"""

from pathlib import Path

import sys

import matplotlib
if "ipykernel" not in sys.modules:      # scripts and the dashboard: draw off-screen
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colour-blind-safe palette (Okabe-Ito based).
COLORS = {
    "ok": "#56B4E9",          # sky blue
    "defective": "#D55E00",   # vermilion
    "accent": "#0072B2",      # blue
    "grey": "#7F7F7F",
}
MODEL_COLORS = {"SVM": "#0072B2", "LogReg": "#E69F00", "KNN": "#8C564B",
                "RandomForest": "#009E73", "MobileNetV2": "#D55E00", "CNN": "#CC79A7"}
METHOD_COLORS = {
    "none": "#7F7F7F", "class_weight": "#0072B2", "random_oversampling": "#56B4E9",
    "smote": "#009E73", "random_undersampling": "#E69F00",
    "augmentation_oversampling": "#CC79A7", "threshold_moving": "#D55E00",
}
METHOD_LABELS = {
    "none": "None (baseline)", "class_weight": "Class weighting",
    "random_oversampling": "Random oversampling", "smote": "SMOTE",
    "random_undersampling": "Random undersampling",
    "augmentation_oversampling": "Augmentation oversampling",
    "threshold_moving": "Threshold moving",
}
MODEL_LABELS = {"SVM": "HOG+LBP+GLCM + SVM", "LogReg": "HOG+LBP+GLCM + LR",
                "KNN": "HOG+LBP+GLCM + k-NN", "RandomForest": "HOG+LBP+GLCM + RF",
                "MobileNetV2": "MobileNetV2 (transfer)", "CNN": "CNN (from scratch)"}
MODEL_SHORT = {"SVM": "SVM", "LogReg": "Logistic Reg.", "KNN": "k-NN",
               "RandomForest": "Random Forest", "MobileNetV2": "MobileNetV2", "CNN": "CNN"}
MODEL_FAMILY = {"SVM": "Classical", "LogReg": "Classical", "KNN": "Classical",
                "RandomForest": "Classical", "MobileNetV2": "Neural", "CNN": "Neural"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 110,
})


def style_axes(ax) -> None:
    """Remove top/right spines and add a light horizontal grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)


def save_figure(fig, path: Path, dpi: int = 200) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
