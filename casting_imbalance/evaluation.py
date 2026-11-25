"""
evaluation.py
-------------
Performance metrics that remain informative under class imbalance.

"Positive" always means Defective (label 1). Accuracy is reported for
completeness but is NOT used to choose models, because a classifier that
labels every casting "OK" can still look accurate when defects are rare.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


class Evaluator:
    """Computes a complete metric set from true labels and scores."""

    @staticmethod
    def compute(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        y_true = np.asarray(y_true).astype(int)
        y_pred = (np.asarray(prob) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall_defective": sensitivity,            # sensitivity / TPR
            "recall_ok": specificity,                   # specificity / TNR
            "precision_defective": precision_score(y_true, y_pred, zero_division=0),
            "f1_defective": f1_score(y_true, y_pred, zero_division=0),
            "g_mean": float(np.sqrt(sensitivity * specificity)),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, prob),
            "pr_auc": average_precision_score(y_true, prob),
            "false_negative_rate": fn / max(fn + tp, 1),  # defects accepted
            "false_positive_rate": fp / max(fp + tn, 1),  # good parts rejected
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "threshold": threshold,
        }
