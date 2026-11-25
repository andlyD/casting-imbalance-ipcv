"""Export the numbers used in the report tables to outputs/report_data.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from casting_imbalance.config import CFG

TAB = CFG.output_dir / "tables"
summ = pd.read_csv(TAB / "summary_main.csv")
summ["ratio"] = summ["ratio"].astype(str)
main = summ[summ.tag == "main"]


def fmt(m, s=None, d=3):
    return f"{m:.{d}f}" if s is None else f"{m:.{d}f} ± {s:.{d}f}"


out = {}
# Full results table (Appendix A)
rows = []
order_m = ["SVM", "LogReg", "KNN", "RandomForest", "MobileNetV2", "CNN"]
order_r = ["natural", "3", "10", "20"]
order_h = ["none", "class_weight", "random_oversampling", "smote", "random_undersampling",
           "augmentation_oversampling", "threshold_moving"]
for mo in order_m:
    for ra in order_r:
        for me in order_h:
            d = main[(main.model == mo) & (main.ratio == ra) & (main.method == me)]
            if d.empty:
                continue
            d = d.iloc[0]
            rows.append([mo, ra, me, fmt(d.balanced_accuracy, d.balanced_accuracy_sd),
                         fmt(d.recall_defective), fmt(d.recall_ok), fmt(d.f1_defective),
                         fmt(d.mcc), fmt(d.roc_auc)])
out["full"] = rows

# Baseline (no handling) table
base = []
for mo in order_m:
    for ra in order_r:
        d = main[(main.model == mo) & (main.ratio == ra) & (main.method == "none")]
        if d.empty:
            continue
        d = d.iloc[0]
        base.append([mo, ra, fmt(d.accuracy), fmt(d.balanced_accuracy, d.balanced_accuracy_sd),
                     fmt(d.recall_defective), fmt(d.recall_ok), fmt(d.mcc), fmt(d.roc_auc)])
out["baseline"] = base

st = pd.read_csv(TAB / "statistics.csv")
out["stats"] = [[r.model, r.method, f"{r.mean_rank:.2f}", f"{r.mean_diff_vs_none:+.3f}",
                 f"{r.wilcoxon_p:.4f}", f"{r.holm_p:.4f}"] for r in st.itertuples()]
out["friedman"] = {m: [float(g.friedman_chi2.iloc[0]), float(g.friedman_p.iloc[0]),
                       float(g.rank_none.iloc[0])] for m, g in st.groupby("model")}

ov = pd.read_csv(TAB / "overall_ranking_imbalanced.csv")
out["overall"] = [[r.model, r.method, fmt(r.balanced_accuracy), fmt(r.recall_defective),
                   fmt(r.recall_ok), fmt(r.mcc)] for r in ov.head(10).itertuples()]
out["overall_all"] = [[r.model, r.method, fmt(r.balanced_accuracy)] for r in ov.itertuples()]

tun = pd.read_csv(TAB / "svm_tuning.csv")
out["svm_tuning"] = [[str(r.C), str(r.gamma), f"{r.val_balanced_accuracy:.4f}",
                      f"{r.val_roc_auc:.4f}", f"{r.val_f1_defective:.4f}"] for r in tun.itertuples()]
card = json.loads((CFG.model_dir / "model_card.json").read_text())
out["final"] = card
out["eda"] = json.loads((TAB / "eda_summary.json").read_text())
(CFG.output_dir / "report_data.json").write_text(json.dumps(out, indent=1))
print("rows:", len(rows), "baseline:", len(base), "stats:", len(out["stats"]))

# ---- scalar values quoted in the text ----------------------------------
def g(model, ratio, method, metric):
    d = main[(main.model == model) & (main.ratio == ratio) & (main.method == method)]
    return float(d[metric].iloc[0]) if len(d) else float("nan")

ovm = ov.set_index(["model", "method"])
C = {
    "svm_nat": f"{g('SVM','natural','none','balanced_accuracy'):.3f}",
    "rf_nat": f"{g('RandomForest','natural','none','balanced_accuracy'):.3f}",
    "mnv2_nat": f"{g('MobileNetV2','natural','none','balanced_accuracy'):.3f}",
    "cnn_nat": f"{g('CNN','natural','none','balanced_accuracy'):.3f}",
    "svm10": f"{g('SVM','10','none','balanced_accuracy'):.3f}",
    "mnv2_10": f"{g('MobileNetV2','10','none','balanced_accuracy'):.3f}",
    "svm20_rec": f"{g('SVM','20','none','recall_defective'):.3f}",
    "svm20_ok": f"{g('SVM','20','none','recall_ok'):.3f}",
    "svm20_ba": f"{g('SVM','20','none','balanced_accuracy'):.3f}",
    "svm20_acc": f"{g('SVM','20','none','accuracy'):.3f}",
    "svm20_auc": f"{g('SVM','20','none','roc_auc'):.3f}",
    "svm20_tm": f"{g('SVM','20','threshold_moving','balanced_accuracy'):.3f}",
    "svm20_tm_rec": f"{g('SVM','20','threshold_moving','recall_defective'):.3f}",
    "mnv2_20_rec": f"{g('MobileNetV2','20','none','recall_defective'):.3f}",
    "mnv2_20_ok": f"{g('MobileNetV2','20','none','recall_ok'):.3f}",
    "mnv2_20_auc": f"{g('MobileNetV2','20','none','roc_auc'):.3f}",
    "rf3_rec": f"{g('RandomForest','3','none','recall_defective'):.3f}",
    "rf20_auc": f"{g('RandomForest','20','none','roc_auc'):.3f}",
    "rf20_tm": f"{g('RandomForest','20','threshold_moving','balanced_accuracy'):.3f}",
    "mnv2_tm_mean": f"{ovm.loc[('MobileNetV2','threshold_moving'),'balanced_accuracy']:.3f}",
    "mnv2_tm_rec": f"{ovm.loc[('MobileNetV2','threshold_moving'),'recall_defective']:.3f}",
    "mnv2_tm_ok": f"{ovm.loc[('MobileNetV2','threshold_moving'),'recall_ok']:.3f}",
    "svm_tm_mean": f"{ovm.loc[('SVM','threshold_moving'),'balanced_accuracy']:.3f}",
    "mnv2_aug_ok": f"{ovm.loc[('MobileNetV2','augmentation_oversampling'),'recall_ok']:.3f}",
    "mnv2_aug_rank": f"{st[(st.model=='MobileNetV2')&(st.method=='augmentation_oversampling')].mean_rank.iloc[0]:.2f}",
    "final_mnv2_ba": f"{card['pipelines']['mobilenet']['test_metrics']['balanced_accuracy']:.3f}",
    "main_runs": f"{len(pd.concat([pd.read_csv(TAB / f'results_{n}.csv') for n in ('classical','classical_extra','mobilenet','cnn') if (TAB / f'results_{n}.csv').exists()]).query('tag==\"main\"')):,}",
}
p_def = 18 / (363 + 18)
rec, spec = float(C["svm20_rec"]), float(C["svm20_ok"])
C["svm20_prod_acc"] = f"{100 * ((1 - p_def) * spec + p_def * rec):.1f}%"
C["svm20_miss"] = f"{round(100 * (1 - rec))}"
out["C"] = C
# Every mean (and SD) of the main experiment, keyed "model|ratio|method|metric".
M = {}
metrics = ["accuracy", "balanced_accuracy", "recall_defective", "recall_ok", "precision_defective",
           "f1_defective", "g_mean", "mcc", "roc_auc", "pr_auc", "threshold"]
for r in main.itertuples():
    for m in metrics:
        if hasattr(r, m):
            M[f"{r.model}|{r.ratio}|{r.method}|{m}"] = round(float(getattr(r, m)), 4)
            sd = getattr(r, m + "_sd", None)
            if sd is not None and sd == sd:
                M[f"{r.model}|{r.ratio}|{r.method}|{m}_sd"] = round(float(sd), 4)
out["M"] = M
out["extra_params"] = json.loads((TAB / "extra_params.json").read_text())
out["svm_params"] = json.loads((TAB / "svm_params.json").read_text())
out["lr_tuning"] = pd.read_csv(TAB / "logreg_tuning.csv").round(4).values.tolist()
out["knn_tuning"] = pd.read_csv(TAB / "knn_tuning.csv").round(4).values.tolist()
raw = pd.concat([pd.read_csv(TAB / f"results_{n}.csv") for n in ("classical_extra",)])
raw["ratio"] = raw["ratio"].astype(str)
kk = raw[raw.tag.str.startswith("knnk_")].groupby(["tag", "ratio", "method"])["balanced_accuracy"].mean()
out["knn_k"] = {"|".join(k): round(float(v), 4) for k, v in kk.items()}
ov_models = ov.groupby("model")["balanced_accuracy"].max()
out["best_imbalanced_per_model"] = {m: [ov[(ov.model == m)].sort_values("balanced_accuracy").iloc[-1].method,
                                        round(float(v), 4)] for m, v in ov_models.items()}
(CFG.output_dir / "report_data.json").write_text(json.dumps(out, indent=1))
print(C)
