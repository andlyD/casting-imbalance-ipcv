"""
train_final_models.py
---------------------
Step 6 of the pipeline: train the two deployable pipelines used by app.py
and predict.py, evaluate them once on the untouched test set and write a
model card.

Both pipelines are trained on the full training split with the imbalance-
handling method recommended by the experiments, and their decision threshold
is fixed on the validation set.

Run from the project root:
    python scripts/train_final_models.py --svm-method threshold_moving \
                                         --mobilenet-method threshold_moving
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from casting_imbalance.config import CFG
from casting_imbalance.data import CastingDataset, DataSplitter
from casting_imbalance.evaluation import Evaluator
from casting_imbalance.imbalance import ThresholdOptimizer, make_strategy
from casting_imbalance.inference import ClassicalPipeline, MobileNetPipeline
from casting_imbalance.models.neural import TransferHead
from casting_imbalance.preprocessing import load_grayscale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svm-method", default="threshold_moving",
                        choices=["none", "class_weight", "threshold_moving"])
    parser.add_argument("--mobilenet-method", default="threshold_moving",
                        choices=["none", "class_weight", "threshold_moving"])
    args = parser.parse_args()

    CFG.ensure_dirs()
    split_df = DataSplitter().load_or_create(CastingDataset().index)
    parts = {s: split_df[split_df.split == s] for s in ("train", "val", "test")}
    images = {s: [load_grayscale(p) for p in d["path"]] for s, d in parts.items()}
    labels = {s: d["label"].to_numpy() for s, d in parts.items()}
    card = {"training_images": int(len(labels["train"])), "pipelines": {}}

    # ------------------------------------------------------ classical (SVM)
    with open(CFG.output_dir / "tables" / "svm_params.json") as f:
        params = json.load(f)
    strategy = make_strategy(args.svm_method)
    svm = ClassicalPipeline().fit(images["train"], labels["train"], params["C"],
                                  params["gamma"], strategy.class_weight(labels["train"]))
    if strategy.tunes_threshold:
        svm.threshold = ThresholdOptimizer().fit(labels["val"], svm.predict_proba(images["val"]))
    svm.save(CFG.model_dir)
    m = Evaluator.compute(labels["test"], svm.predict_proba(images["test"]), svm.threshold)
    card["pipelines"]["classical"] = {"name": svm.name, "imbalance_handling": args.svm_method,
                                      "threshold": svm.threshold, "test_metrics": m}
    print("SVM test:", {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})

    # ------------------------------------------------ transfer (MobileNetV2)
    mnet = MobileNetPipeline()
    emb = {s: mnet.embedder.embed(mnet.preprocess(images[s])) for s in images}
    strategy = make_strategy(args.mobilenet_method)
    head = TransferHead(seed=CFG.global_seed).fit(
        emb["train"], labels["train"], class_weight=strategy.class_weight(labels["train"]),
        X_val=emb["val"], y_val=labels["val"])
    mnet.head = head.model
    if strategy.tunes_threshold:
        mnet.threshold = ThresholdOptimizer().fit(labels["val"], head.predict_proba(emb["val"]))
    mnet.save(CFG.model_dir)
    m = Evaluator.compute(labels["test"], head.predict_proba(emb["test"]), mnet.threshold)
    card["pipelines"]["mobilenet"] = {"name": mnet.name,
                                      "imbalance_handling": args.mobilenet_method,
                                      "threshold": mnet.threshold, "test_metrics": m}
    print("MobileNetV2 test:", {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})

    with open(CFG.model_dir / "model_card.json", "w") as f:
        json.dump(card, f, indent=2, default=float)

    # Grad-CAM examples for the report (two defective, two OK test images).
    import cv2
    import matplotlib.pyplot as plt
    from casting_imbalance.plotting import save_figure
    probs = head.predict_proba(emb["test"])
    picks = (list(np.where(labels["test"] == 1)[0][:2]) + list(np.where(labels["test"] == 0)[0][:2]))
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
    for j, i in enumerate(picks):
        img = images["test"][i]
        cam = mnet.grad_cam(img)
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        overlay = (0.6 * np.repeat(img[..., None], 3, -1) + 0.4 * heat).astype(np.uint8)
        truth = "Defective" if labels["test"][i] == 1 else "OK"
        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].set_title(f"True: {truth}", fontsize=8.5)
        axes[1, j].imshow(overlay)
        axes[1, j].set_title(f"P(defective) = {probs[i]:.3f}", fontsize=8.5)
        axes[0, j].axis("off"); axes[1, j].axis("off")
    save_figure(fig, CFG.output_dir / "figures" / "fig_gradcam.png")


if __name__ == "__main__":
    main()
