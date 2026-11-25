"""
inference.py
------------
Deployable end-to-end pipelines used by the inspection application.

    ClassicalPipeline   image -> preprocessing -> HOG+LBP+GLCM -> scaler+PCA -> SVM
    MobileNetPipeline   image -> preprocessing -> MobileNetV2 -> dense head
                        (+ Grad-CAM heat-map explaining the decision)
    InspectionService   loads both pipelines and returns a structured verdict.

Each pipeline stores its own decision threshold, chosen on the validation set,
so the application uses exactly the operating point reported in the study.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import joblib
import numpy as np

from .config import CFG, Config
from .features import ALL_GROUPS, HandcraftedFeatureExtractor
from .models.classical import FeatureTransformer, SVMClassifier
from .preprocessing import ImagePreprocessor


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded file bytes into an 8-bit grayscale image."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("The file could not be decoded as an image.")
    return image


class ClassicalPipeline:
    """Handcrafted-feature SVM pipeline with a stored decision threshold."""

    name = "SVM (HOG + LBP + GLCM)"

    def __init__(self, config: Config = CFG):
        self.cfg = config
        self.pre = ImagePreprocessor(config.classical_size, "gaussian", config)
        self.extractor = HandcraftedFeatureExtractor(self.pre, ALL_GROUPS, config)
        self.transformer: Optional[FeatureTransformer] = None
        self.svm: Optional[SVMClassifier] = None
        self.class_weight: Optional[Dict[int, float]] = None
        self.threshold = 0.5

    def features(self, images: List[np.ndarray]) -> np.ndarray:
        rows = []
        for img in images:
            p = self.pre.process(img)
            rows.append(np.concatenate([self.extractor.hog_features(p),
                                        self.extractor.lbp_features(p),
                                        self.extractor.glcm_features(p)]))
        return np.asarray(rows, dtype=np.float32)

    def fit(self, images, y, C, gamma, class_weight=None):
        X = self.features(images)
        self.transformer = FeatureTransformer().fit(X)
        self.class_weight = class_weight
        self.svm = SVMClassifier(C=C, gamma=gamma).fit(self.transformer.transform(X), y,
                                                       class_weight=class_weight)
        return self

    def predict_proba(self, images: List[np.ndarray]) -> np.ndarray:
        return self.svm.predict_proba(self.transformer.transform(self.features(images)))

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({"transformer": self.transformer, "svm": self.svm,
                     "threshold": self.threshold, "class_weight": self.class_weight},
                    folder / "classical_pipeline.joblib")

    @classmethod
    def load(cls, folder: Path, config: Config = CFG) -> "ClassicalPipeline":
        obj = cls(config)
        state = joblib.load(folder / "classical_pipeline.joblib")
        obj.transformer, obj.svm = state["transformer"], state["svm"]
        obj.threshold, obj.class_weight = state["threshold"], state["class_weight"]
        return obj


class MobileNetPipeline:
    """MobileNetV2 feature extractor + trained dense head, with Grad-CAM."""

    name = "MobileNetV2 (transfer learning)"

    def __init__(self, config: Config = CFG):
        import tensorflow as tf   # imported lazily so the SVM path works without TF
        from .models.neural import MobileNetV2Embedder
        self.tf = tf
        self.cfg = config
        self.pre = ImagePreprocessor(config.mobilenet_size, "gaussian", config)
        self.embedder = MobileNetV2Embedder(config)
        self.head = None
        self.threshold = 0.5
        self._full_model = None

    def preprocess(self, images: List[np.ndarray]) -> np.ndarray:
        return np.stack([self.pre.process(img) for img in images])

    def predict_proba(self, images: List[np.ndarray]) -> np.ndarray:
        emb = self.embedder.embed(self.preprocess(images))
        return self.head.predict(emb, verbose=0).ravel()

    # ------------------------------------------------------------ Grad-CAM
    def _build_full_model(self):
        """Chain backbone (to its last conv activation) and head in one graph."""
        tf = self.tf
        backbone = self.embedder.backbone
        conv = backbone.get_layer("out_relu").output
        pooled = tf.keras.layers.GlobalAveragePooling2D()(conv)
        out = self.head(pooled)
        self._full_model = tf.keras.Model(backbone.input, [conv, out])

    def grad_cam(self, image: np.ndarray) -> np.ndarray:
        """Return a [0, 1] heat-map (same size as the input) of the regions that
        increased the predicted probability of 'Defective'."""
        tf = self.tf
        if self._full_model is None:
            self._build_full_model()
        x = self.embedder.to_network_input(self.preprocess([image]))
        with tf.GradientTape() as tape:
            conv, pred = self._full_model(x, training=False)
            score = pred[:, 0]
        grads = tape.gradient(score, conv)[0]                  # (h, w, c)
        weights = tf.reduce_mean(grads, axis=(0, 1))           # importance of each map
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
        cam = cam / (cam.max() + 1e-8)
        return cv2.resize(cam, (image.shape[1], image.shape[0]))

    # ----------------------------------------------------------- persistence
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.head.save(folder / "mobilenet_head.keras")
        with open(folder / "mobilenet_meta.json", "w") as f:
            json.dump({"threshold": self.threshold}, f)

    @classmethod
    def load(cls, folder: Path, config: Config = CFG) -> "MobileNetPipeline":
        obj = cls(config)
        obj.head = obj.tf.keras.models.load_model(folder / "mobilenet_head.keras")
        with open(folder / "mobilenet_meta.json") as f:
            obj.threshold = json.load(f)["threshold"]
        return obj


class InspectionService:
    """Facade used by the GUI and the command-line tool."""

    def __init__(self, model_dir: Path = CFG.model_dir, use_mobilenet: bool = True):
        self.pipelines = {"classical": ClassicalPipeline.load(model_dir)}
        if use_mobilenet:
            self.pipelines["mobilenet"] = MobileNetPipeline.load(model_dir)
        card = model_dir / "model_card.json"
        self.model_card = json.loads(card.read_text()) if card.exists() else {}

    def inspect(self, image: np.ndarray) -> Dict[str, dict]:
        results = {}
        for key, pipe in self.pipelines.items():
            p = float(pipe.predict_proba([image])[0])
            results[key] = {"model": pipe.name, "score": p, "threshold": pipe.threshold,
                            "verdict": "Defective" if p >= pipe.threshold else "OK"}
        return results
