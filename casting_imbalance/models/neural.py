"""
neural.py
---------
Neural-architecture classifiers.

    ImageAugmenter       - random geometric / photometric transformation of one
                           grayscale image (used to build augmented copies).
    MobileNetV2Embedder  - ImageNet-pretrained MobileNetV2 used as a frozen
                           feature extractor (transfer learning).
    TransferHead         - small trainable dense head on top of the embeddings.
    SimpleCNN            - compact CNN trained end-to-end from scratch.
"""

import os
from typing import Dict, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

from ..config import CFG, Config
from .base import BaseClassifier


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow for repeatable neural training."""
    tf.keras.utils.set_random_seed(seed)


class ImageAugmenter:
    """Label-preserving augmentation for top-view impeller images.

    The impeller is roughly rotation-symmetric and photographed from above, so
    arbitrary rotations and flips produce realistic new views; small shifts and
    brightness/contrast changes mimic placement and lighting variation.
    """

    def __init__(self, max_shift: float = 0.04, max_brightness: float = 0.10,
                 max_contrast: float = 0.10):
        self.max_shift = max_shift
        self.max_brightness = max_brightness
        self.max_contrast = max_contrast

    def __call__(self, img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        h, w = img.shape[:2]
        angle = rng.uniform(0, 360)
        tx, ty = rng.uniform(-self.max_shift, self.max_shift, 2) * w
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        M[:, 2] += (tx, ty)
        out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if rng.rand() < 0.5:
            out = cv2.flip(out, 1)
        alpha = 1.0 + rng.uniform(-self.max_contrast, self.max_contrast)   # contrast
        beta = 255 * rng.uniform(-self.max_brightness, self.max_brightness)  # brightness
        return np.clip(alpha * out.astype(np.float32) + beta, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
#                       Transfer learning: MobileNetV2
# --------------------------------------------------------------------------
class MobileNetV2Embedder:
    """Frozen ImageNet MobileNetV2 that maps a 224x224 image to 1,280 features.

    Grayscale images are replicated into three identical channels because the
    network was pretrained on RGB images, then scaled to [-1, 1] with the
    official MobileNetV2 ``preprocess_input`` function.
    """

    def __init__(self, config: Config = CFG):
        self.cfg = config
        weights = (str(config.mobilenet_weights)
                   if config.mobilenet_weights.exists() else "imagenet")
        size = config.mobilenet_size
        self.backbone = tf.keras.applications.MobileNetV2(
            input_shape=(size, size, 3), include_top=False,
            weights=weights, pooling="avg")
        self.backbone.trainable = False

    @staticmethod
    def to_network_input(gray_images: np.ndarray) -> np.ndarray:
        rgb = np.repeat(gray_images[..., None], 3, axis=-1).astype(np.float32)
        return tf.keras.applications.mobilenet_v2.preprocess_input(rgb)

    def embed(self, gray_images: np.ndarray, batch_size: int = 64) -> np.ndarray:
        x = self.to_network_input(gray_images)
        return self.backbone.predict(x, batch_size=batch_size, verbose=0)


class TransferHead(BaseClassifier):
    """Dense(128, ReLU) -> Dropout(0.3) -> Dense(1, sigmoid) on embeddings."""

    name = "MobileNetV2"

    def __init__(self, input_dim: int = 1280, seed: int = 42, config: Config = CFG):
        self.input_dim, self.seed, self.cfg = input_dim, seed, config
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[dict] = None

    def _build(self) -> tf.keras.Model:
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ], name="transfer_head")
        model.compile(optimizer=optimizers.Adam(self.cfg.learning_rate),
                      loss="binary_crossentropy",
                      metrics=[tf.keras.metrics.AUC(name="auc")])
        return model

    def fit(self, X, y, class_weight=None, X_val=None, y_val=None):
        set_global_seed(self.seed)
        self.model = self._build()
        # Early stopping on validation ROC-AUC: AUC does not depend on the
        # class prior, so it is a fair criterion when the training and
        # validation class distributions differ.
        stop = callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                       patience=self.cfg.patience,
                                       restore_best_weights=True)
        h = self.model.fit(X, y.astype(np.float32), validation_data=(X_val, y_val),
                           epochs=self.cfg.max_epochs, batch_size=self.cfg.batch_size,
                           class_weight=class_weight, callbacks=[stop],
                           shuffle=True, verbose=0)
        self.history = h.history
        return self

    def predict_proba(self, X):
        return self.model.predict(X, batch_size=256, verbose=0).ravel()


# --------------------------------------------------------------------------
#                       End-to-end CNN trained from scratch
# --------------------------------------------------------------------------
class SimpleCNN(BaseClassifier):
    """Four convolutional blocks + global average pooling.

    Random flips, rotations, shifts and contrast changes are applied on the
    fly to every training batch (Keras preprocessing layers are inactive at
    inference time).
    """

    name = "CNN"

    def __init__(self, size: int = CFG.cnn_size, seed: int = 42, config: Config = CFG):
        self.size, self.seed, self.cfg = size, seed, config
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[dict] = None

    def build(self) -> tf.keras.Model:
        augment = models.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.5, fill_mode="reflect"),
            layers.RandomTranslation(0.04, 0.04, fill_mode="reflect"),
            layers.RandomContrast(0.1),
        ], name="augmentation")
        inputs = layers.Input(shape=(self.size, self.size, 1))
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = augment(x)
        for filters in (32, 64, 128, 128):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            # momentum 0.9 lets the moving statistics follow the weights
            # closely on a small dataset (few optimisation steps per epoch).
            x = layers.BatchNormalization(momentum=0.9)(x)
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs, outputs, name="simple_cnn")
        model.compile(optimizer=optimizers.Adam(self.cfg.cnn_learning_rate),
                      loss="binary_crossentropy",
                      metrics=[tf.keras.metrics.AUC(name="auc")])
        return model

    def fit(self, X, y, class_weight=None, X_val=None, y_val=None):
        set_global_seed(self.seed)
        self.model = self.build()
        # Training from scratch is noisier than training a small head, so the
        # learning rate is halved whenever validation loss stops improving and
        # the weights with the lowest validation loss are restored at the end.
        stop = callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                       patience=self.cfg.cnn_patience,
                                       restore_best_weights=True)
        reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-5)
        h = self.model.fit(X[..., None], y.astype(np.float32),
                           validation_data=(X_val[..., None], y_val),
                           epochs=self.cfg.cnn_max_epochs, batch_size=self.cfg.batch_size,
                           class_weight=class_weight, callbacks=[stop, reduce],
                           shuffle=True, verbose=0)
        self.history = h.history
        return self

    def predict_proba(self, X):
        return self.model.predict(X[..., None], batch_size=64, verbose=0).ravel()
