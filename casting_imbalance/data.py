"""
data.py
-------
Dataset indexing, stratified splitting and construction of controlled
class-imbalance scenarios.

Classes
    CastingDataset          - scans the image folders and builds an index.
    DataSplitter            - stratified train / validation / test split.
    ImbalanceScenarioBuilder- creates imbalanced TRAINING subsets only.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CFG, Config

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp")


class CastingDataset:
    """Index of every image file with its integer label (0 = OK, 1 = Defective)."""

    def __init__(self, config: Config = CFG):
        self.cfg = config
        self.index: pd.DataFrame = self._scan()

    def _scan(self) -> pd.DataFrame:
        """Walk the two class folders and return a sorted DataFrame index."""
        records = []
        for label, folder in enumerate(self.cfg.class_folders):
            folder_path = self.cfg.data_dir / folder
            if not folder_path.is_dir():
                raise FileNotFoundError(
                    f"Missing class folder: {folder_path}\n"
                    "Download the Kaggle dataset 'ravirajsinh45/real-life-industrial-"
                    "dataset-of-casting-product', unzip it, and copy the UNAUGMENTED "
                    "casting_512x512 folder (781 def_front + 519 ok_front images) into "
                    f"{self.cfg.project_root / 'data'}. Nested folders are found "
                    "automatically; alternatively set the CASTING_DATA_DIR environment "
                    "variable to the folder that contains def_front and ok_front.")
            for path in sorted(folder_path.iterdir()):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append({"path": str(path), "filename": path.name,
                                    "label": label,
                                    "class_name": self.cfg.class_names[label]})
        # Sorting by filename makes the index (and every split) deterministic.
        return pd.DataFrame(records).sort_values("filename").reset_index(drop=True)

    def class_counts(self) -> pd.Series:
        return self.index["class_name"].value_counts()

    def __len__(self) -> int:
        return len(self.index)


class DataSplitter:
    """Stratified 70 / 15 / 15 split that is created once and saved to disk.

    The validation and test sets are never resampled or augmented, so every
    experiment is evaluated on exactly the same untouched images.
    """

    def __init__(self, config: Config = CFG):
        self.cfg = config

    def split(self, index: pd.DataFrame) -> pd.DataFrame:
        test_frac = self.cfg.test_fraction
        val_frac = self.cfg.val_fraction / (1.0 - test_frac)   # of remainder
        train_val, test = train_test_split(
            index, test_size=test_frac, stratify=index["label"],
            random_state=self.cfg.global_seed)
        train, val = train_test_split(
            train_val, test_size=val_frac, stratify=train_val["label"],
            random_state=self.cfg.global_seed)
        out = index.copy()
        out["split"] = "train"
        out.loc[val.index, "split"] = "val"
        out.loc[test.index, "split"] = "test"
        return out

    def load_or_create(self, index: pd.DataFrame) -> pd.DataFrame:
        """Reuse the saved split if present so results stay comparable."""
        path = self.cfg.output_dir / "tables" / "data_split.csv"
        if path.exists():
            split_df = pd.read_csv(path)
        else:
            split_df = self.split(index)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Store only file names so the split file works on any computer.
            split_df.drop(columns="path").to_csv(path, index=False)
        # Rebuild absolute paths from the configured data folder.
        folders = dict(enumerate(self.cfg.class_folders))
        split_df["path"] = [str(self.cfg.data_dir / folders[int(l)] / f)
                            for l, f in zip(split_df["label"], split_df["filename"])]
        return split_df


class ImbalanceScenarioBuilder:
    """Creates a controlled imbalance in the training set.

    For a target ratio r (OK : Defective), all OK training images are kept and
    round(n_ok / r) Defective images are drawn at random WITHOUT replacement.
    A different seed gives a different minority sub-sample, which is how the
    repeated runs estimate the variability caused by *which* defects are seen.
    """

    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df.reset_index(drop=True)

    def build(self, ratio: Union[str, int], seed: int) -> pd.DataFrame:
        if ratio == "natural":
            return self.train_df.copy()
        ratio = float(ratio)
        ok = self.train_df[self.train_df["label"] == 0]
        defective = self.train_df[self.train_df["label"] == 1]
        n_def = int(round(len(ok) / ratio))
        if n_def > len(defective):
            raise ValueError("Requested ratio needs more defective images "
                             "than are available.")
        rng = np.random.RandomState(seed)
        chosen = defective.iloc[rng.choice(len(defective), n_def, replace=False)]
        return pd.concat([ok, chosen]).sort_values("filename").reset_index(drop=True)

    @staticmethod
    def describe(df: pd.DataFrame) -> Dict[str, float]:
        n_ok = int((df["label"] == 0).sum())
        n_def = int((df["label"] == 1).sum())
        return {"n_ok": n_ok, "n_defective": n_def,
                "imbalance_ratio_ok_to_def": n_ok / max(n_def, 1),
                "minority_share": min(n_ok, n_def) / (n_ok + n_def)}
