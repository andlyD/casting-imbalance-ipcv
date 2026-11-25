"""
predict.py
----------
Command-line inspection of one image or a folder of images.

    python predict.py path/to/image.jpeg
    python predict.py path/to/folder --no-mobilenet --csv results.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from casting_imbalance.config import CFG
from casting_imbalance.inference import InspectionService
from casting_imbalance.preprocessing import load_grayscale

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Casting defect inspection")
    parser.add_argument("target", type=Path, help="image file or folder")
    parser.add_argument("--no-mobilenet", action="store_true",
                        help="use only the SVM pipeline (no TensorFlow needed)")
    parser.add_argument("--csv", type=Path, help="optional output CSV file")
    args = parser.parse_args()

    paths = ([args.target] if args.target.is_file() else
             sorted(p for p in args.target.iterdir() if p.suffix.lower() in EXTENSIONS))
    service = InspectionService(CFG.model_dir, use_mobilenet=not args.no_mobilenet)
    rows = []
    for p in paths:
        res = service.inspect(load_grayscale(str(p)))
        row = {"file": p.name}
        for key, r in res.items():
            row[f"{key}_verdict"] = r["verdict"]
            row[f"{key}_score"] = round(r["score"], 4)
        rows.append(row)
        print(row)
    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print("Saved", args.csv)


if __name__ == "__main__":
    main()
