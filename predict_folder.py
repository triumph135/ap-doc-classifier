"""Classify every document in a folder and write predictions.csv.

    python predict_folder.py "C:\path\to\folder"
"""
import argparse
from pathlib import Path

import pandas as pd

from docclf import config
from docclf.classify import Classifier
from docclf.extraction import SUPPORTED_EXT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--threshold", type=float, default=config.REVIEW_THRESHOLD)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    clf = Classifier()
    for w in clf.warnings:
        print("WARNING:", w)
    if clf.missing_classes:
        print(f"NOTE: this model has no examples of {clf.missing_classes}; those documents cannot be recognised.")

    rows = []
    for f in sorted(Path(args.folder).rglob("*")):
        if f.suffix.lower() not in SUPPORTED_EXT:
            continue
        r = clf.predict_file(f, args.threshold, use_cache=True)
        rows.append({"file": f.name, "path": str(f), **r})
        print(f"{f.name} -> {r['final']} (p={r['confidence']:.2f})")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved {args.out} ({len(rows)} documents)")


if __name__ == "__main__":
    main()
