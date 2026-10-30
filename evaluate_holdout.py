"""Evaluate a model on the holdout documents in data/labels.csv.

    python evaluate_holdout.py                      # the promoted model
    python evaluate_holdout.py --model models/versions/<id>/model.joblib
"""
import argparse

import pandas as pd

from docclf import config
from docclf.classify import Classifier
from docclf.evaluation import evaluate
from docclf.extraction import file_hash, get_texts
from docclf.labels import load_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--labels", default=str(config.LABELS_CSV))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    clf = Classifier(args.model)
    df = load_labels(args.labels)
    df = df[df["split"] == "holdout"].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No holdout documents. Run: python train_model.py --init-holdout 0.15")
    print(f"Evaluating {len(df)} holdout documents with model {clf.meta.get('version', 'unknown')}")
    results = get_texts(df["abs_path"], [file_hash(p) for p in df["abs_path"]], clf.mode, args.workers)
    texts = [r["text"] for r in results]
    proba = clf.model.predict_proba(texts)
    metrics, report = evaluate(df["label"], proba, clf.classes, [r["source"] for r in results], "Holdout set")
    print(report)

    out = df[["path", "label"]].copy()
    out["predicted"] = [clf.classes[i] for i in proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1).round(4)
    out["correct"] = out["label"] == out["predicted"]
    out.sort_values(["correct", "confidence"], ascending=[True, False]).to_csv("holdout_results.csv", index=False)
    print("\nSaved holdout_results.csv (mistakes first)")


if __name__ == "__main__":
    main()
