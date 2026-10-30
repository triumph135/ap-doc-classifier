"""Score a model on labelled documents it did not train on.

    python evaluate_holdout.py                       # the holdout rows of data/labels.csv
    python evaluate_holdout.py --labels data/vendor_check.csv --all
                                                     # every row of another labels file, for
                                                     # example documents from unseen vendors
    python evaluate_holdout.py --model models/versions/<id>/model.joblib

When the labels file has a group column the results are also broken down by group.
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
    ap.add_argument("--all", action="store_true", help="use every row, not only the holdout rows")
    ap.add_argument("--threshold", type=float, default=config.REVIEW_THRESHOLD,
                    help="review threshold used for the manual review summary")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    clf = Classifier(args.model)
    for w in clf.warnings:
        print("WARNING:", w)
    df = load_labels(args.labels)
    if not args.all:
        df = df[df["split"] == "holdout"]
    df = df.reset_index(drop=True)
    if df.empty:
        raise SystemExit("No documents to evaluate. Run: python train_model.py --init-holdout 0.15 (or use --all)")
    unknown = sorted(set(df["label"]) - set(clf.classes))
    if unknown:
        print(f"NOTE: the model has never seen {unknown}; documents of those types count as wrong.")

    print(f"Evaluating {len(df)} documents with model {clf.meta.get('version', 'unknown')} (from {args.labels})")
    results = get_texts(df["abs_path"], [file_hash(p) for p in df["abs_path"]], clf.mode, args.workers)
    proba = clf.model.predict_proba([r["text"] for r in results])
    metrics, report = evaluate(df["label"], proba, clf.classes, [r["source"] for r in results], "Evaluation")
    print(report)

    out = df[["path", "label", "group"]].copy()
    out["predicted"] = [clf.classes[i] for i in proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1).round(4)
    out["correct"] = out["label"] == out["predicted"]
    out["sent_to_review"] = out["confidence"] < args.threshold

    auto = out[~out["sent_to_review"]]
    print(f"\nAt the {args.threshold:.2f} review threshold: {len(auto)} of {len(out)} documents "
          f"({len(auto) / len(out):.0%}) would be handled automatically, "
          f"{len(out) - len(auto)} sent to manual review.")
    print(f"Wrong answers that would get through automatically: {int((~auto['correct']).sum())}")

    if out["group"].str.strip().ne("").any():
        print("\nBy group (accuracy before the review threshold, then what happens at the threshold)")
        rows = []
        for g, sub in out.groupby(out["group"].replace("", "(none)")):
            kept = sub[~sub["sent_to_review"]]
            rows.append({"group": g, "documents": len(sub), "accuracy": round(sub["correct"].mean(), 3),
                         "wrong": int((~sub["correct"]).sum()), "sent to review": int(sub["sent_to_review"].sum()),
                         "wrong and not reviewed": int((~kept["correct"]).sum())})
        print(pd.DataFrame(rows).to_string(index=False))

    out.sort_values(["correct", "confidence"], ascending=[True, False]).to_csv("holdout_results.csv", index=False)
    print("\nSaved holdout_results.csv (mistakes first)")


if __name__ == "__main__":
    main()
