"""Train (or retrain) the AP document type classifier.

Everyday use:
    python train_model.py

One time setup, reserve a fixed holdout set that never trains the model:
    python train_model.py --init-holdout 0.15

Every run reads the whole master file data/labels.csv, trains on the classes
that have enough examples, evaluates with cross validation and the holdout set,
saves a dated model under models/versions/, and promotes it to models/latest.*
only if it is not worse than the current one on the holdout set.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from docclf import config
from docclf.evaluation import evaluate
from docclf.extraction import EXTRACTION_VERSION, file_hash, get_texts, normalize_text
from docclf.labels import canonical_label, load_labels, read_raw


def build_model(C=1.0):
    features = FeatureUnion([
        # whole words and word pairs, for example "bill of lading", "amount due"
        ("words", TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2), min_df=2,
                                  max_df=0.95, sublinear_tf=True, strip_accents="unicode")),
        # character pieces, which tolerate OCR typos such as "lnvoice"
        ("chars", TfidfVectorizer(preprocessor=normalize_text, analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=3, max_df=0.95, sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
    ])


def init_holdout(labels_path, fraction):
    raw = read_raw(labels_path)
    if "split" not in raw.columns:
        raw["split"] = ""
    if (raw["split"].str.strip().str.lower() == "holdout").any():
        sys.exit("A holdout set already exists. It is meant to stay fixed, so nothing was changed.")
    labels = raw["label"].map(canonical_label)
    if labels.isna().any():
        sys.exit("Some labels are not recognised. Fix them first (run python train_model.py to see which).")
    rng = np.random.RandomState(config.RANDOM_SEED)
    raw["split"] = "train"
    for _, idx in raw.groupby(labels).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        raw.loc[idx[:max(1, int(round(len(idx) * fraction)))], "split"] = "holdout"
    raw.to_csv(labels_path, index=False, encoding="utf-8")
    print(f"Marked {int((raw['split'] == 'holdout').sum())} of {len(raw)} documents as holdout in {labels_path}:")
    print(raw.assign(label=labels).groupby(["label", "split"]).size().unstack(fill_value=0))


def resolve_duplicates(df):
    """Identical files count once; identical files with different labels are dropped."""
    keep, conflicts = [], []
    for _, grp in df.groupby("sha1", sort=False):
        if grp["label"].nunique() > 1:
            conflicts.append(grp[["path", "label", "sha1"]])
            continue
        hold = grp[grp["split"] == "holdout"]
        keep.append((hold if len(hold) else grp).index[0])
    out = df.loc[keep].sort_index()
    dup_count = len(df) - len(out) - sum(len(c) for c in conflicts)
    if dup_count:
        print(f"Ignored {dup_count} exact duplicate files.")
    if conflicts:
        review_file = config.DATA_DIR / "label_conflicts.csv"
        pd.concat(conflicts).to_csv(review_file, index=False)
        print(f"WARNING: {len(conflicts)} sets of identical files carry different labels "
              f"({sum(len(c) for c in conflicts)} files). They are excluded from training. "
              f"Review list: {review_file}")
    return out


def run_cv(texts, y, groups, folds, C):
    classes = sorted(set(y))
    folds = min(folds, int(pd.Series(y).value_counts().min()))
    if groups is not None:
        folds = min(folds, len(set(groups)))  # every fold needs at least one whole group
    if folds < 2:
        return None, classes
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED)
        proba = cross_val_predict(build_model(C), texts, y, cv=cv, groups=groups, method="predict_proba")
    else:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED)
        proba = cross_val_predict(build_model(C), texts, y, cv=cv, method="predict_proba")
    return proba, classes


def group_table(oof, group):
    """Out of fold accuracy per group, plus the label mix, so a vendor that always shares one label stands out."""
    d = oof.assign(group=group.str.strip().replace("", "(none)"))
    lines = ["Out of fold accuracy by group (each group was predicted by models that never saw it)"]
    for g, sub in d.groupby("group"):
        mix = ", ".join(f"{k} {v}" for k, v in sub["label"].value_counts().items())
        lines.append(f"  {g:<16} n={len(sub):<5} accuracy={sub['correct'].mean():.3f}   ({mix})")
    return "\n".join(lines)


def holdout_macro_f1(model, texts, labels):
    proba = model.predict_proba(texts)
    pred = np.asarray(model.classes_)[proba.argmax(axis=1)]
    return f1_score(labels, pred, labels=sorted(set(labels)), average="macro", zero_division=0)


def decide_promotion(model, hold, args):
    latest = config.LATEST_MODEL
    if args.promote:
        return True, "forced with --promote"
    if not latest.exists():
        return True, "first model"
    if hold is None or not len(hold):
        return False, "no holdout set to compare against (use --promote to override)"
    try:
        old = joblib.load(latest)
        old_f1 = holdout_macro_f1(old, hold["text"].tolist(), hold["label"].tolist())
    except Exception as exc:
        return True, f"current model could not be evaluated ({exc})"
    new_f1 = holdout_macro_f1(model, hold["text"].tolist(), hold["label"].tolist())
    ok = new_f1 >= old_f1 - 0.005
    return ok, f"holdout macro F1 new {new_f1:.3f} vs current {old_f1:.3f}"


def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(config.LABELS_CSV), help="master labels file")
    ap.add_argument("--mode", choices=["auto", "ocr"], default="auto",
                    help="auto: native PDF text when present, else OCR. ocr: OCR every page (slower, consistent)")
    ap.add_argument("--workers", type=int, default=4, help="parallel OCR workers")
    ap.add_argument("--init-holdout", type=float, metavar="FRACTION", help="one time: reserve this share as holdout")
    ap.add_argument("--min-examples", type=int, default=config.MIN_EXAMPLES_PER_CLASS)
    ap.add_argument("--folds", type=int, default=config.CV_FOLDS)
    ap.add_argument("--C", type=float, default=1.0, help="logistic regression strength (higher = less regularised)")
    ap.add_argument("--skip-missing", action="store_true", help="skip labelled files that are not on disk")
    ap.add_argument("--promote", action="store_true", help="promote this model even if the holdout check says no")
    args = ap.parse_args()

    if args.init_holdout:
        return init_holdout(args.labels, args.init_holdout)

    started = time.time()
    df = load_labels(args.labels, skip_missing=args.skip_missing)
    print(f"Loaded {len(df)} labelled documents from {args.labels}")
    df["sha1"] = [file_hash(p) for p in df["abs_path"]]
    df = resolve_duplicates(df)

    print(f"Extracting text (mode={args.mode}, workers={args.workers}). First runs are slow because scans are OCR'd; "
          "results are cached, so later runs only process new files.")
    results = get_texts(df["abs_path"], df["sha1"], mode=args.mode, workers=args.workers)
    df["text"] = [r["text"] for r in results]
    df["source"] = [r["source"] for r in results]
    df["error"] = [r.get("error") for r in results]
    unreadable = df[(df["source"] == "error") | (df["text"].str.strip().str.len() < config.MIN_TEXT_CHARS)]
    if len(unreadable):
        print(f"WARNING: {len(unreadable)} documents have no readable text and are left out, for example:")
        for _, r in unreadable.head(10).iterrows():
            print(f"  {r['path']}  ({r['error'] or 'too little text'})")
        df = df.drop(unreadable.index)

    train = df[df["split"] == "train"].copy()
    hold = df[df["split"] == "holdout"].copy()
    hold = hold[~hold["sha1"].isin(train["sha1"])]

    counts = train["label"].value_counts()
    trainable = [c for c in config.CLASSES if counts.get(c, 0) >= args.min_examples]
    print("\nClass coverage (training documents):")
    for c in config.CLASSES:
        n = int(counts.get(c, 0))
        status = "trained" if c in trainable else (f"NOT trained, needs {args.min_examples}" if n else "NOT trained, no examples yet")
        print(f"  {c:<14} {n:>5}  {status}")
    if len(trainable) < 2:
        sys.exit(f"Need at least two classes with {args.min_examples}+ examples to train.")
    train = train[train["label"].isin(trainable)]
    hold = hold[hold["label"].isin(trainable)]
    print(f"Training on {len(train)} documents; holdout has {len(hold)} documents.\n")

    texts, y = train["text"].tolist(), train["label"].tolist()
    groups = None
    if train["group"].str.strip().ne("").any():
        groups = [g.strip() or f"row{i}" for i, g in enumerate(train["group"])]
        print("Using the group column so documents from the same group stay in the same fold.")

    reports, metrics = [], {}
    print("Cross validation...")
    proba, cv_classes = run_cv(texts, y, groups, args.folds, args.C)
    oof = None
    if proba is not None:
        kind = "whole groups held out" if groups is not None else "random folds"
        metrics["cross_validation"], text = evaluate(y, proba, cv_classes, train["source"],
                                                     f"{args.folds} fold cross validation ({kind})")
        reports.append(text)
        pred = np.asarray(cv_classes)[proba.argmax(axis=1)]
        oof = train[["path", "label", "source"]].assign(predicted=pred, confidence=proba.max(axis=1).round(4))
        oof["correct"] = oof["label"] == oof["predicted"]
        if groups is not None:
            reports.append(group_table(oof, train["group"]))
            print("Cross validation with random folds, for comparison...")
            proba_r, _ = run_cv(texts, y, None, args.folds, args.C)
            metrics["cross_validation_random_folds"], text_r = evaluate(
                y, proba_r, cv_classes, train["source"],
                f"{args.folds} fold cross validation (random folds: the same group can be in train and test)")
            reports.append(text_r)
    else:
        reports.append("Cross validation skipped: a class has too few examples.")

    print("Fitting final model...")
    model = build_model(args.C).fit(texts, y)

    if len(hold):
        hold_proba = model.predict_proba(hold["text"].tolist())
        metrics["holdout"], text = evaluate(hold["label"], hold_proba, model.classes_, hold["source"], "Holdout set")
        reports.append(text)
    else:
        reports.append("No holdout set. Run: python train_model.py --init-holdout 0.15")
    report_text = "\n\n".join(reports)
    print("\n" + report_text)

    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = config.MODELS_DIR / "versions" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": version,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "classes": [str(c) for c in model.classes_],
        "classes_without_examples": [c for c in config.CLASSES if c not in trainable],
        "documents_per_class": {k: int(v) for k, v in train["label"].value_counts().items()},
        "training_documents": int(len(train)),
        "holdout_documents": int(len(hold)),
        "extraction_mode": args.mode,
        "extraction_version": EXTRACTION_VERSION,
        "sklearn_version": sklearn.__version__,
        "git_commit": git_commit(),
        "labels_file_sha1": hashlib.sha1(open(args.labels, "rb").read()).hexdigest(),
        "C": args.C,
        "metrics": metrics,
    }
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=jsonable), encoding="utf-8")
    (out_dir / "report.txt").write_text(report_text, encoding="utf-8")
    if oof is not None:
        oof.sort_values(["correct", "confidence"], ascending=[True, False]).to_csv(out_dir / "cv_predictions.csv", index=False)

    promote, why = decide_promotion(model, hold, args)
    if promote:
        shutil.copy2(out_dir / "model.joblib", config.LATEST_MODEL)
        meta["promoted_at"] = datetime.now().isoformat(timespec="seconds")
        config.LATEST_META.write_text(json.dumps(meta, indent=2, default=jsonable), encoding="utf-8")
    print(f"\nSaved {out_dir}")
    print(("PROMOTED to models/latest.joblib: " if promote else "NOT promoted: ") + why)
    print(f"Done in {time.time() - started:.0f}s.")


if __name__ == "__main__":
    main()
