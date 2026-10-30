"""Read and validate the master labels file (data/labels.csv).

Columns: path, label (required); split ("holdout" or blank for train) and
group (optional, for example a vendor or site) are optional.
"""
from pathlib import Path

import pandas as pd

from . import config


def canonical_label(raw):
    key = " ".join(str(raw).strip().lower().split())
    return config.LABEL_ALIASES.get(key)


def resolve_path(p):
    path = Path(p)
    return path if path.is_absolute() else config.ROOT / path


def read_raw(csv_path):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = {"path", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} must have columns path,label (missing: {sorted(missing)})")
    return df[df["path"].str.strip() != ""].copy()


def load_labels(csv_path=None, skip_missing=False):
    csv_path = Path(csv_path or config.LABELS_CSV)
    df = read_raw(csv_path)
    for col in ("split", "group"):
        if col not in df.columns:
            df[col] = ""
    df["label_raw"] = df["label"]
    df["label"] = df["label_raw"].map(canonical_label)

    bad = df[df["label"].isna()]
    if len(bad):
        counts = bad["label_raw"].value_counts().to_dict()
        hint = ""
        if any(str(k).strip().lower() == "material receipt" for k in counts):
            hint = ("\n'Material Receipt' is ambiguous. Label each document 'BOL Purchase' "
                    "or 'PO Receipt'; the Cargas document type is added automatically.")
        raise ValueError(
            f"{len(bad)} rows in {csv_path} have labels the model does not recognise: {counts}\n"
            f"Accepted labels: {config.CLASSES} (plus aliases in docclf/config.py).{hint}"
        )

    df["split"] = df["split"].str.strip().str.lower().replace("", "train")
    invalid = set(df["split"]) - {"train", "holdout"}
    if invalid:
        raise ValueError(f"split must be blank, 'train' or 'holdout', found: {sorted(invalid)}")

    df["abs_path"] = df["path"].map(resolve_path)
    gone = ~df["abs_path"].map(lambda p: p.exists())
    if gone.any():
        names = df.loc[gone, "path"].head(10).tolist()
        if not skip_missing:
            raise FileNotFoundError(
                f"{int(gone.sum())} files listed in {csv_path} do not exist, for example: {names}\n"
                "Fix the paths or run with --skip-missing."
            )
        print(f"WARNING: skipping {int(gone.sum())} missing files, for example: {names}")
        df = df[~gone]
    return df.reset_index(drop=True)
