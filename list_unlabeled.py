"""List documents that are not yet in data/labels.csv, ready to label.

    python list_unlabeled.py [folder]      # default folder: data/docs

Writes data/unlabeled.csv with a blank label column. Fill in the label (BOL Invoice,
BOL Purchase, PO Receipt or PO Invoice), then append the rows to data/labels.csv.
"""
import sys
from pathlib import Path

import pandas as pd

from docclf import config
from docclf.extraction import SUPPORTED_EXT
from docclf.labels import read_raw, resolve_path


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else config.DATA_DIR / "docs"
    known = {resolve_path(p).resolve() for p in read_raw(config.LABELS_CSV)["path"]} if config.LABELS_CSV.exists() else set()
    rows = []
    for f in sorted(folder.rglob("*")):
        if f.suffix.lower() in SUPPORTED_EXT and f.resolve() not in known:
            try:
                rel = f.resolve().relative_to(config.ROOT).as_posix()
            except ValueError:
                rel = str(f.resolve())
            rows.append({"path": rel, "label": "", "split": "", "group": ""})
    out = config.DATA_DIR / "unlabeled.csv"
    pd.DataFrame(rows, columns=["path", "label", "split", "group"]).to_csv(out, index=False)
    print(f"{len(rows)} unlabeled documents written to {out}")


if __name__ == "__main__":
    main()
