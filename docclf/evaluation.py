"""Metrics and a readable report for a set of predictions."""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def evaluate(y_true, proba, classes, sources, title="Evaluation"):
    """Return (metrics dict, report text). `proba` columns follow `classes`.
    A true label the model cannot predict simply counts as wrong."""
    y_true = np.asarray(list(y_true))
    sources = np.asarray(list(sources))
    proba = np.asarray(proba)
    classes = np.asarray(list(classes))
    pred = classes[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    correct = pred == y_true
    labels = sorted(set(y_true) | set(pred))

    by_source = {s: {"n": int((sources == s).sum()),
                     "accuracy": float(correct[sources == s].mean())}
                 for s in sorted(set(sources))}
    by_class_source = {}
    for label in sorted(set(y_true)):
        for s in sorted(set(sources)):
            m = (y_true == label) & (sources == s)
            if m.any():
                by_class_source[f"{label} | {s}"] = {"n": int(m.sum()), "accuracy": float(correct[m].mean())}

    thresholds = []
    for t in THRESHOLDS:
        auto = conf >= t
        thresholds.append({
            "threshold": t,
            "auto_processed": float(auto.mean()),
            "accuracy_of_auto": float(correct[auto].mean()) if auto.any() else None,
            "errors_let_through": int((~correct[auto]).sum()),
        })

    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "per_class": classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0),
        "by_source": by_source,
        "by_class_source": by_class_source,
        "thresholds": thresholds,
    }

    cm = pd.DataFrame(confusion_matrix(y_true, pred, labels=labels), index=labels, columns=labels)
    lines = [
        f"=== {title} ===",
        f"Documents: {metrics['n']}   Accuracy: {metrics['accuracy']:.3f}   Macro F1: {metrics['macro_f1']:.3f}",
        "",
        classification_report(y_true, pred, labels=labels, zero_division=0),
        "Confusion matrix (rows = true, columns = predicted)",
        cm.to_string(),
        "",
        "Accuracy by text source (native = embedded PDF text, ocr = scanned page)",
    ]
    for key, v in by_source.items():
        lines.append(f"  {key:<8} n={v['n']:<5} accuracy={v['accuracy']:.3f}")
    lines.append("Accuracy by class and text source (large gaps can mean the model reads the scan type, not the content)")
    for key, v in by_class_source.items():
        lines.append(f"  {key:<28} n={v['n']:<5} accuracy={v['accuracy']:.3f}")
    lines += ["", "Review threshold trade off (documents at or above the threshold are processed automatically)"]
    for row in thresholds:
        acc = "n/a" if row["accuracy_of_auto"] is None else f"{row['accuracy_of_auto']:.3f}"
        lines.append(f"  >= {row['threshold']:.2f}  automatic={row['auto_processed']:.1%}  "
                     f"accuracy of automatic={acc}  errors let through={row['errors_let_through']}")
    return metrics, "\n".join(lines)
