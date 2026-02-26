import pandas as pd
import joblib

from train_model import extract_text, clean_text  # reuse your functions
from sklearn.metrics import classification_report, confusion_matrix

def main():
    model = joblib.load("doc_type_model.joblib")
    df = pd.read_csv("data/holdout_eval_labels.csv")

    y_true = df["label"].tolist()
    y_pred = []
    probs = []

    print(f"Evaluating on {len(df)} holdout documents...")  

    for path in df["path"]:
        text = clean_text(extract_text(path))
        prob_vec = model.predict_proba([text])[0]
        classes = model.classes_
        best_idx = int(prob_vec.argmax())
        y_pred.append(classes[best_idx])
        probs.append(float(prob_vec[best_idx]))


    out = df.copy()
    out["pred"] = y_pred
    out["prob"] = probs
    out["correct"] = out["pred"] == out["label"]
    out.to_csv("holdout_results.csv", index=False)

    #print("Accuracy:", out["correct"].mean())
    #print("Wrong examples:", (~out["correct"]).sum())
    #print("\nMost confident wrong (if any):")
    #print(out[~out["correct"]].sort_values("prob", ascending=False).head(10)[["path","label","pred","prob"]])

    #print("\nLowest confidence correct:")
    #print(out[out["correct"]].sort_values("prob", ascending=True).head(10)[["path","label","pred","prob"]])

    print("Total documents evaluated:", len(out)
          , "\nCorrect:", out["correct"].sum()
          , "\nWrong:", (~out["correct"]).sum()
          , "\nAccuracy:", out["correct"].mean())
    print(out[out["correct"]][["path","label","pred","prob"]])

    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred))

    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    labels_sorted = sorted(list(set(y_true)))
    print(labels_sorted)
    print(confusion_matrix(y_true, y_pred, labels=labels_sorted))

if __name__ == "__main__":
    main()