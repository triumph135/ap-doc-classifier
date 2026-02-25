import re
from pathlib import Path

import fitz
import pandas as pd
from PIL import Image
import pytesseract

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# If needed:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

LABELS = [
    "Bill of Lading Invoice",
    "Bill of Lading",
    "Purchase Order Receipt",
    "Purchase Order Invoice",
]


def ocr_image(img: Image.Image) -> str:
    # Upscale small images a bit (helps OCR)
    w, h = img.size
    if max(w, h) < 1500:
        img = img.resize((w * 2, h * 2))
    t1 = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
    t2 = pytesseract.image_to_string(img, config="--oem 3 --psm 4")
    t = t1 if len(t1) >= len(t2) else t2
    return t.lower()


def extract_text_from_pdf(pdf_path: str, dpi: int = 300, max_pages: int = 2) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for i in range(min(len(doc), max_pages)):
        page = doc[i]

        # Prefer native text if present
        native = page.get_text("text").strip()
        if len(native) > 50:
            parts.append(native.lower())
            continue

        # Otherwise OCR
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        parts.append(ocr_image(img))

    return "\n".join(parts)


def extract_text(path: str) -> str:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".pdf":
        return extract_text_from_pdf(str(p))
    if suf in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        return ocr_image(Image.open(p).convert("RGB"))
    raise ValueError(f"Unsupported: {suf}")


def clean_text(text: str, max_chars: int = 8000) -> str:
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t[:max_chars]


def main():

    print("TRAIN START")
    
    df = pd.read_csv("data/labels.csv")

    # Basic validation
    df["label"] = df["label"].astype(str).str.strip()
    bad = df[~df["label"].isin(LABELS)]
    if len(bad) > 0:
        raise ValueError(f"Found invalid labels in labels.csv:\n{bad}")

    texts = []
    for path in df["path"]:
        text = clean_text(extract_text(path))
        texts.append(text)

    X = texts
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            strip_accents="unicode"
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n=== Classification report ===")
    print(classification_report(y_test, preds))

    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    labels_sorted = sorted(list(set(y)))
    print(labels_sorted)
    print(confusion_matrix(y_test, preds, labels=labels_sorted))

    joblib.dump(model, "doc_type_model.joblib")
    print("\nSaved model to doc_type_model.joblib")


if __name__ == "__main__":
    main()