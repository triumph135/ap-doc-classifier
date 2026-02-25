import re
from pathlib import Path

import fitz
from PIL import Image
import pytesseract
import joblib
import pandas as pd


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

LABELS = [
    "Bill of Lading Invoice",
    "Bill of Lading",
    "Purchase Order Receipt",
    "Purchase Order Invoice",
]

SUPPORTED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def ocr_image(img: Image.Image) -> str:
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
        native = page.get_text("text").strip()
        if len(native) > 50:
            parts.append(native.lower())
            continue
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        parts.append(ocr_image(img))
    return "\n".join(parts)


def extract_text(path: str) -> str:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".pdf":
        return extract_text_from_pdf(str(p))
    if suf in SUPPORTED_EXT:
        return ocr_image(Image.open(p).convert("RGB"))
    raise ValueError(f"Unsupported: {suf}")


def clean_text(text: str, max_chars: int = 8000) -> str:
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t[:max_chars]


def main():
    model = joblib.load("doc_type_model.joblib")

    folder = Path(r"data/docs")  # change if needed
    threshold = 0.60             # if max prob < threshold -> Manual Review

    rows = []
    for f in folder.rglob("*"):
        if f.suffix.lower() not in SUPPORTED_EXT:
            continue
        text = clean_text(extract_text(str(f)))

        probs = model.predict_proba([text])[0]
        classes = model.classes_
        best_idx = int(probs.argmax())
        best_label = classes[best_idx]
        best_prob = float(probs[best_idx])

        final = best_label if best_prob >= threshold else "Manual Review"

        rows.append({
            "file": f.name,
            "path": str(f),
            "predicted": best_label,
            "prob": round(best_prob, 4),
            "final": final
        })

        print(f"{f.name} -> {final} (pred={best_label}, p={best_prob:.2f})")

    out = pd.DataFrame(rows).sort_values(["final", "prob"], ascending=[True, False])
    out.to_csv("predictions.csv", index=False)
    print("\nSaved predictions.csv")


if __name__ == "__main__":
    main()