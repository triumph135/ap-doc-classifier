import re
import tempfile
from pathlib import Path

import joblib
import streamlit as st

import fitz  # pymupdf
from PIL import Image
import shutil
import os
import pytesseract


# ---- Config ----
MODEL_PATH = "doc_type_model.joblib"
import os
TESSERACT_EXE = os.environ.get("TESSERACT_CMD")
if TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
else:
    # If tesseract is installed via packages.txt it should be on PATH
    if shutil.which("tesseract") is None:
        # No OCR available on this host
        pass

SUPPORTED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---- Text extraction (same idea as your training script) ----
def ocr_image(img: Image.Image) -> str:
    w, h = img.size
    if max(w, h) < 1500:
        img = img.resize((w * 2, h * 2))
    t1 = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
    t2 = pytesseract.image_to_string(img, config="--oem 3 --psm 4")
    return (t1 if len(t1) >= len(t2) else t2).lower()


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
    raise ValueError(f"Unsupported file type: {suf}")


def clean_text(text: str, max_chars: int = 8000) -> str:
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t[:max_chars]


# ---- Streamlit UI ----
st.set_page_config(page_title="Doc Type Classifier", layout="centered")
st.title("Document Type Classifier")

model = joblib.load(MODEL_PATH)

uploaded = st.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"])

threshold = st.slider("Manual review threshold", 0.0, 1.0, 0.60, 0.01)

show_text = st.checkbox("Show extracted text", value=False)

if uploaded:
    suffix = "." + uploaded.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Extracting text..."):
        text = clean_text(extract_text(tmp_path))

    if len(text.strip()) < 40:
        st.error("Could not extract enough text from this file (OCR may have failed).")
    else:
        with st.spinner("Classifying..."):
            prob_vec = model.predict_proba([text])[0]
            classes = model.classes_
            best_idx = int(prob_vec.argmax())
            best_label = classes[best_idx]
            best_prob = float(prob_vec[best_idx])

        final = best_label if best_prob >= threshold else "Manual Review"

        st.subheader("Result")
        st.write(f"**Prediction:** {final}")
        st.write(f"**Model top class:** {best_label}")
        st.write(f"**Confidence:** {best_prob:.2f}")

        if show_text:
            st.subheader("Extracted text (truncated)")
            st.text(text[:4000])