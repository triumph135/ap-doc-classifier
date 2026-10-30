"""Streamlit app: upload a document and see its predicted type.

    streamlit run classifier_app.py
"""
import os
import tempfile
from pathlib import Path

import streamlit as st

from docclf import config
from docclf.classify import MANUAL_REVIEW, Classifier

# Typical training documents have about 85 to 90 percent of their words in the model vocabulary.
# Below this share the app explains that the document looks unfamiliar (a rough guide, not a rule).
FAMILIARITY_WARNING = 0.70

st.set_page_config(page_title="Doc Type Classifier", layout="centered")
st.title("Document Type Classifier")


@st.cache_resource
def load_classifier():
    return Classifier()


try:
    clf = load_classifier()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

for w in clf.warnings:
    st.warning(w)
version = clf.meta.get("version", "unknown")
st.caption(f"Model version {version}. Knows: {', '.join(clf.classes)}.")
if clf.missing_classes:
    st.warning(f"This model has no training examples for: {', '.join(clf.missing_classes)}. "
               "Those documents will be sent to another class or to manual review.")

uploaded = st.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"])
threshold = st.slider("Manual review threshold", 0.0, 1.0, config.REVIEW_THRESHOLD, 0.01)

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    try:
        with st.spinner("Reading and classifying..."):
            result = clf.predict_file(tmp_path, threshold, keep_text=True)  # not cached on disk
    finally:
        os.unlink(tmp_path)

    st.subheader("Result")
    if result["final"] == MANUAL_REVIEW:
        st.warning(f"Manual review. {result['reason']}")
    else:
        st.success(f"{result['final']}")
        st.write(f"**Cargas Energy document type:** {result['cargas_document_type']}")
    st.write(f"**Top class:** {result['predicted']}  ({result['confidence']:.2f})")
    if result["runner_up"]:
        st.write(f"**Runner up:** {result['runner_up']}  ({result['runner_up_confidence']:.2f})")
    st.caption(f"Text source: {result['text_source']}")

    if result["predicted"] is not None:
        with st.expander("Why this result?"):
            info = clf.explain_text(result["text"])
            st.write(f"**Familiarity:** {info['known_words']} of {info['total_words']} distinct words "
                     f"({info['familiarity']:.0%}) appear in the model's training vocabulary.")
            if info["familiarity"] < FAMILIARITY_WARNING:
                st.info("This document uses wording the model rarely saw in training, which usually means a "
                        "new vendor or format. Lower confidence is expected here. Adding labeled examples of "
                        "this kind of document will help.")
            left, right = st.columns(2)
            left.markdown(f"**Pushing toward {info['top_class']}**")
            left.write(", ".join(info["words_for_top"]) or "none")
            right.markdown(f"**Pushing toward {info['runner_up']}**")
            right.write(", ".join(info["words_for_runner_up"]) or "none")
            if st.checkbox("Show extracted text"):
                st.text(result["text"][:4000])
