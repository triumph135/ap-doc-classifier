"""Central settings for the AP document classifier.

Edit this file to change class names, label aliases, thresholds or paths.
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The document types the model is meant to recognise.
CLASSES = ["BOL Invoice", "BOL Purchase", "PO Receipt", "PO Invoice"]

# How each class is filed in Cargas Energy. BOL purchases and PO receipts are
# both "Material Receipt" documents there, but the model tells them apart.
CARGAS_DOCUMENT_TYPE = {
    "BOL Invoice": "BOL Invoice",
    "BOL Purchase": "Material Receipt",
    "PO Receipt": "Material Receipt",
    "PO Invoice": "PO Invoice",
}

# Accepted spellings in labels.csv (lowercase) mapped to the official class.
# "Material Receipt" is deliberately NOT listed: it is ambiguous, so each
# document must be labelled BOL Purchase or PO Receipt.
LABEL_ALIASES = {name.lower(): name for name in CLASSES}
LABEL_ALIASES.update({
    "bill of lading invoice": "BOL Invoice",
    "fuel invoice": "BOL Invoice",
    "bill of lading": "BOL Purchase",
    "bol": "BOL Purchase",
    "purchase order receipt": "PO Receipt",
    "purchase order invoice": "PO Invoice",
})

# Training rules
MIN_EXAMPLES_PER_CLASS = 20   # a class with fewer labelled examples is not trained
CV_FOLDS = 5
RANDOM_SEED = 42

# Prediction rules
REVIEW_THRESHOLD = 0.90       # below this confidence a document goes to manual review
MIN_TEXT_CHARS = 40           # less readable text than this goes to manual review

# Text extraction
MAX_PAGES = 2
OCR_DPI = 300
MAX_CHARS = 8000

# Paths (paths inside labels.csv are relative to ROOT)
DATA_DIR = Path(os.environ.get("DOCCLF_DATA_DIR", ROOT / "data"))
LABELS_CSV = DATA_DIR / "labels.csv"
CACHE_DIR = Path(os.environ.get("DOCCLF_CACHE_DIR", DATA_DIR / "text_cache"))
MODELS_DIR = Path(os.environ.get("DOCCLF_MODELS_DIR", ROOT / "models"))
LATEST_MODEL = MODELS_DIR / "latest.joblib"
LATEST_META = MODELS_DIR / "latest.json"


def configure_tesseract():
    """Point pytesseract at Tesseract: TESSERACT_CMD, then PATH, then the Windows default."""
    import pytesseract

    cmd = os.environ.get("TESSERACT_CMD")
    if not cmd and shutil.which("tesseract") is None:
        default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if default.exists():
            cmd = str(default)
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
