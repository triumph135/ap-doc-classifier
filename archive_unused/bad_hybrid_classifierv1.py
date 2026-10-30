import re
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from transformers import pipeline


# --- OCR setup ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


LABELS = [
    "Bill of Lading Invoice",
    "Bill of Lading",
    "Purchase Order Receipt",
    "Purchase Order Invoice",
]

SUPPORTED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# --- Zero-shot model (no training data needed) ---
# Accurate but slower on CPU. We’ll only call it when rules are uncertain.
zsc = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1,
)


def ocr_image(img: Image.Image) -> str:
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6").lower()


def extract_text_from_pdf(pdf_path: str, dpi: int = 300, max_pages: int = 2) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    for i in range(min(len(doc), max_pages)):
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        texts.append(ocr_image(img))
    return "\n".join(texts)


def extract_text(path: str) -> str:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".pdf":
        return extract_text_from_pdf(str(p))
    if suf in SUPPORTED_EXT:
        img = Image.open(p).convert("RGB")
        return ocr_image(img)
    raise ValueError(f"Unsupported file type: {suf}")


def clean_for_zsc(text: str, max_chars: int = 3500) -> str:
    # Helps MNLI models by reducing OCR noise and keeping the most informative part.
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t[:max_chars]


def rule_scores(t: str) -> dict:
    """Your original 'first iteration' signals, but returns scores for confidence/margin."""
    def any_in(words):
        return any(w in t for w in words)

    bol_terms = [
        "bill of lading", "shipping paper", "b/l", "bol", "carrier", "driver", "truck", "trailer",
        "loading", "unloading", "origin", "destination", "gross", "tare", "net weight", "net gallons",
        "un1075", "hazmat", "flammable", "placard",
        "bol no",  # added small improvement without changing the spirit of v1
    ]
    fuel_terms = [
        "propane", "lpg", "liquefied petroleum gas", "diesel", "gasoline", "heating oil", "kerosene",
        "ulsd", "gallons", "tank", "compartment", "fill-up", "fill up", "station"
    ]
    invoice_terms = ["invoice number", "invoice date", "amount due", "balance due", "due date", "terms", "invoice"]
    money_terms = ["total", "subtotal", "tax", "rate", "unit price", "extended", "amount"]

    procurement_terms = [
        "purchase order", "packing slip", "received by", "receiving", "qty received",
        "part", "part number", "sku", "backorder", "item", "items"
    ]

    # PO signals (v1-ish)
    has_po_phrase = "purchase order" in t
    has_po_number = re.search(r"\bpo\s*(number|no|#)?\s*[:\-]?\s*[a-z0-9\-]+\b", t) is not None

    # Gallons extraction (v1)
    gallons = []
    for m in re.finditer(r"(\d{2,6}(?:\.\d+)?)\s*(?:net\s*)?gallons?\b", t):
        try:
            gallons.append(float(m.group(1)))
        except Exception:
            pass
    max_gal = max(gallons) if gallons else 0

    bol_score = 0
    if any_in(bol_terms):
        bol_score += 3
    if any_in(fuel_terms):
        bol_score += 2
    if max_gal >= 200:
        bol_score += 2
    if max_gal >= 500:
        bol_score += 2

    invoice_score = 0
    if any_in(invoice_terms):
        invoice_score += 3
    if any_in(money_terms):
        invoice_score += 2

    po_score = 0
    if has_po_phrase:
        po_score += 3
    if has_po_number and any_in(procurement_terms):
        po_score += 2
    if any_in(procurement_terms):
        po_score += 1

    return {
        "bol": bol_score,
        "invoice": invoice_score,
        "po": po_score,
        "max_gal": max_gal,
    }


def rule_decision(scores: dict) -> str:
    """Same decision logic as v1 (with Manual Review option)."""
    bol_score = scores["bol"]
    invoice_score = scores["invoice"]
    po_score = scores["po"]

    looks_like_bol = bol_score >= 4
    looks_like_invoice = invoice_score >= 4
    looks_like_po = po_score >= 4

    if looks_like_bol and looks_like_invoice:
        return "Bill of Lading Invoice"
    if looks_like_bol:
        return "Bill of Lading"

    if looks_like_po and looks_like_invoice:
        return "Purchase Order Invoice"
    if looks_like_po:
        return "Purchase Order Receipt"

    # if it smells like invoice but no strong BOL/PO, we don’t guess in hybrid mode
    return "Manual Review"


def rule_confidence(scores: dict) -> tuple[float, float, str]:
    """
    Returns (confidence, margin, top_bucket) based on the 3 rule scores.
    confidence is a rough 0..1 heuristic; margin is top-second.
    """
    buckets = {
        "bol": scores["bol"],
        "invoice": scores["invoice"],
        "po": scores["po"],
    }
    ranked = sorted(buckets.items(), key=lambda x: x[1], reverse=True)
    (top_name, top_score), (_, second_score) = ranked[0], ranked[1]
    margin = top_score - second_score

    # heuristic confidence: higher score + higher margin => more confident
    # clamp to 0..1
    conf = min(1.0, max(0.0, (top_score / 8.0) * 0.6 + (margin / 6.0) * 0.4))
    return conf, margin, top_name


def zsc_classify(text: str, threshold: float = 0.55) -> tuple[str, float]:
    cleaned = clean_for_zsc(text)
    result = zsc(cleaned, LABELS, multi_label=False)
    label = result["labels"][0]
    score = float(result["scores"][0])
    if score < threshold:
        return "Manual Review", score
    return label, score


def classify_hybrid(path: str,
                    rule_conf_threshold: float = 0.62,
                    rule_margin_threshold: float = 2.0,
                    zsc_threshold: float = 0.50) -> tuple[str, dict]:
    """
    Returns (label, debug_info).
    - If rules are confident => accept
    - Else => zero-shot
    - Else => Manual Review
    """
    text = extract_text(path)

    scores = rule_scores(text)
    rule_label = rule_decision(scores)
    conf, margin, top_bucket = rule_confidence(scores)

    debug = {
        "rule_label": rule_label,
        "rule_conf": round(conf, 3),
        "rule_margin": margin,
        "scores": {k: scores[k] for k in ("bol", "invoice", "po")},
        "max_gal": scores["max_gal"],
        "used": "rules",
    }

    # If rules already say manual review, skip straight to zero-shot
    if rule_label == "Manual Review":
        z_label, z_score = zsc_classify(text, threshold=zsc_threshold)
        debug.update({"used": "zsc", "zsc_label": z_label, "zsc_score": round(z_score, 3)})
        return z_label, debug

    # If rules are confident, trust them
    if conf >= rule_conf_threshold and margin >= rule_margin_threshold:
        return rule_label, debug

    # Otherwise, ask zero-shot to decide
    z_label, z_score = zsc_classify(text, threshold=zsc_threshold)
    debug.update({"used": "zsc", "zsc_label": z_label, "zsc_score": round(z_score, 3)})
    return z_label, debug


if __name__ == "__main__":
    folder_path = Path(r"C:\Users\sfleming\Documents\Doc Extraction\Example docs\BOL Purchases\EDP")

    for file_path in folder_path.iterdir():
        if file_path.suffix.lower() in SUPPORTED_EXT:
            try:
                label, info = classify_hybrid(str(file_path))
                print(f"{file_path.name} -> {label}   [{info['used']}] "
                      f"(rules={info['scores']} conf={info['rule_conf']} margin={info['rule_margin']}"
                      + (f" zsc={info.get('zsc_score')}" if info["used"] == "zsc" else "")
                      + ")")
            except Exception as e:
                print(f"{file_path.name} -> ERROR: {e}")