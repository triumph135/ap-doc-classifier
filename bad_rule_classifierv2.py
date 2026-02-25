import re
from pathlib import Path

import fitz  # PyMuPDF (pip install pymupdf)
from PIL import Image
import pytesseract

# If Tesseract isn't on PATH, set:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CLASSES = [
    "Bill of Lading Invoice",
    "Bill of Lading",
    "Purchase Order Receipt",
    "Purchase Order Invoice",
    "Manual Review",
]


def ocr_image(img: Image.Image) -> str:
    # Better defaults than plain OCR (helps headers like Invoice Number / Total Due)
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6").lower()


def extract_text_from_pdf(pdf_path: str, dpi: int = 300, max_pages: int = 3) -> str:
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
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(str(p))
    if suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        img = Image.open(p).convert("RGB")
        return ocr_image(img)
    raise ValueError(f"Unsupported file type: {suffix}")


def extract_max_number_near_keywords(t: str, keywords):
    """
    Finds numbers adjacent to keywords either before or after.
    Examples matched:
      - "2041.2 quantity"
      - "quantity: 2041.2"
      - "9603 gallons"
      - "net gallons 9603"
    """
    nums = []
    for kw in keywords:
        # number BEFORE keyword: "500 gallons"
        pattern1 = re.compile(rf"(\d{{1,7}}(?:[,\d]{{0,7}})?(?:\.\d+)?)\s*{re.escape(kw)}\b")
        # keyword BEFORE number: "qty: 500"
        pattern2 = re.compile(rf"{re.escape(kw)}\s*[:#\-]?\s*(\d{{1,7}}(?:[,\d]{{0,7}})?(?:\.\d+)?)\b")

        for m in pattern1.finditer(t):
            s = m.group(1).replace(",", "")
            try:
                nums.append(float(s))
            except Exception:
                pass

        for m in pattern2.finditer(t):
            s = m.group(1).replace(",", "")
            try:
                nums.append(float(s))
            except Exception:
                pass

    return max(nums) if nums else 0


def classify_text(text: str) -> str:
    t = text.lower()

    def any_in(words):
        return any(w in t for w in words)

    # --------------------
    # Signals
    # --------------------
    bol_header_terms = [
        "bol no", "b.o.l", "b/l", "bill of lading", "bill-of-lading",
        "shipping paper", "straight bill of lading",
    ]

    bol_terms = [
        "carrier", "driver", "truck", "trailer",
        "loading", "unloading", "origin", "destination",
        "gross", "tare", "net weight", "net gallons",
        "scale ticket", "hazmat", "placard", "flammable", "un",
    ]

    fuel_terms = [
        "propane", "lpg", "liquefied petroleum gas",
        "diesel", "gasoline", "heating oil", "kerosene", "ulsd",
        "tank", "compartment",
        "fill-up", "fill up", "fillup", "station",
    ]

    # Invoice: split strong vs weak
    strong_invoice_terms = [
        "invoice number", "inv number", "invoice no", "inv no",
        "invoice date", "amount due", "total due", "balance due",
        "net invoice total", "total invoice",
    ]
    weak_invoice_terms = ["invoice"]  # only a small bump, and suppressed by boilerplate check

    money_terms = ["total", "subtotal", "tax", "rate", "unit price", "extended", "amount", "discount"]

    procurement_terms = [
        "packing slip", "received by", "receiving", "qty received",
        "backorder", "part number", "sku", "serial", "model", "item", "items", "parts",
    ]

    # PO: stricter to avoid header-only triggers
    has_po_field_with_value = re.search(
        r"\b(purchase\s*order|po)\s*(number|no|#)?\s*[:\-]\s*[a-z0-9\-]{3,}\b", t
    ) is not None
    has_purchase_order_phrase = "purchase order" in t

    # Quantity / gallons extraction
    max_gal = extract_max_number_near_keywords(t, ["gallon", "gallons", "net gallons"])
    max_qty = extract_max_number_near_keywords(t, ["qty", "quantity", "total qty", "delivered", "delivery qty"])

    # Invoice evidence: dollar amounts & total/amount due lines
    has_dollar_amount = re.search(r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?", t) is not None
    has_total_line = re.search(r"\b(total|amount due|total due|balance due|net invoice total)\b", t) is not None
    boilerplate_invoice_context = re.search(
        r"(finance charge|unpaid balance|annual percentage rate|apr).*invoice", t
    ) is not None

    # --------------------
    # Scores
    # --------------------
    bol_score = 0
    if any_in(bol_header_terms):
        bol_score += 5
    if any_in(bol_terms):
        bol_score += 2
    if any_in(fuel_terms):
        bol_score += 2

    # Prefer gallons; only use qty if gallons not found
    if max_gal >= 200:
        bol_score += 2
    if max_gal >= 500:
        bol_score += 2

    if max_gal == 0:
        if max_qty >= 500 and any_in(fuel_terms):
            bol_score += 2
        elif max_qty >= 500:
            bol_score += 1
        elif max_qty >= 200 and any_in(fuel_terms):
            bol_score += 1

    invoice_score = 0
    if any(term in t for term in strong_invoice_terms):
        invoice_score += 4
    if has_dollar_amount:
        invoice_score += 3
    if has_total_line:
        invoice_score += 1
    if any(term in t for term in weak_invoice_terms) and not boilerplate_invoice_context:
        invoice_score += 1
    # money words help a little, but can appear elsewhere too
    if any_in(money_terms):
        invoice_score += 1

    po_score = 0
    if has_po_field_with_value:
        po_score += 3
    if has_purchase_order_phrase:
        po_score += 1
    if any_in(procurement_terms):
        po_score += 2

    looks_like_bol = bol_score >= 4
    looks_like_invoice = invoice_score >= 5
    looks_like_po = po_score >= 4

    # --------------------
    # Decisions
    # --------------------
    # If we have strong BOL header + invoice signals, it's a BOL Invoice immediately
    if any_in(bol_header_terms) and looks_like_invoice:
        return "Bill of Lading Invoice"

    if looks_like_bol and looks_like_invoice:
        return "Bill of Lading Invoice"
    if looks_like_bol and not looks_like_po:
        return "Bill of Lading"

    if looks_like_po and looks_like_invoice:
        return "Purchase Order Invoice"
    if looks_like_po and not looks_like_bol:
        return "Purchase Order Receipt"

    # Manual Review gate (weak or ambiguous)
    scores = {"BOL": bol_score, "INVOICE": invoice_score, "PO": po_score}
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    (best_name, best_score), (second_name, second_score) = top[0], top[1]

    # If everything is weak → Manual Review
    if best_score < 4:
        return "Manual Review"

    # If close call → Manual Review
    if (best_score - second_score) < 2:
        return "Manual Review"

    # If conflicting strong signals → Manual Review
    if looks_like_bol and looks_like_po:
        return "Manual Review"
    if looks_like_invoice and not (looks_like_bol or looks_like_po):
        return "Manual Review"

    return "Manual Review"


def classify(path: str) -> str:
    text = extract_text(path)
    return classify_text(text)


if __name__ == "__main__":
    folder_path = Path(r"C:\Users\sfleming\Documents\Doc Extraction\Example docs\BOL Purchases\EDP")
    supported_ext = [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

    for file_path in folder_path.iterdir():
        if file_path.suffix.lower() in supported_ext:
            try:
                label = classify(str(file_path))
                print(f"{file_path.name} -> {label}")
            except Exception as e:
                print(f"{file_path.name} -> ERROR: {e}")