import re
from pathlib import Path

import fitz  # PyMuPDF (pip install pymupdf)
from PIL import Image
import pytesseract

# If Tesseract isn't on PATH, set this:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CLASSES = [
    "Bill of Lading Invoice",
    "Bill of Lading",
    "Purchase Order Receipt",
    "Purchase Order Invoice",
    "Manual Review",
]

DEBUG = False  # set True to print scores/features


def ocr_image(img: Image.Image) -> str:
    # Better OCR for forms/headers than default
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
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(str(p))
    if suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        img = Image.open(p).convert("RGB")
        return ocr_image(img)
    raise ValueError(f"Unsupported file type: {suffix}")


def extract_max_number_near_keywords(t: str, keywords):
    nums = []
    for kw in keywords:
        # number BEFORE keyword: "9603 gallons"
        pattern1 = re.compile(rf"(\d{{1,7}}(?:[,\d]{{0,7}})?(?:\.\d+)?)\s*{re.escape(kw)}\b")
        # keyword BEFORE number: "quantity: 2041.2"
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
    # BOL signals
    # --------------------
    bol_header_terms = [
        "bill of lading", "shipping paper", "bol no", "b.o.l", "b/l", "bill-of-lading"
    ]

    bol_logistics_terms = [
        "carrier", "driver", "truck", "trailer",
        "loading", "unloading", "origin", "destination",
        "gross", "tare", "net weight", "scale ticket",
        "hazmat", "placard", "flammable", "un1075", "un"
    ]

    fuel_terms = [
        "propane", "lpg", "liquefied petroleum gas",
        "diesel", "gasoline", "heating oil", "kerosene", "ulsd",
        "tank", "compartment", "station",
        "fill-up", "fill up", "fillup"
    ]

    # --------------------
    # Invoice signals (strong anchors)
    # --------------------
    strong_invoice_terms = [
        "invoice number", "invoice no", "inv no", "inv #",
        "invoice date", "due date", "terms",
        "amount due", "total due", "balance due",
        "net invoice total", "invoice total"
    ]

    # Boilerplate that often contains the word "invoice" on NON-invoices
    boilerplate_invoice_context = re.search(
        r"(finance charge|unpaid balance|annual percentage rate|apr).{0,80}invoice", t
    ) is not None

    # currency patterns: with or without $
    has_dollar_amount = re.search(r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?", t) is not None
    has_total_money_line = re.search(
        r"\b(total|amount due|total due|balance due|invoice total|net invoice total)\b.{0,40}\$?\s*\d",
        t
    ) is not None

    # Invoice table-ish cues (common on your BOL invoices)
    invoice_table_terms = ["rate", "unit price", "extended", "amount", "subtotal", "tax", "total"]

    # --------------------
    # PO signals (make these STRICT)
    # --------------------
    procurement_terms = [
        "packing slip", "received by", "receiving", "qty received",
        "backorder", "part number", "sku", "serial", "model",
        "item", "items", "parts"
    ]

    # Require PO field with a value (prevents "Purchase Order No." header from triggering)
    has_po_field_with_value = re.search(
        r"\b(purchase\s*order|po)\s*(number|no|#)?\s*[:\-]\s*[a-z0-9\-]{3,}\b", t
    ) is not None

    # --------------------
    # Quantity extraction (fuel BOLs may not say gallons)
    # --------------------
    max_gal = extract_max_number_near_keywords(t, ["gallon", "gallons", "net gallons"])
    max_qty = extract_max_number_near_keywords(t, ["qty", "quantity", "delivered quantity", "delivered"])

    # --------------------
    # Scoring
    # --------------------
    bol_score = 0
    if any_in(bol_header_terms):
        bol_score += 6  # very strong
    if any_in(bol_logistics_terms):
        bol_score += 2
    if any_in(fuel_terms):
        bol_score += 2

    # quantity boosts (gallons strongest; qty only if fuel context exists)
    if max_gal >= 200:
        bol_score += 2
    if max_gal >= 500:
        bol_score += 2

    if max_gal == 0 and max_qty >= 500 and any_in(fuel_terms):
        bol_score += 2
    elif max_gal == 0 and max_qty >= 200 and any_in(fuel_terms):
        bol_score += 1

    invoice_score = 0
    if any_in(strong_invoice_terms):
        invoice_score += 5
    if has_dollar_amount:
        invoice_score += 3
    if has_total_money_line:
        invoice_score += 2
    if any_in(invoice_table_terms):
        invoice_score += 1
    # The word "invoice" alone is weak; only count it if not boilerplate and other invoice evidence exists
    if ("invoice" in t) and not boilerplate_invoice_context and (has_dollar_amount or has_total_money_line):
        invoice_score += 1

    po_score = 0
    if has_po_field_with_value:
        po_score += 4
    if any_in(procurement_terms):
        po_score += 3

    # --------------------
    # Thresholds
    # --------------------
    looks_like_bol = bol_score >= 5
    looks_like_invoice = invoice_score >= 6
    looks_like_po = po_score >= 5

    # --------------------
    # Decisions (BOL-first for your world)
    # --------------------
    # 1) BOL Invoice: needs BOL-ish + invoice-ish
    if looks_like_bol and looks_like_invoice:
        if DEBUG:
            print("DECISION: BOL INVOICE", bol_score, invoice_score, po_score)
        return "Bill of Lading Invoice"

    # 2) Plain BOL: strong BOL, weak invoice
    if looks_like_bol and not looks_like_invoice:
        if DEBUG:
            print("DECISION: BOL", bol_score, invoice_score, po_score)
        return "Bill of Lading"

    # 3) PO Invoice / Receipt (only when PO is truly strong)
    if looks_like_po and looks_like_invoice and not looks_like_bol:
        return "Purchase Order Invoice"
    if looks_like_po and not looks_like_invoice and not looks_like_bol:
        return "Purchase Order Receipt"

    # --------------------
    # Manual Review gate (ambiguous)
    # --------------------
    scores = {"BOL": bol_score, "INVOICE": invoice_score, "PO": po_score}
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    (best_name, best_score), (second_name, second_score) = top[0], top[1]

    if DEBUG:
        print("SCORES:", scores, "BEST:", top[0], "SECOND:", top[1],
              "max_gal:", max_gal, "max_qty:", max_qty,
              "boilerplate_invoice_context:", boilerplate_invoice_context)

    # If everything is weak or too close, don’t guess
    if best_score < 5 or (best_score - second_score) < 2:
        return "Manual Review"

    # If it looks invoice-y but we can't decide BOL vs PO confidently
    if looks_like_invoice and not (looks_like_bol or looks_like_po):
        return "Manual Review"

    return "Manual Review"


def classify(path: str) -> str:
    text = extract_text(path)
    return classify_text(text)


if __name__ == "__main__":
    folder_path = Path(r"C:\Users\sfleming\Documents\Doc Extraction\Example docs\BOL Purchases\EDP")
    supported_ext = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    for file_path in folder_path.iterdir():
        if file_path.suffix.lower() in supported_ext:
            try:
                label = classify(str(file_path))
                print(f"{file_path.name} -> {label}")
            except Exception as e:
                print(f"{file_path.name} -> ERROR: {e}")