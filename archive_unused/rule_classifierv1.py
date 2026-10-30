import re
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# If Tesseract isn't on PATH, uncomment and set:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CLASSES = [
    "Bill of Lading Invoice",
    "Bill of Lading",
    "Purchase Order Receipt",
    "Purchase Order Invoice",
]


def ocr_image(img: Image.Image) -> str:
    text = pytesseract.image_to_string(img)
    return text.lower()


def extract_text_from_pdf(pdf_path: str, dpi: int = 200, max_pages: int = 3) -> str:
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
    elif suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        img = Image.open(p).convert("RGB")
        return ocr_image(img)
    else:
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
            except:
                pass

        for m in pattern2.finditer(t):
            s = m.group(1).replace(",", "")
            try:
                nums.append(float(s))
            except:
                pass

    return max(nums) if nums else 0


def classify_text(text: str) -> str:
    t = text.lower()

    def any_in(words):
        return any(w in t for w in words)

    # --- Signals ---
    # Strong "this is a BOL" headers/fields
    bol_header_terms = [
        "bol no", "b.o.l", "b/l", "bill of lading", "bill-of-lading",
        "shipping paper"
    ]

    # Broader BOL logistics cues (often present on true BOL forms)
    bol_terms = [
        "carrier", "driver", "truck", "trailer",
        "loading", "unloading", "origin", "destination",
        "gross", "tare", "net weight", "net gallons",
        "scale ticket", "hazmat", "placard", "flammable", "un"
    ]

    # Fuel cues (your big differentiator)
    fuel_terms = [
        "propane", "lpg", "liquefied petroleum gas",
        "diesel", "gasoline", "heating oil", "kerosene", "ulsd",
        "tank", "compartment",
        "fill-up", "fill up", "fillup", "station"
    ]

    invoice_terms = ["invoice", "invoice number", "inv #", "amount due", "balance due", "terms", "due date"]
    money_terms = ["total", "subtotal", "tax", "rate", "unit price", "extended", "amount"]

    # Procurement / parts cues (helps distinguish PO world)
    procurement_terms = [
        "packing slip", "received by", "receiving", "qty received",
        "backorder", "part number", "sku", "serial", "model", "item", "items"
    ]

    # PO: make it stricter (avoid being triggered by a column header "Purchase Order No.")
    # Require an explicit PO field with a value (e.g., "PO No: 100685", "Purchase Order #: ABC123")
    has_po_field_with_value = re.search(
        r"\b(purchase\s*order|po)\s*(number|no|#)?\s*[:\-]\s*[a-z0-9\-]{3,}\b", t
    ) is not None

    # Still allow the full phrase as a weaker signal (but don't let it override BOL)
    has_purchase_order_phrase = "purchase order" in t

    # Quantity / gallons extraction
    max_gal = extract_max_number_near_keywords(t, ["gallon", "gallons", "net gallons"])
    max_qty = extract_max_number_near_keywords(t, ["qty", "quantity", "total qty", "delivered", "delivery qty"])

    # --- Scores ---
    bol_score = 0

    # Big boost if we see BOL header fields (this fixes your "BOL Invoice w/o gallons" case)
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
        # qty is weaker unless fuel cues exist
        if max_qty >= 500 and any_in(fuel_terms):
            bol_score += 2
        elif max_qty >= 500:
            bol_score += 1
        elif max_qty >= 200 and any_in(fuel_terms):
            bol_score += 1

    invoice_score = 0
    if any_in(invoice_terms):
        invoice_score += 3
    if any_in(money_terms):
        invoice_score += 2

    po_score = 0
    if has_po_field_with_value:
        po_score += 3
    if has_purchase_order_phrase:
        po_score += 1
    if any_in(procurement_terms):
        po_score += 2

        # --- Decisions (BOL-first bias) ---
    looks_like_bol = bol_score >= 4
    looks_like_invoice = invoice_score >= 4
    looks_like_po = po_score >= 4

    # Strong matches
    if looks_like_bol and looks_like_invoice:
        return "Bill of Lading Invoice"
    if looks_like_bol and not looks_like_po:
        return "Bill of Lading"

    if looks_like_po and looks_like_invoice:
        return "Purchase Order Invoice"
    if looks_like_po and not looks_like_bol:
        return "Purchase Order Receipt"

    # --- Ambiguous / weak cases → UNKNOWN ---
    scores = {
        "BOL": bol_score,
        "Invoice": invoice_score,
        "PO": po_score
    }

    max_score = max(scores.values())

    # If everything is weak → Manual Review
    if max_score < 3:
        return "Manual Review"

    # If signals conflict strongly → Manual Review
    if looks_like_bol and looks_like_po:
        return "Manual Review"

    if looks_like_invoice and not (looks_like_bol or looks_like_po):
        return "Manual Review"

    return "Manual Review"


def classify(path: str) -> str:
    text = extract_text(path)
    return classify_text(text)


if __name__ == "__main__":

    folder_path = Path(r"C:\Users\sfleming\Documents\Doc Extraction\Example docs\BOL Purchases\EDP\data\docs")

    supported_ext = [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

    for file_path in folder_path.iterdir():

        if file_path.suffix.lower() in supported_ext:
            try:
                label = classify(str(file_path))
                print(f"{file_path.name} -> {label}")
            except Exception as e:
                print(f"{file_path.name} -> ERROR: {e}")