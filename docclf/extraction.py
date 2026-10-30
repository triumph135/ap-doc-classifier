"""Text extraction (native PDF text, OCR fallback) with an on disk cache.

Training and prediction both use this module, so a document is always turned
into text the same way. Bump EXTRACTION_VERSION whenever extraction changes so
old cached text is ignored.
"""
import hashlib
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF
    import fitz
from PIL import Image
import pytesseract

from . import config

EXTRACTION_VERSION = 1
SUPPORTED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

config.configure_tesseract()


def normalize_text(text):
    """Used by the TF IDF vectorizers: lowercase and collapse digit runs so
    document numbers and dates do not become noise features."""
    return re.sub(r"\d+", " num ", text.lower())


def clean_text(text, max_chars=None):
    max_chars = max_chars or config.MAX_CHARS
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t[:max_chars]


def file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ocr_image(img):
    w, h = img.size
    if max(w, h) < 1500:  # upscale small images, it helps OCR
        img = img.resize((w * 2, h * 2))
    t1 = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
    t2 = pytesseract.image_to_string(img, config="--oem 3 --psm 4")
    return (t1 if len(t1) >= len(t2) else t2).lower()


def _extract_pdf(path, mode):
    doc = fitz.open(path)
    parts, sources = [], []
    try:
        for i in range(min(len(doc), config.MAX_PAGES)):
            page = doc[i]
            native = page.get_text("text").strip()
            if mode == "auto" and len(native) > 50:
                parts.append(native.lower())
                sources.append("native")
                continue
            pix = page.get_pixmap(dpi=config.OCR_DPI)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            parts.append(ocr_image(img))
            sources.append("ocr")
    finally:
        doc.close()
    if not sources:
        return "", "empty"
    kinds = set(sources)
    return "\n".join(parts), (kinds.pop() if len(kinds) == 1 else "mixed")


def extract_text(path, mode="auto"):
    """Return (cleaned text, source). mode "auto" uses native PDF text when
    present and OCR otherwise; mode "ocr" forces OCR on every page."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text, source = _extract_pdf(str(path), mode)
    elif suffix in SUPPORTED_EXT:
        text, source = ocr_image(Image.open(path).convert("RGB")), "ocr"
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return clean_text(text), source


def _cache_file(sha, mode):
    return config.CACHE_DIR / f"{sha}_{mode}_v{EXTRACTION_VERSION}.json"


def _write_cache(cache_file, result):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    tmp.replace(cache_file)


def _read_cache(cache_file):
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # missing or damaged entry


def get_text(path, mode="auto", use_cache=True, sha=None):
    """Extract text for one file, reading and writing the cache when enabled.
    Never raises: failures come back as source "error"."""
    path = Path(path)
    cache_file = None
    if use_cache:
        sha = sha or file_hash(path)
        cache_file = _cache_file(sha, mode)
        cached = _read_cache(cache_file) if cache_file.exists() else None
        if cached is not None:
            return cached
        if mode == "ocr":
            # A document whose pages were all OCR'd in auto mode has identical text in ocr mode.
            auto_cached = _read_cache(_cache_file(sha, "auto"))
            if auto_cached is not None and auto_cached.get("source") == "ocr":
                _write_cache(cache_file, auto_cached)
                return auto_cached
    try:
        text, source = extract_text(path, mode)
    except Exception as exc:  # keep one bad file from stopping a whole run
        return {"text": "", "source": "error", "error": str(exc)}
    result = {"text": text, "source": source, "error": None}
    if cache_file is not None:
        _write_cache(cache_file, result)
    return result


def _worker(job):
    path, mode, sha = job
    return get_text(path, mode, True, sha)


def get_texts(paths, shas, mode="auto", workers=4, progress_every=25):
    """Extract many files, in parallel when workers > 1. Results are cached one
    file at a time, so an interrupted run resumes where it stopped."""
    jobs = [(str(p), mode, s) for p, s in zip(paths, shas)]
    total, started, results = len(jobs), time.time(), []

    def progress(done):
        if done % progress_every == 0 or done == total:
            print(f"  extracted {done}/{total} documents ({time.time() - started:.0f}s)", flush=True)

    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            results.append(_worker(job))
            progress(i)
    else:
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # one Tesseract thread per worker
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, res in enumerate(pool.map(_worker, jobs, chunksize=1), 1):
                results.append(res)
                progress(i)
    return results
