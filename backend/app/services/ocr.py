from __future__ import annotations

"""OCR for scanned / image-only PDFs.

Backends (tried in order):
1. PyMuPDF TextPage OCR (requires Tesseract on PATH)
2. pytesseract (requires Tesseract on PATH)
3. easyocr (optional heavy install)

If all fail, returns empty strings — pipeline may fall back to VLM page-read.
"""

import hashlib
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger("avrseek.ocr")

# Below this many chars of native PDF text → treat as needing OCR
OCR_CHAR_THRESHOLD = 400
OCR_DPI = 200


@dataclass
class OcrPageResult:
    page_number: int
    text: str
    backend: str


@dataclass
class OcrDocumentResult:
    pages: list[OcrPageResult]
    backend: str
    total_chars: int

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


def needs_ocr(native_text: str, page_count: int) -> bool:
    chars = len((native_text or "").strip())
    if page_count <= 0:
        return False
    if chars == 0:
        return True
    return chars < max(OCR_CHAR_THRESHOLD, page_count * 80)


def _cache_path(output_dir: Path, pdf_path: Path) -> Path:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    cache_dir = output_dir / "ocr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{digest.hexdigest()}.json"


def load_ocr_cache(output_dir: Path, pdf_path: Path) -> OcrDocumentResult | None:
    path = _cache_path(output_dir, pdf_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pages = [
            OcrPageResult(page_number=p["page_number"], text=p["text"], backend=p.get("backend", "cache"))
            for p in data.get("pages", [])
        ]
        return OcrDocumentResult(
            pages=pages,
            backend=data.get("backend", "cache"),
            total_chars=int(data.get("total_chars", 0)),
        )
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        return None


def save_ocr_cache(output_dir: Path, pdf_path: Path, result: OcrDocumentResult) -> None:
    path = _cache_path(output_dir, pdf_path)
    payload = {
        "backend": result.backend,
        "total_chars": result.total_chars,
        "pages": [
            {"page_number": p.page_number, "text": p.text, "backend": p.backend}
            for p in result.pages
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def render_page_image(page: fitz.Page, dpi: int = OCR_DPI) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


_TESSERACT_CANDIDATES = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)


def tesseract_executable() -> Path | None:
    which = shutil.which("tesseract")
    if which:
        return Path(which)
    for candidate in _TESSERACT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def tessdata_dir() -> Path:
    """Project tessdata folder (*.traineddata files live here)."""
    return Path(__file__).resolve().parents[2] / "tessdata"


def configure_tesseract() -> Path | None:
    """Point pytesseract / this process at the Tesseract binary if it is not on PATH."""
    exe = tesseract_executable()
    if exe is None:
        return None
    tess_dir = str(exe.parent)
    os.environ["PATH"] = tess_dir + os.pathsep + os.environ.get("PATH", "")
    local_tessdata = tessdata_dir()
    if local_tessdata.is_dir() and any(local_tessdata.glob("*.traineddata")):
        # TESSDATA_PREFIX is the parent of the tessdata directory
        os.environ["TESSDATA_PREFIX"] = str(local_tessdata.parent)
    else:
        if (exe.parent / "tessdata").is_dir():
            os.environ.setdefault("TESSDATA_PREFIX", str(exe.parent))
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = str(exe)
    except ImportError:
        pass
    return exe


configure_tesseract()


def _tesseract_lang(languages: str) -> str:
    # Active now: eng, rus, chi_sim, chi_tra. Later: fas, spa, heb
    mapping = {
        "eng": "eng",
        "en": "eng",
        "rus": "rus",
        "ru": "rus",
        "chi_sim": "chi_sim",
        "chi_tra": "chi_tra",
        "zh": "chi_sim",
        "zh_cn": "chi_sim",
        "zh_tw": "chi_tra",
        "chinese": "chi_sim",
        "fas": "fas",
        "fa": "fas",
        "farsi": "fas",
        "persian": "fas",
        "spa": "spa",
        "es": "spa",
        "spanish": "spa",
        "heb": "heb",
        "he": "heb",
        "hebrew": "heb",
    }
    parts = []
    for token in re.split(r"[+,;\s]+", languages.lower()):
        if not token:
            continue
        code = mapping.get(token, token)
        if code not in parts:
            parts.append(code)
    return "+".join(parts) or "eng"


def _tessdata_config() -> str:
    local = tessdata_dir()
    if local.is_dir() and any(local.glob("*.traineddata")):
        return f'--tessdata-dir "{local}"'
    return ""


def _ocr_page_pymupdf(page: fitz.Page, languages: str = "eng") -> str | None:
    try:
        tp = page.get_textpage_ocr(dpi=OCR_DPI, language=_tesseract_lang(languages), full=True)
        text = page.get_text("text", textpage=tp) or ""
        return text.strip() or None
    except Exception as exc:
        logger.debug("PyMuPDF OCR failed on page: %s", exc)
        return None


def _ocr_image_tesseract(image: Image.Image, languages: str = "eng") -> str | None:
    try:
        import pytesseract

        configure_tesseract()
        cfg = _tessdata_config()
        kwargs: dict = {"lang": _tesseract_lang(languages)}
        if cfg:
            kwargs["config"] = cfg
        text = pytesseract.image_to_string(image, **kwargs) or ""
        return text.strip() or None
    except Exception as exc:
        logger.debug("pytesseract OCR failed: %s", exc)
        return None


_easyocr_reader = None


def _ocr_image_easyocr(image: Image.Image) -> str | None:
    global _easyocr_reader
    try:
        import easyocr
        import numpy as np

        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(["en", "ru"], gpu=False)
        arr = np.array(image.convert("RGB"))
        lines = _easyocr_reader.readtext(arr, detail=0, paragraph=True)
        text = "\n".join(str(x) for x in lines)
        return text.strip() or None
    except Exception as exc:
        logger.debug("easyocr failed: %s", exc)
        return None


def ocr_pdf(
    pdf_path: Path,
    *,
    output_dir: Path | None = None,
    languages: str = "eng",
) -> OcrDocumentResult:
    """OCR every page; prefer cache when available."""
    if output_dir is not None:
        cached = load_ocr_cache(output_dir, pdf_path)
        if cached and cached.total_chars > 0:
            logger.info("OCR cache hit for %s (%s chars)", pdf_path.name, cached.total_chars)
            return cached

    doc = fitz.open(pdf_path)
    pages: list[OcrPageResult] = []
    backend_used = "none"

    for i, page in enumerate(doc, start=1):
        text = ""
        backend = "none"

        # 1) PyMuPDF OCR
        pm = _ocr_page_pymupdf(page, languages)
        if pm:
            text, backend = pm, "pymupdf_ocr"
        else:
            image = render_page_image(page)
            # 2) Tesseract
            te = _ocr_image_tesseract(image, languages)
            if te:
                text, backend = te, "pytesseract"
            else:
                # 3) EasyOCR
                eo = _ocr_image_easyocr(image)
                if eo:
                    text, backend = eo, "easyocr"

        if backend != "none":
            backend_used = backend
        pages.append(OcrPageResult(page_number=i, text=text, backend=backend))

    result = OcrDocumentResult(
        pages=pages,
        backend=backend_used,
        total_chars=sum(len(p.text) for p in pages),
    )
    if output_dir is not None and result.total_chars > 0:
        save_ocr_cache(output_dir, pdf_path, result)
    return result


def available_ocr_languages() -> list[str]:
    configure_tesseract()
    local = tessdata_dir()
    if local.is_dir():
        langs = sorted(p.stem for p in local.glob("*.traineddata") if p.stem != "osd")
        if langs:
            return langs
    try:
        import pytesseract

        return [lang for lang in pytesseract.get_languages(config="") if lang != "osd"]
    except Exception:
        return []


def available_ocr_backends() -> list[str]:
    found: list[str] = []
    exe = configure_tesseract()
    if exe is not None:
        found.append(f"tesseract ({exe})")
        found.append("pymupdf_ocr")
    else:
        found.append("pymupdf_ocr (needs Tesseract)")
    try:
        import pytesseract  # noqa: F401

        found.append("pytesseract" if exe else "pytesseract (Tesseract binary missing)")
    except ImportError:
        pass
    try:
        import easyocr  # noqa: F401

        found.append("easyocr")
    except ImportError:
        pass
    langs = available_ocr_languages()
    if langs:
        found.append("langs:" + "+".join(langs))
    return found
