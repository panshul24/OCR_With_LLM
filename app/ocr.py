"""OCR and PDF rendering utilities.

This module provides:
- Tesseract-based OCR for images and rendered PDF pages
- Optional OpenCV preprocessing (denoise, binarize, deskew)
- Born-digital PDF text extraction using pymupdf4llm (markdown)
"""
from __future__ import annotations
from typing import Tuple, Dict, Any, List
import io
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import os
import numpy as np
import cv2
import pymupdf4llm

def _ocr_image(image: Image.Image) -> str:
    """Run Tesseract with optional env-driven tuning.
    TESSERACT_LANG, TESSERACT_PSM, TESSERACT_OEM, TESSERACT_CHAR_WHITELIST
    """
    lang = os.getenv('TESSERACT_LANG', 'eng')
    psm = os.getenv('TESSERACT_PSM')
    oem = os.getenv('TESSERACT_OEM')
    whitelist = os.getenv('TESSERACT_CHAR_WHITELIST')
    cfg_parts: list[str] = []
    if psm:
        cfg_parts.append(f"--psm {psm}")
    if oem:
        cfg_parts.append(f"--oem {oem}")
    if whitelist:
        cfg_parts.append(f"-c tessedit_char_whitelist={whitelist}")
    config = ' '.join(cfg_parts) if cfg_parts else None
    return pytesseract.image_to_string(image, lang=lang, config=config)


def _preprocess_image(image: Image.Image) -> Image.Image:
    """Optional pre-processing: binarize (Otsu/adaptive), deskew, denoise.
    Controlled via env flags: PREPROCESS_ENABLE, PREPROCESS_BINARIZE, PREPROCESS_DESKEW, PREPROCESS_DENOISE.
    """
    # Convert PIL → OpenCV BGR
    cv = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)

    if os.getenv('PREPROCESS_DENOISE', '0') == '1':
        # Gentle denoise (bilateral preserves edges)
        cv = cv2.bilateralFilter(cv, 7, 50, 50)

    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)

    if os.getenv('PREPROCESS_BINARIZE', '0') == '1':
        # Adaptive threshold for variable lighting
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    else:
        # Otsu as default binarization
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if os.getenv('PREPROCESS_DESKEW', '0') == '1':
        # Estimate skew angle via minAreaRect over text pixels
        coords = np.column_stack(np.where(gray < 255))
        if coords.size > 0:
            rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(gray, mode='L')

def _pdf_to_images(pdf_bytes: bytes) -> list[Image.Image]:
    """Render each PDF page at 300 DPI to PIL Image instances."""
    images = []
    with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(alpha=False, dpi=300)
            img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            images.append(img)
    return images


def _pdf_text_if_born_digital(pdf_bytes: bytes, min_chars: int = 50) -> tuple[str, bool, int]:
    """Extract text using pymupdf4llm markdown extraction for better layout.
    Returns (text, is_born_digital, page_count).
    """
    try:
        with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
            page_count = len(doc)
            md = pymupdf4llm.to_markdown(doc)
            # Convert markdown to plain-ish text for the LLM input if desired
            extracted = md.strip()
            is_born = len(extracted.replace('\n', '').strip()) >= min_chars
            return extracted, is_born, page_count
    except Exception:
        return '', False, 0

def extract_text_from_bytes(data: bytes, filename: str | None = None) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    try:
        if filename and filename.lower().endswith('.pdf'):
            # Try born-digital fast path first
            if os.getenv('PDF_BORNDIGITAL_CHECK', '1') == '1':
                text_digital, is_born, page_count = _pdf_text_if_born_digital(data, min_chars=int(os.getenv('PDF_MIN_CHARS', '50')))
                if is_born:
                    meta['pages'] = page_count
                    meta['source'] = 'pdf-text'
                    return text_digital, meta
            # Fallback to OCR per-page
            images = _pdf_to_images(data)
            meta['pages'] = len(images)
            texts = []
            for img in images:
                pimg = _preprocess_image(img) if (os.getenv('PREPROCESS_ENABLE', '0') == '1') else img
                texts.append(_ocr_image(pimg))
            meta['source'] = 'pdf+ocr'
            return '\n\n'.join(texts), meta
        else:
            image = Image.open(io.BytesIO(data))
            pimg = _preprocess_image(image) if (os.getenv('PREPROCESS_ENABLE', '0') == '1') else image
            text = _ocr_image(pimg)
            meta['pages'] = 1
            meta['source'] = 'image+ocr'
            return text, meta
    except Exception as e:
        meta['error'] = str(e)
        return '', meta


def bytes_to_images(data: bytes, filename: str | None = None, dpi: int = 300, max_pages: int | None = None) -> List[Image.Image]:
    """
    Render input bytes into a list of PIL Images.
    - PDF: render each page (respecting max_pages)
    - Image: return a single PIL Image
    """
    images: List[Image.Image] = []
    try:
        if filename and filename.lower().endswith('.pdf'):
            with fitz.open(stream=data, filetype='pdf') as doc:
                page_count = len(doc)
                limit = page_count if max_pages is None else min(max_pages, page_count)
                for page_index in range(limit):
                    page = doc.load_page(page_index)
                    pix = page.get_pixmap(alpha=False, dpi=dpi)
                    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                    images.append(img)
        else:
            images.append(Image.open(io.BytesIO(data)).convert('RGB'))
    except Exception:
        # Best-effort: return empty list on render error
        return []
    return images
