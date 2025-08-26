from __future__ import annotations
from typing import Tuple, Dict, Any
import io
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

def _ocr_image(image: Image.Image) -> str:
    # Default engine; can pass language via TESSERACT_LANG env later
    return pytesseract.image_to_string(image)

def _pdf_to_images(pdf_bytes: bytes) -> list[Image.Image]:
    images = []
    with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(alpha=False, dpi=200)
            img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            images.append(img)
    return images

def extract_text_from_bytes(data: bytes, filename: str | None = None) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    try:
        if filename and filename.lower().endswith('.pdf'):
            images = _pdf_to_images(data)
            meta['pages'] = len(images)
            texts = [ _ocr_image(img) for img in images ]
            meta['source'] = 'pdf+ocr'
            return '\n\n'.join(texts), meta
        else:
            image = Image.open(io.BytesIO(data))
            text = _ocr_image(image)
            meta['pages'] = 1
            meta['source'] = 'image+ocr'
            return text, meta
    except Exception as e:
        meta['error'] = str(e)
        return '', meta
