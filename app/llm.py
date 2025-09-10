"""LLM interaction utilities (Ollama + optional vision backends).

Provides:
- categorize_text: strict JSON extraction from OCR/born-digital text
- segment_and_categorize: optional multi-document segmentation
- vision helpers (available, currently not used in API route)
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import os
import requests

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")

SYSTEM_PROMPT = (
    "You are a precise information extractor for administrative documents.\n"
    "Given raw OCR text, return STRICT JSON with these keys: \n"
    "- document_type: one of [license, degree_certificate, transcript, government_id, insurance, medical_form, invoice, bank_statement, utility_bill, tax_document, other] \n"
    "- name: full name if present or null \n"
    "- date: main date in ISO 8601 YYYY-MM-DD or null \n"
    "- id_number: primary identifier (registration/roll/policy/etc) or null \n"
    "- amount, address, email, phone: nullable \n"
    "- extra: object containing any additional key/values (nullable).\n"
    "Rules: No markdown, no code fences, only a single JSON object. Use null when unknown. Validate: if uncertain, prefer null."
)


def _parse_json(output: str):
    import json, re
    cleaned = output.strip()
    # Strip common markdown fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n?", "", cleaned)
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def categorize_text(text: str, model: str = "llama3.1") -> Dict[str, Any]:
    prompt = f"OCR TEXT:\n{text[:12000]}\n\nReturn ONLY the JSON object."
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        output = data.get("response", "").strip()
        try:
            parsed = _parse_json(output)
            if isinstance(parsed, dict):
                parsed.setdefault("_debug_prompt", prompt)
                parsed.setdefault("_debug_raw", output)
            return parsed
        except Exception:
            # Try to salvage a JSON object substring
            import re, json as _json
            m = re.search(r"\{[\s\S]*\}\s*$", output)
            if m:
                try:
                    parsed = _parse_json(m.group(0))
                    if isinstance(parsed, dict):
                        parsed.setdefault("_debug_prompt", prompt)
                        parsed.setdefault("_debug_raw", output)
                        return parsed
                except Exception:
                    pass
            return {"raw": output, "_debug_prompt": prompt, "_debug_raw": output}
    except Exception as e:
        return {"error": str(e)}


def segment_and_categorize(text: str, model: str = "llama3.1") -> Any:
    """
    Return either a list of segments with categories/fields or a single-object fallback.
    Expected JSON output shape:
    [
      {
        "document_type": "license" | "form" | "id" | "invoice" | ...,
        "fields": { ... extracted key/values ... },
        "text_span": "...optional excerpt..."
      },
      ...
    ]
    """
    system = (
        "You are an expert document triage and extraction system. Given raw OCR text that may contain one or "
        "more documents back-to-back, segment the text into logical documents and for each segment return a "
        "JSON object with: document_type (from [license, degree_certificate, transcript, government_id, insurance, medical_form, invoice, bank_statement, utility_bill, tax_document, other]), "
        "fields (key-value map like name/date/id_number/amount/address/phone/email/etc), "
        "and an optional text_span excerpt. Return a JSON array only. Use nulls when unknown."
    )
    prompt = f"OCR TEXT:\n{text[:16000]}\n\nReturn ONLY the JSON array."
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        output = data.get("response", "").strip()

        import json
        try:
            parsed = _parse_json(output)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception:
            return {"raw": output}
    except Exception as e:
        return {"error": str(e)}


def _ollama_vision_generate(prompt: str, images_b64: List[str], model: str) -> str:
    """Call Ollama multimodal model with image(s). images_b64: list of base64-encoded PNG/JPEG strings."""
    # Try requested model, then fall back to common vision models
    candidates = [model]
    if model != "qwen2.5-vl":
        candidates.append("qwen2.5-vl")
    if model != "llama3.2-vision":
        candidates.append("llama3.2-vision")
    last_err = None
    for m in candidates:
        try:
            payload = {
                "model": m,
                "prompt": prompt,
                "images": images_b64,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.1},
            }
            resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            last_err = str(e)
            continue
    return last_err or "vision generate failed"


def _openrouter_vision_generate(prompt: str, images_b64: List[str], model: str) -> str:
    """Call OpenRouter (OpenAI-compatible) vision model with data URLs.
    Expects OPENROUTER_API_KEY in env. Model example: 'qwen2.5-vl-7b-instruct'.
    """
    import json
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "missing OPENROUTER_API_KEY"
    try:
        data_urls = [f"data:image/png;base64,{b64}" for b64 in images_b64]
        contents = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": u}} for u in data_urls
        ]
        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": contents}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    except Exception as e:
        # Return a structured error string that upstream can show in sources
        return f"openrouter_error: {e}"


def vision_extract(images: List["Image.Image"], preferred_model: str | None = None) -> Dict[str, Any]:
    """Run a vision LLM over one or more images to extract structured fields.
    - preferred_model: override (for example 'qwen2-vl' or 'llama3.2-vision') else auto-select.
    Returns JSON object or {raw: ...} on fallback.
    """
    import base64
    import io as _io
    import json

    # Choose backend: OpenRouter if API key exists, else Ollama
    use_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    model = preferred_model or (
        os.getenv("OPENROUTER_VISION_MODEL", "qwen2.5-vl-7b-instruct") if use_openrouter
        else os.getenv("OLLAMA_VISION_MODEL", "qwen2.5-vl")
    )

    # Encode a page budget to reduce payload size
    page_budget = int(os.getenv("VISION_PAGE_BUDGET", "2"))
    chosen = images[:page_budget]
    images_b64: List[str] = []
    for img in chosen:
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    system = (
        "You are a precise vision information extractor for administrative documents. "
        "Return STRICT JSON with keys: document_type (from [license, degree_certificate, transcript, government_id, "
        "insurance, medical_form, invoice, bank_statement, utility_bill, tax_document, other]), name, date (YYYY-MM-DD), "
        "id_number, amount, address, email, phone, extra (object). Use null when unknown."
    )
    prompt = (
        "Analyze these page images and extract the fields. Return ONLY a JSON object."
    )
    if use_openrouter:
        output = _openrouter_vision_generate(prompt, images_b64, model)
    else:
        output = _ollama_vision_generate(prompt, images_b64, model)
    try:
        return json.loads(output)
    except Exception:
        return {"raw": output}


def hybrid_extract_and_categorize(text: str, images: List["Image.Image"], text_model: str = "llama3.1", vision_model: str | None = None) -> Dict[str, Any]:
    """Run both text and vision extraction, then fuse with simple guardrails.
    Returns single JSON with document_type and fields; includes confidence + provenance per field.
    """
    import re
    import json

    text_json = categorize_text(text, model=text_model) if text else {}
    vision_json = vision_extract(images, preferred_model=vision_model) if images else {}

    def norm_date(val: Any) -> Any:
        if not isinstance(val, str):
            return None if val is None else val
        m = re.search(r"(20\d{2})[-/ ]?(\d{2})[-/ ]?(\d{2})", val)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return val

    def norm_email(val: Any) -> Any:
        if not isinstance(val, str):
            return None if val is None else val
        return val if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", val.strip()) else None

    def norm_phone(val: Any) -> Any:
        if not isinstance(val, str):
            return None if val is None else val
        digits = re.sub(r"\D", "", val)
        return digits if len(digits) >= 7 else None

    def norm_amount(val: Any) -> Any:
        if val is None:
            return None
        s = str(val)
        m = re.search(r"([0-9]+(?:\.[0-9]{1,2})?)", s.replace(",", ""))
        return float(m.group(1)) if m else None

    def choose(field: str) -> Tuple[Any, str, float]:
        tv = text_json.get(field) if isinstance(text_json, dict) else None
        vv = vision_json.get(field) if isinstance(vision_json, dict) else None
        if field == "date":
            tv = norm_date(tv)
            vv = norm_date(vv)
        # Prefer agreement
        if tv is not None and vv is not None:
            if str(tv).strip() == str(vv).strip():
                return tv, "both", 0.95
        # Otherwise prefer vision for visual fields, text for address-like
        visual_pref = {"document_type", "name", "id_number"}
        text_pref = {"address"}
        if field in visual_pref and vv is not None:
            return vv, "vision", 0.8
        if field in text_pref and tv is not None:
            return tv, "text", 0.8
        # fallback
        if vv is not None:
            return vv, "vision", 0.6
        if tv is not None:
            return tv, "text", 0.6
        return None, "none", 0.0

    doc_type, src_dt, conf_dt = choose("document_type")
    name, src_name, conf_name = choose("name")
    date, src_date, conf_date = choose("date")
    id_number, src_id, conf_id = choose("id_number")
    amount, src_amount, conf_amount = choose("amount")
    address, src_addr, conf_addr = choose("address")
    email, src_email, conf_email = choose("email")
    phone, src_phone, conf_phone = choose("phone")

    # Normalize common fields
    if email is not None:
        email = norm_email(email)
    if phone is not None:
        phone = norm_phone(phone)
    if amount is not None:
        amount = norm_amount(amount)

    allowed_types = {"license","degree_certificate","transcript","government_id","insurance","medical_form","invoice","bank_statement","utility_bill","tax_document","other"}
    if isinstance(doc_type, str) and doc_type not in allowed_types:
        doc_type = "other"

    extra_text = text_json.get("extra") if isinstance(text_json, dict) else None
    extra_vision = vision_json.get("extra") if isinstance(vision_json, dict) else None
    extra = extra_vision or extra_text

    return {
        "document_type": doc_type,
        "name": name,
        "date": date,
        "id_number": id_number,
        "amount": amount,
        "address": address,
        "email": email,
        "phone": phone,
        "extra": extra,
        "confidence": {
            "document_type": conf_dt,
            "name": conf_name,
            "date": conf_date,
            "id_number": conf_id,
            "amount": conf_amount,
            "address": conf_addr,
            "email": conf_email,
            "phone": conf_phone
        },
        "provenance": {
            "document_type": src_dt,
            "name": src_name,
            "date": src_date,
            "id_number": src_id,
            "amount": src_amount,
            "address": src_addr,
            "email": src_email,
            "phone": src_phone
        },
        "sources": {
            "text_raw": text,
            "vision_raw": vision_json if isinstance(vision_json, dict) else {"raw": vision_json},
        }
    }
