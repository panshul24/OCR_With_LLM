from __future__ import annotations

from typing import Dict, Any
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
    "Rules: No markdown, no code fences, only a single JSON object. Use null when unknown."
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
            return _parse_json(output)
        except Exception:
            return {"raw": output}
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
