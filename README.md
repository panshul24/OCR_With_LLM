# OCR_With_LLM

## OCR Categorizer and Extractor (API + Ollama + OpenWebUI)

This stack provides an API-only service to OCR PDFs/images and extract structured JSON via a local LLM. OpenWebUI is included for prompt-based interaction with your local models.

### Stack
- API: FastAPI (OCR via Tesseract + born‑digital via PyMuPDF + layout via pymupdf4llm)
- LLM: Ollama (`llama3.1`)
- UI: Built‑in dark upload UI (/) + optional OpenWebUI (wired to Ollama)

### Run (Docker Compose)
```bash
# from the project directory
docker compose up -d --build

# Pull a model into Ollama (first time)
docker exec -it ollama ollama pull llama3.1
```

- API: `http://localhost:8000`
  - GET `/` → upload UI
  - POST `/api/process` → multipart form-data: `files` (one or many), `model` (default `llama3.1`)
- OpenWebUI: `http://localhost:3000`
- Ollama API: `http://localhost:11434`

### How extraction works
1) Input → `extract_text_from_bytes` (app/ocr.py)
   - PDF: born‑digital first (`pymupdf4llm.to_markdown`). If sparse, fallback to OCR (render 300 DPI → optional preprocess → Tesseract).
   - Image: optional preprocess → Tesseract.
2) Text → `categorize_text` (app/llm.py) → Ollama; strict JSON enforced, with debug fields attached.
3) Telemetry → `compute_keyword_scores` (RapidFuzz) for quick relevance hints.
4) Response → JSON: `{ filename, ocr_meta, text_preview, categories, fuzzy_scores }`.

### Using OpenWebUI to drive categorization
OpenWebUI is exposed as its own app. If you want OpenWebUI to call the API after a file upload, you can:
- Build a small OpenWebUI tool/plugin that uploads the file to `/api/process` and displays the JSON; or
- Keep using OpenWebUI for prompts, and call the API via `curl`/automation for extraction.

### Local development without Docker
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Configuration (env)
- OCR preprocess toggles: `PREPROCESS_ENABLE`, `PREPROCESS_BINARIZE`, `PREPROCESS_DESKEW`, `PREPROCESS_DENOISE`
- Tesseract: `TESSERACT_LANG` (default `eng`), `TESSERACT_PSM` (`6`), `TESSERACT_OEM` (`3`), `TESSERACT_CHAR_WHITELIST`
- PDF heuristics: `PDF_BORNDIGITAL_CHECK=1`, `PDF_MIN_CHARS=50`
- Watcher: `INBOX_DIR=/data/inbox`, `OUTBOX_DIR=/data/outbox`, `WATCH_INTERVAL`
- Models: pass `model` with the request (defaults to `llama3.1`)

### Notes
- Vision LLM path exists but is currently disabled in the API route (per current requirements). Can be re‑enabled via `hybrid_extract_and_categorize`.
- For large PDFs, OCR can take time. The watcher service provides a background workflow.
