# OCR_With_LLM

## OCR Categorizer and Extractor (API + Ollama + OpenWebUI)

This stack provides an API-only service to OCR PDFs/images and extract structured JSON via a local LLM. OpenWebUI is included for prompt-based interaction with your local models.

### Stack
- API: FastAPI (OCR via Tesseract + PyMuPDF)
- LLM: Ollama
- UI: OpenWebUI (optional, already wired to Ollama)

### Run (Docker Compose)
```bash
# from the project directory
docker compose up -d --build

# Pull a model into Ollama (first time)
docker exec -it ollama ollama pull llama3.1
```

- API available at: `http://localhost:8000`
  - Endpoint: `POST /api/process` (multipart form-data: `file`, `model`)
- OpenWebUI at: `http://localhost:3000`
- Ollama API at: `http://localhost:11434`

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

### Notes
- Ensure Tesseract is present in the API image (already installed in Dockerfile).
- For big PDFs, OCR may take time. Consider queuing or async processing later.
- You can change the Ollama model by passing `model` (e.g., `llama3`, `mistral`).
