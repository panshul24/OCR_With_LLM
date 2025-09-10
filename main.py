"""FastAPI entrypoints and inline upload UI.

Provides:
- GET / → index(): dark themed upload UI
- POST /api/process → process(): unified single/multi file processing
- POST /api/process-batch → process_batch(): legacy alias
"""
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path
from typing import List
import os
from app.ocr import extract_text_from_bytes
from app.llm import categorize_text
from app.fuzzy import load_keywords, compute_keyword_scores

app = FastAPI(title="OCR Categorizer and Extractor API")
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "/data/config"))
KEYWORDS_FILE = CONFIG_DIR / "keywords.json"

@app.get('/', response_class=HTMLResponse)
async def index():
    return (
        """
        <!doctype html>
        <html>
        <head>
          <meta charset=\"utf-8\" />
          <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
          <title>OCR Categorizer</title>
          <style>
            :root { --bg:#0b0f17; --panel:#121826; --muted:#9aa4b2; --text:#e5e7eb; --brand:#5b8cff; --border:#1f2a44; --accent:#172036; --accent-hover:#1d2944; }
            * { box-sizing: border-box; }
            body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: var(--bg); color: var(--text); }
            .wrapper { max-width: 1000px; margin: 48px auto; padding: 0 16px; }
            .card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
            .header { padding: 18px 22px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
            h2 { margin: 0; font-size: 20px; }
            .muted { color: var(--muted); font-size: 13px; }
            .content { padding: 18px 22px; }
            .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
            select, button { height: 40px; border-radius: 10px; border: 1px solid var(--border); padding: 0 12px; background: var(--accent); color: var(--text); }
            select:hover, button:hover { background: var(--accent-hover); }
            label.btn { height: 40px; border-radius: 10px; border: 1px solid var(--border); padding: 0 12px; background: var(--accent); color: var(--text); display: inline-flex; align-items: center; cursor: pointer; }
            label.btn:hover { background: var(--accent-hover); }
            button.primary { background: var(--brand); border-color: var(--brand); color: #fff; }
            button[disabled] { opacity: .7; cursor: not-allowed; }
            .visually-hidden { position:absolute; left:-9999px; width:1px; height:1px; opacity:0; }
            .dropzone { display: grid; place-items: center; min-height: 170px; padding: 18px; border: 2px dashed var(--border); border-radius: 12px; background: var(--accent); color: var(--muted); cursor: pointer; text-align: center; }
            .dropzone.highlight { border-color: var(--brand); background: #13203e; color: var(--text); }
            .filelist { margin: 10px 0 0; padding: 0; list-style: none; max-height: 180px; overflow: auto; font-size: 13px; }
            .filelist li { padding: 6px 0; border-bottom: 1px dashed var(--border); display: flex; justify-content: space-between; color: var(--muted); }
            .actions { margin-top: 12px; display: flex; gap: 10px; align-items: center; }
            pre { background: #0a0e1a; color: #d1e0ff; padding: 14px; border-radius: 10px; overflow: auto; border: 1px solid var(--border); }
            .result-head { display: flex; align-items: center; justify-content: space-between; }
            .small { font-size: 12px; color: var(--muted); }
            .chat { background: #10172a; border: 1px solid var(--border); border-radius: 10px; padding: 12px; margin-bottom: 10px; color: #c7d2fe; }
          </style>
        </head>
        <body>
          <div class=\"wrapper\">
            <div class=\"card\">
              <div class=\"header\"><div>
                <h2>OCR Categorizer</h2>
                <div class=\"muted\">Upload one or many docs (or a folder on WebKit). We'll OCR and extract via a local LLM.</div>
              </div></div>
              <div class=\"content\">
                <div id=\"upload\">
                  <div id=\"dropzone\" class=\"dropzone\">
                    <div>
                      <div style=\"font-weight:600\">Drag & drop files or folder here</div>
                      <div class=\"small\">Accepted: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP</div>
                      <div style=\"margin-top:8px\">
                        <button class=\"btn\" type=\"button\" id=\"uploadFilesBtn\" onclick=\"document.getElementById('filesInput').click();return false;\">Upload Files</button>
                        <button class=\"btn\" type=\"button\" id=\"uploadFolderBtn\" onclick=\"document.getElementById('folderInput').click();return false;\">Upload Folder</button>
                      </div>
                    </div>
                  </div>
                  <input class=\"visually-hidden\" type=\"file\" id=\"filesInput\" name=\"files\" accept=\"application/pdf,image/*\" multiple />
                  <input class=\"visually-hidden\" type=\"file\" id=\"folderInput\" name=\"files\" multiple webkitdirectory directory />

                  <ul id=\"filelist\" class=\"filelist\" style=\"display:none\"></ul>

                  <div class=\"row\" style=\"margin-top: 12px;\">
                    <select id=\"model\" name=\"model\">
                      <option value=\"llama3.1\">llama3.1</option>
                      <option value=\"llama3\">llama3</option>
                      <option value=\"mistral\">mistral</option>
                    </select>
                    <div class=\"actions\">
                      <button class=\"primary\" type=\"button\" id=\"submitBtn\">Process</button>
                      <span class=\"small\" id=\"status\"></span>
                    </div>
                  </div>
                </div>

                <div id=\"result\" style=\"margin-top:16px; display:none;\">
                  <div class=\"result-head\">
                    <h3 style=\"margin:0\">Result</h3>
                    <button type=\"button\" id=\"copyBtn\">Copy JSON</button>
                  </div>
                  <div id=\"summary\" class=\"chat\" style=\"display:none\"></div>
                  <div id=\"debug\" style=\"display:none\"> 
                    <div class=\"small\">OCR (Tesseract) Text</div>
                    <pre id=\"ocrText\"></pre>
                    <div class=\"small\">Text LLM Prompt</div>
                    <pre id=\"textPrompt\"></pre>
                    <div class=\"small\">Text LLM Raw Output</div>
                    <pre id=\"textRaw\"></pre>
                    <div class=\"small\" id=\"visionPromptLabel\" style=\"display:none\">Vision LLM Prompt</div>
                    <pre id=\"visionPrompt\" style=\"display:none\"></pre>
                    <div class=\"small\" id=\"visionRawLabel\" style=\"display:none\">Vision LLM Raw Output</div>
                    <pre id=\"visionRaw\" style=\"display:none\"></pre>
                  </div>
                  <pre id=\"json\"></pre>
                </div>
              </div>
            </div>
          </div>

          <script>
            const drop = document.getElementById('dropzone');
            const filesInput = document.getElementById('filesInput');
            const folderInput = document.getElementById('folderInput');
            const uploadFilesBtn = document.getElementById('uploadFilesBtn');
            const uploadFolderBtn = document.getElementById('uploadFolderBtn');
            const fileListEl = document.getElementById('filelist');
            const statusEl = document.getElementById('status');
            const copyBtn = document.getElementById('copyBtn');
            const submitBtn = document.getElementById('submitBtn');

            let selectedFiles = [];

            function renderFileList() {
              fileListEl.innerHTML = '';
              if (!selectedFiles.length) { fileListEl.style.display = 'none'; statusEl.textContent=''; return; }
              fileListEl.style.display = 'block';
              selectedFiles.forEach(function(f){
                const li = document.createElement('li');
                const left = document.createElement('span'); left.textContent = f.webkitRelativePath || f.name;
                const right = document.createElement('span'); right.textContent = (f.size/1024).toFixed(1) + ' KB';
                li.appendChild(left); li.appendChild(right); fileListEl.appendChild(li);
              });
              statusEl.textContent = String(selectedFiles.length) + ' file(s) selected';
            }

            function setBusy(busy){ submitBtn.disabled = !!busy; statusEl.textContent = busy ? 'Processing…' : statusEl.textContent; }

            drop.addEventListener('click', function(e){ if (e.target === drop) filesInput.click(); });
            uploadFilesBtn.addEventListener('click', function(e){ e.preventDefault(); e.stopPropagation(); filesInput.click(); });
            uploadFolderBtn.addEventListener('click', function(e){ e.preventDefault(); e.stopPropagation(); folderInput.click(); });
            document.addEventListener('change', function(e){
              if (e.target === filesInput || e.target === folderInput) {
                selectedFiles = [].concat(Array.from(filesInput.files||[]), Array.from(folderInput.files||[]));
                renderFileList();
              }
            }, true);
            filesInput.addEventListener('change', function(){ selectedFiles = [].concat(Array.from(filesInput.files||[]), Array.from(folderInput.files||[])); renderFileList(); });
            folderInput.addEventListener('change', function(){ selectedFiles = [].concat(Array.from(filesInput.files||[]), Array.from(folderInput.files||[])); renderFileList(); });
            ['dragenter','dragover'].forEach(function(ev){ drop.addEventListener(ev, function(e){ e.preventDefault(); e.stopPropagation(); drop.classList.add('highlight'); }); });
            ['dragleave','drop'].forEach(function(ev){ drop.addEventListener(ev, function(e){ e.preventDefault(); e.stopPropagation(); drop.classList.remove('highlight'); }); });
            drop.addEventListener('drop', function(e){ const files = Array.from((e.dataTransfer||{}).files||[]); if (files.length) { selectedFiles = files; renderFileList(); } });

            async function processFiles(){
              const model = document.getElementById('model').value;
              const files = selectedFiles.length ? selectedFiles : [].concat(Array.from(filesInput.files||[]), Array.from(folderInput.files||[]));
              if (!files.length) { alert('Please select at least one file.'); return; }
              setBusy(true);
              try {
                const fd = new FormData();
                files.forEach(function(f,i){ fd.append('files', f, f.name || ('file_' + i)); });
                fd.append('model', model);
                const res = await fetch('/api/process', { method:'POST', body: fd });
                if (!res.ok) throw new Error('HTTP ' + res.status);
                const data = await res.json();
                const summaryEl = document.getElementById('summary');
                summaryEl.style.display = 'block';
                // Fill debug panel
                const dbg = document.getElementById('debug');
                const ocrEl = document.getElementById('ocrText');
                const tPrompt = document.getElementById('textPrompt');
                const tRaw = document.getElementById('textRaw');
                const vPrompt = document.getElementById('visionPrompt');
                const vRaw = document.getElementById('visionRaw');
                const vPromptLabel = document.getElementById('visionPromptLabel');
                const vRawLabel = document.getElementById('visionRawLabel');
                dbg.style.display = 'block';
                ocrEl.textContent = data.text_preview || '';
                // Support both direct object and items[]
                const firstItem = Array.isArray(data.items) ? (data.items[0] || {}) : data;
                const src = (firstItem.categories && firstItem.categories.sources) || firstItem.categories || {};
                tPrompt.textContent = src._debug_prompt || src.text_prompt || '';
                tRaw.textContent = src._debug_raw || src.text_raw || '';
                if (src.vision_prompt || src._vision_debug_prompt) { vPrompt.textContent = src.vision_prompt || src._vision_debug_prompt; vPrompt.style.display='block'; vPromptLabel.style.display='block'; } else { vPrompt.style.display='none'; vPromptLabel.style.display='none'; }
                if (src.vision_raw || src._vision_debug_raw) { vRaw.textContent = src.vision_raw || src._vision_debug_raw; vRaw.style.display='block'; vRawLabel.style.display='block'; } else { vRaw.style.display='none'; vRawLabel.style.display='none'; }
                function formatAddress(addr){
                  if (!addr) return null;
                  if (typeof addr === 'string') return addr;
                  const parts = [];
                  if (addr.street) parts.push(addr.street);
                  if (addr.city) parts.push(addr.city);
                  if (addr.state) parts.push(addr.state);
                  if (addr.postal_code) parts.push(addr.postal_code);
                  if (addr.country) parts.push(addr.country);
                  return parts.join(', ');
                }
                function tryParseJSON(s){ try { return JSON.parse(s); } catch(e){ return null; } }
                function catObjFrom(item){
                  let c = item.categories || {};
                  if (c && typeof c.raw === 'string') {
                    const p = tryParseJSON(c.raw); if (p) return p;
                  }
                  const src = (c && c.sources) ? c.sources : {};
                  if (src && typeof src.text_llm_raw === 'string') {
                    const p = tryParseJSON(src.text_llm_raw); if (p) return p;
                  }
                  if (src && typeof src.vision_llm_raw === 'string') {
                    const p = tryParseJSON(src.vision_llm_raw); if (p) return p;
                  }
                  return c || {};
                }
                function line(item){
                  const c = catObjFrom(item); const fields = [];
                  if (c.name) fields.push('name: ' + c.name);
                  if (c.date) fields.push('date: ' + c.date);
                  if (c.id_number) fields.push('id: ' + c.id_number);
                  const addrStr = formatAddress(c.address);
                  if (addrStr) fields.push('address: ' + addrStr);
                  if (c.email) fields.push('email: ' + c.email);
                  if (c.phone) fields.push('phone: ' + c.phone);
                  const prov = c.provider || {};
                  const npi = (prov.personal && prov.personal.npi) ? prov.personal.npi : null;
                  if (npi) fields.push('npi: ' + npi);
                  const cat = c.document_type || (c.extra && c.extra.document_type) || 'unknown';
                  return 'Category: ' + cat + ' — ' + (fields.length ? ('Fields: ' + fields.join(', ')) : 'Fields: -');
                }
                summaryEl.textContent = Array.isArray(data.items) ? data.items.map(line).join(' | ') : line(data);
                document.getElementById('result').style.display = 'block';
                document.getElementById('json').textContent = JSON.stringify(data, null, 2);
              } catch (err) { console.error(err); alert('Upload failed. Check console/network tab.'); }
              finally { setBusy(false); statusEl.textContent = ''; }
            }
            submitBtn.addEventListener('click', function(e){ e.preventDefault(); processFiles(); });

            copyBtn.addEventListener('click', async function(){
              try { const txt = document.getElementById('json').textContent || ''; await navigator.clipboard.writeText(txt); statusEl.textContent = 'Copied to clipboard'; setTimeout(function(){ statusEl.textContent=''; }, 1500); }
              catch(e){ alert('Copy failed'); }
            });
          </script>
        </body>
        </html>
        """
    )


def _process_many(uploaded: List[UploadFile], model: str):
    keywords = load_keywords(KEYWORDS_FILE)
    items = []
    for f in uploaded:
        content = f.file.read()
        text, meta = extract_text_from_bytes(content, filename=f.filename)
        result = categorize_text(text, model=model)
        fuzzy = compute_keyword_scores(text, keywords)
        items.append({
            'filename': f.filename,
            'ocr_meta': meta,
            'text_preview': text[:1000],
            'categories': result,
            'fuzzy_scores': fuzzy
        })
    if len(items) == 1:
        return items[0]
    return {'items': items}


@app.post('/api/process')
async def process(files: List[UploadFile] = File(...), model: str = Form(default='llama3.1')):
    for up in files:
        await up.seek(0)
    return JSONResponse(_process_many(files, model))


@app.post('/api/process-batch')
async def process_batch(files: List[UploadFile] = File(...), model: str = Form(default='llama3.1')):
    return JSONResponse(_process_many(files, model))
