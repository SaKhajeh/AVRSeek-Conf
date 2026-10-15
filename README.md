# AVRSeek

**Antiviral Reaction Seek** — extract multi-step reaction knowledge from antiviral papers and patents.

Built beside [ReactionSeek](../README.md), with an antiviral-focused schema, OpenAI-compatible APIs (e.g. GapGPT), and a one-page upload UI. Supports **single or batch PDF upload**, JSON/Excel export, and processing-time estimates.

## Layout

```
AVRSeek/
  backend/          FastAPI pipeline + GapGPT client
  frontend/         React + Vite one-page UI
```

## Pipeline (v0.3)

1. **PDF parse + OCR** — native PyMuPDF text; if scanned/empty, Tesseract/PyMuPDF/easyocr (cached under `outputs/ocr_cache/`). Languages: **English, Russian, Chinese** (`OCR_LANGUAGES=eng+rus+chi_sim+chi_tra`; Farsi/Spanish/Hebrew later)  
2. **Text extract** — antiviral multi-step prompt, Example-aware chunking → JSON route  
3. **Table extract** — pdfplumber + LLM  
4. **Images** — skip logos; scheme VLM / optional RxnScribe; **page-read VLM** when text is still empty  
5. **Reconcile** — merge text + page-read + tables + scheme steps (names, reagents, conditions kept)  
6. **Standardize** — PubChem/OPSIN/LLM + RDKit + time normalization  

Excel download sheets: **Summary**, **Routes**, **Steps**, **Compounds**, **Warnings**.

### OCR

Install [Tesseract](https://github.com/tesseract-ocr/tesseract) and optionally:

```bash
pip install pytesseract
```

Without Tesseract, scanned PDFs still get **VLM page-read** (slower, uses `MODEL_VLM`).

If you re-process a PDF after this upgrade, delete the old cache entry in `outputs/cache/index.json` (or the job JSON) so it is not reused.

### Eval

```bash
cd AVRSeek/backend
.venv\Scripts\python.exe -m app.eval.run_eval --results outputs
```  

## Setup

### Backend

```bash
cd AVRSeek/backend
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
copy .env.example .env
# edit .env — set OPENAI_API_KEY to your GapGPT key
```

`.env` example (GapGPT):

```env
OPENAI_API_KEY=YOUR_GAPGPT_API_KEY
OPENAI_BASE_URL=https://api.gapgpt.app/v1
MODEL_TEXT=claude-opus-5
MODEL_VLM=claude-opus-5
MODEL_STANDARDIZE=claude-opus-5
ENABLE_OCR=true
ENABLE_VLM_PAGE_READ=true
AVRSEEK_PORT=8765
```

Run API:

```bash
cd AVRSeek/backend
.venv\Scripts\activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8765
```

Optional: `pip install rdkit` for SMILES validation in the standardize stage.

### Frontend

Node 16+ recommended. If `npm` hangs on Windows, call the CLI via Node:

```bash
cd AVRSeek/frontend
node "C:\Program Files\nodejs\node_modules\npm\bin\npm-cli.js" install
node "C:\Program Files\nodejs\node_modules\npm\bin\npm-cli.js" run dev
```

Or normally:

```bash
cd AVRSeek/frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173 — Vite proxies `/api` to the backend (`AVRSEEK_PORT`, default 8765 in this repo).

## Usage

1. Start backend + frontend  
2. Upload one or more PDFs (up to 20) — papers or patents  
3. Review the **estimated processing time** (all pages, **every embedded image** via VLM, tables, standardization)
4. Optionally override the text model name  
5. Run batch extraction — the UI shows **current file**, **pipeline stage**, and **remaining time**
6. Download combined **JSON** or **Excel** (Summary, Routes, Steps, Compounds, Warnings), or per-file exports  

### Batch API

| Endpoint | Description |
|----------|-------------|
| `POST /api/extract/batch/estimate` | Quick PDF scan (no LLM) → time estimate |
| `POST /api/extract/batch/stream` | SSE batch run; skips PDFs already in cache (matched by SHA-256 of file bytes) |
| `GET /api/batch/{batch_id}/download.json` | Combined batch JSON |
| `GET /api/batch/{batch_id}/download.xlsx` | Combined batch Excel |
| `GET /api/jobs/{job_id}/download.json` | Single-file JSON |
| `GET /api/jobs/{job_id}/download.xlsx` | Single-file Excel |

## Next steps

- Richer PDF layout (Docling/Marker) for complex multi-column patents  
- Optional RxnScribe/MolScribe for cropped schemes (`OCSR_LOCAL_MODEL`)  
- Automated literature/patent fetch APIs into the same ingest path 
