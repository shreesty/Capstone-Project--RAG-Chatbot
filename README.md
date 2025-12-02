## RAG Pipeline Usage Guide

This project has a simple, script-based RAG pipeline with four stages:
1) Preprocess + translate scraped data
2) Embed chunks into FAISS
3) Query embeddings for inspection
4) Full RAG (retrieve + LLM answer)

Keep `.env` in the repo root with your keys (e.g., `GROQ_API_KEY`, `OPENAI_API_KEY`). The scripts call `load_dotenv()` so the keys are picked up automatically.

---

### 1) Preprocess & Translate
Script: `translator/preprocess.py`

Purpose: extract text from txt/html/docx/pdf/images, detect Nepali (`langdetect`), translate Nepali to English (Groq or OpenAI), and write normalized JSONL.

Key flags:
- `--root`: scraped data root (default `scraped_data`)
- `--translator`: `groq`, `openai`, or `none`
- `--translator-model`: optional model override (defaults: `groq` -> `llama-3.3-70b-versatile`; `openai` -> `gpt-4o-mini`)
- `--ocr-pdf-pages`: how many pages to OCR if PDFs have no extractable text (default 2)
- `--limit`: process only N files (dry runs or partial)
- `--start-index`: skip files before this index (useful to resume after rate limits)
- `--append`: append to output instead of overwriting (use when resuming)
- `--output`: JSONL destination (default `processed_data/english_documents.jsonl`)

Examples:
```
# Fresh run (Groq)
python translator/preprocess.py --root scraped_data --translator groq --ocr-pdf-pages 3 --output processed_data/english_documents.jsonl

# Quick dry run of 10 files
python translator/preprocess.py --root scraped_data --translator groq --limit 10 --output processed_data/sample.jsonl

# Resume after a rate limit (e.g., stopped at 442)
python translator/preprocess.py --root scraped_data --translator groq --start-index 442 --append --output processed_data/english_documents.jsonl
```

Notes:
- File order is stable (sorted) so `--start-index` reliably resumes.
- Use `--append` when resuming so previous lines are preserved.
- If you change API keys, just update `.env` and rerun; translation skips non-Nepali automatically.

---

### 2) Embed into FAISS
Script: `translator/embed.py`

Purpose: chunk processed docs, embed with SentenceTransformers, normalize, and store in FAISS + metadata JSONL.

Key flags:
- `--input`: processed JSONL (default `processed_data/english_documents.jsonl`)
- `--output-dir`: where to write `index.faiss` and `metadata.jsonl` (default `vector_store/faiss`)
- `--embedding-model`: SentenceTransformers model (default `sentence-transformers/all-MiniLM-L6-v2`)
- `--chunk-size`: characters per chunk (default 800)
- `--chunk-overlap`: overlap between chunks (default 200)
- `--batch-size`: embedding batch size (default 64)

Example:
```
python translator/embed.py --input processed_data/english_documents.jsonl --output-dir vector_store/faiss --embedding-model sentence-transformers/all-MiniLM-L6-v2 --chunk-size 800 --chunk-overlap 200
```

Output:
- `vector_store/faiss/index.faiss` (FAISS index)
- `vector_store/faiss/metadata.jsonl` (one line per chunk, with text and metadata)

---

### 3) Inspect Similarity Search
Script: `translator/query.py`

Purpose: run a similarity search against FAISS and print top matches with previews.

Key flags:
- `--query`: the search string (required)
- `--top-k`: number of results (default 5)
- `--index`: FAISS path (default `vector_store/faiss/index.faiss`)
- `--metadata`: metadata path (default `vector_store/faiss/metadata.jsonl`)
- `--embedding-model`: must match embedding step (default `sentence-transformers/all-MiniLM-L6-v2`)

Example:
```
python translator/query.py --query "scholarship eligibility for engineering students"  --top-k 5 --embedding-model sentence-transformers/all-MiniLM-L6-v2
```                                 

### 4) Full RAG (Retrieve + LLM Answer)
Script: `translator/rag.py`

Purpose: retrieve top-k chunks and have an LLM answer using only that context.

Key flags:
- `--query`: user question (required)
- `--top-k`: passages to retrieve (default 4)
- `--index`, `--metadata`, `--embedding-model`: same as query script
- `--llm-provider`: `groq` or `openai` (default `groq`)
- `--llm-model`: provider-specific model (default `llama-3.3-70b-versatile`; use e.g. `gpt-4o-mini` for OpenAI)

Examples:
```
# Groq
python translator/rag.py --query "entrance requirements for medical programs" --top-k 4 --embedding-model sentence-transformers/all-MiniLM-L6-v2 --llm-provider groq --llm-model llama-3.3-70b-versatile

# OpenAI
python translator/rag.py --query "scholarship eligibility for engineering students" --top-k 4 --embedding-model sentence-transformers/all-MiniLM-L6-v2 --llm-provider openai --llm-model gpt-4o-mini
```

### Workflow Recap
1) Preprocess/translate:
   - Fresh: `preprocess.py --translator groq ...`
   - Resume after rate limit: add `--start-index N --append`
2) Embed: `embed.py` (use the same embedding model for all steps)
3) Inspect: `query.py --query "..."`
4) RAG answer: `rag.py --query "..."`

Keep using `--start-index` + `--append` if you need to extend the processed JSONL later (after getting a higher rate limit or a new key). Then re-run `embed.py` to refresh the FAISS index before querying/RAG. 
