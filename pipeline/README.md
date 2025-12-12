# NLLB-Aligned Pipeline

This folder holds the data pipeline for sentence-wise translation with NLLB-600M.

## Stage 1: Extraction (ready)
- Script: `pipeline/extract.py`
- Reads `scraped_data/**/metadata.jsonl`, prefers the `text/` versions, falls back to raw HTML, and also sweeps `docs/`, `images/`, and other supported files.
- Supports HTML, TXT, PDF (with OCR fallback), DOCX, and images (OCR).
- Output: JSONL at `processed_data/extracted_documents.jsonl` by default, one record per source with `{source_path, source_domain, source_url, kind, language, text, length}`.

Run:
```bash
python pipeline/extract.py 
```

## Stage 2: Sentence splitting (ready)
- Script: `pipeline/split_sentences.py`
- Input: `processed_data/extracted_documents.jsonl`
- Output: `processed_data/sentences.jsonl` with overlap-aware chunks `{uid, doc_id, sentence_index, source_path, source_domain, source_url, kind, doc_language, sentence}`.
 - Uses LangChain `RecursiveCharacterTextSplitter` with configurable `--chunk-size`, `--chunk-overlap`, `--min-chars`, and optional `--separators` (comma-separated, defaults to sentence-biased separators).

Run:
```bash
python pipeline/split_sentences.py 
```

## Stage 3: Translation with NLLB-600M (ready)
- Script: `pipeline/translate.py`
- Input: `processed_data/sentences.jsonl`
- Output: `processed_data/translated_sentences.jsonl` with added translation metadata.
- Defaults: translates only Nepali (`doc_language == "ne"`) sentences using NLLB language codes `npi_Deva` (src) → `eng_Latn` (tgt).

Azure option: add `--translator-backend azure --azure-deployment <deployment>` to use Azure OpenAI GPT-4.0-mini instead of NLLB (requires `AZURE_OPENAI_*` env vars).

Run (ensure the model is downloaded/cached):
```bash
python pipeline/translate.py 
```

## Stage 4: Embeddings (ready)
- Script: `pipeline/embed_sentences.py`
- Input: `processed_data/translated_sentences.jsonl`
- Output: FAISS index + metadata in `vector_store/faiss_sentences/`.
- By default embeds the translated English text; add `--no-translation` to embed original sentences.

Run:
```bash
python pipeline/embed_sentences.py
```

## Stage 5: Query testing (ready)
- Script: `pipeline/query_tester.py`
- Input: FAISS index + metadata from `vector_store/faiss_sentences/`.
- Runs a similarity search against the saved vectors to sanity check retrieval.

Run:
```bash
python pipeline/query_tester.py --query "your search text" --top-k 5 
```

## Stage 6: RAG with Llama via Groq (ready)
- Script: `pipeline/rag_query.py`
- Input: FAISS index + metadata; uses the same embedding model for query encoding and Groq's Llama model for generation.
- Requires `GROQ_API_KEY` in the environment (already read from `.env` if loaded).
- Azure option: use `--llm-backend azure --azure-deployment <deployment>` (requires `AZURE_OPENAI_*` env vars; `--model` is treated as the deployment name).

Run:
```bash
python pipeline/rag_query.py --query "your question" --top-k 5 --print-context  
```
Add `--query-file path/to/query.txt` for longer prompts. Adjust `--temperature` or choose `--model llama-3.3-8b-instant` if you want a lighter/cheaper option. If you see a decommission notice, switch to the latest `llama-3.3-*` model names above.

## Stage 7: FastAPI RAG service (ready)
- Script: `pipeline/api_server.py`
- Exposes `/health` and `/query` endpoints for retrieval-augmented answers powered by Groq's Llama models.
- Requires `GROQ_API_KEY` in the environment; optionally set `FAISS_INDEX_PATH`, `FAISS_METADATA_PATH`, `EMBEDDING_MODEL` to override defaults.
- Azure option: set `LLM_BACKEND=azure` and populate `AZURE_OPENAI_*` in `.env`. `model` field in requests is treated as the Azure deployment name when backend=azure.

Run:
```bash
uvicorn pipeline.api_server:app --host localhost --port 8000
```

Example query:
```bash

http://localhost:8000/docs 
```

## Optional frontend (dev)
- Static UI lives in `frontend/`. When present, the API serves it at `/ui` (CORS open for local dev). Launch the API and visit: http://localhost:8000/ui/
