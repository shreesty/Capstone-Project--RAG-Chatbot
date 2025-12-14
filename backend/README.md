# Backend (FastAPI)

Chat and session-aware RAG API that serves the vector index, conversation persistence, and the built frontend.

## Prereqs
- `pip` or `uv` (project uses `pyproject.toml`)
- FAISS index + metadata at `vector_store/faiss_sentences/index.faiss` and `vector_store/faiss_sentences/metadata.jsonl`
- Environment:
  - `LLM_BACKEND` (`groq` or `azure`, defaults to `azure`)
  - For Groq: `GROQ_API_KEY`
  - For Azure: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` (defaults to `2024-08-01-preview`)


## Run (dev)
```bash
uvicorn backend.app.main:app --reload --port 8000
```
The API will auto-serve a static frontend at `/ui/` if a Next.js export is present in `frontend/out`.

## Endpoints (quick)
- `GET /health`
- `GET /sessions` / `POST /sessions`
- `GET /sessions/{id}`
- `POST /sessions/{id}/rename` (body: `{"name": "New name"}`)
- `DELETE /sessions/{id}`
- `POST /query` — accepts `query`, `backend`, `model`, `session_id`, `session_name`

## Data
Conversation history is persisted under `backend/data/` (gitignored).
