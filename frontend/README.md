# Frontend (Next.js)

Chat-style UI (ChatGPT-like) for the NepEd RAG backend.

## Prereqs
- Backend running at `http://localhost:8000` (or set `NEXT_PUBLIC_API_BASE`)

## Install
```bash
cd frontend
npm install
```

## Dev
```bash
npm run dev
# app at http://localhost:3000
```
Use `NEXT_PUBLIC_API_BASE` to point to a non-default backend origin:
```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

## Build & Export (served by FastAPI)
```bash
npm run build
npm run export   # emits static site to frontend/out
```
Place `frontend/out` alongside the backend; FastAPI will serve it at `/ui/` automatically when `frontend/out` exists.
