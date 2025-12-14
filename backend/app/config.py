from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # repository root
DATA_DIR = Path(__file__).resolve().parent / "data"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
SESSIONS_FILE = DATA_DIR / "sessions.json"
USERS_FILE = DATA_DIR / "users.json"
DEFAULT_INDEX_PATH = ROOT_DIR / "vector_store/faiss_sentences/index.faiss"
DEFAULT_META_PATH = ROOT_DIR / "vector_store/faiss_sentences/metadata.jsonl"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
DEFAULT_LLM_BACKEND = os.getenv("LLM_BACKEND", "azure")
FRONTEND_DIR = ROOT_DIR / "frontend" / "out"
HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "6"))
