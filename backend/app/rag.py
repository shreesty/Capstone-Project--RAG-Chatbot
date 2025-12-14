from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict

import faiss
from sentence_transformers import SentenceTransformer

from .config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDEX_PATH,
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MODEL,
    DEFAULT_META_PATH,
)

logger = logging.getLogger(__name__)


def load_metadata(meta_path: Path) -> Dict[int, Dict]:
    mapping: Dict[int, Dict] = {}
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line in %s", meta_path)
                continue
            idx = rec.get("id")
            if idx is None:
                continue
            mapping[int(idx)] = rec
    return mapping


def load_resources(app) -> None:
    index_path = Path(os.getenv("FAISS_INDEX_PATH", DEFAULT_INDEX_PATH))
    meta_path = Path(os.getenv("FAISS_METADATA_PATH", DEFAULT_META_PATH))
    embed_model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
    llm_backend = os.getenv("LLM_BACKEND", DEFAULT_LLM_BACKEND).lower()
    app.state.llm_backend = llm_backend

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    logger.info("Loading FAISS index from %s", index_path)
    index = faiss.read_index(str(index_path))

    logger.info("Loading metadata from %s", meta_path)
    metadata = load_metadata(meta_path)

    logger.info("Loading embedding model %s", embed_model_name)
    embed_model = SentenceTransformer(embed_model_name)

    app.state.index_path = str(index_path)
    app.state.metadata_path = str(meta_path)
    app.state.index = index
    app.state.metadata = metadata
    app.state.embed_model = embed_model
    app.state.embedding_model_name = embed_model_name
    app.state.default_llm_model = os.getenv(
        "DEFAULT_LLM_MODEL",
        DEFAULT_LLM_MODEL if llm_backend == "groq" else os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
    )
    logger.info("Model: %s", app.state.default_llm_model)
