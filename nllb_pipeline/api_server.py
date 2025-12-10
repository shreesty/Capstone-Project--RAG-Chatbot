from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path("vector_store/faiss_sentences/index.faiss")
DEFAULT_META_PATH = Path("vector_store/faiss_sentences/metadata.jsonl")
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
DEFAULT_LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")

app = FastAPI(
    title="RAG API",
    description="FAISS retrieval + Groq Llama answer generation.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question.")
    top_k: int = Field(5, ge=1, le=50, description="Number of neighbors to retrieve.")
    model: str = Field(DEFAULT_LLM_MODEL, description="Groq Llama model name.")
    backend: str | None = Field(
        default=None, description="LLM backend to use: groq or azure (falls back to env LLM_BACKEND)."
    )
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="LLM temperature.")
    return_contexts: bool = Field(
        False, description="Return retrieved contexts alongside the answer."
    )


class Hit(BaseModel):
    id: int
    score: float
    text: str
    metadata: Dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    model: str
    top_k: int
    answer: str
    hits: List[Hit]
    index_path: str
    metadata_path: str
    embedding_model: str
    contexts: List[str] | None = None


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


def embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    vector = model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(vector)
    return vector.astype(np.float32)


def build_context(records: List[Dict]) -> str:
    lines: List[str] = []
    for i, rec in enumerate(records, start=1):
        text = rec.get("text", "<missing text>")
        meta = rec.get("metadata", {})
        source = meta.get("source_url") or meta.get("source_path") or meta.get("source_domain") or "unknown"
        lines.append(f"[{i}] {text}\nsource: {source}")
    return "\n\n".join(lines)


def answer_with_llm(query: str, contexts: List[Dict], model_name: str, temperature: float) -> str:
    system_prompt = (
        "You are a retrieval QA assistant. Use only the provided context snippets to answer the question. "
        "If the context is insufficient, say you do not have enough information. Be concise."
    )
    context_block = build_context(contexts)
    user_prompt = f"Question: {query}\n\nContext:\n{context_block}\n\nAnswer:"

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    llm = get_llm(model_name=model_name, temperature=temperature)
    try:
        response = llm.invoke(messages)
    except HTTPException:
        raise
    except Exception as exc:
        suggestion = (
            "LLM call failed. For Groq, try llama-3.3-70b-versatile or llama-3.3-8b-instant. "
            "For Azure, check deployment/endpoint/api key."
        )
        raise HTTPException(status_code=502, detail=f"LLM call failed ({model_name}): {exc}\n{suggestion}")
    return response.content.strip()


def azure_llm_config(requested_model: str | None = None) -> Tuple[str, str, str, str]:
    deployment = requested_model or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    missing = [name for name, val in (
        ("AZURE_OPENAI_DEPLOYMENT", deployment),
        ("AZURE_OPENAI_ENDPOINT", endpoint),
        ("AZURE_OPENAI_API_KEY", api_key),
    ) if not val]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing Azure config: {', '.join(missing)}")
    return deployment, endpoint, api_key, api_version  # type: ignore[return-value]


def get_llm(model_name: str, temperature: float):
    backend = getattr(app.state, "llm_backend", DEFAULT_LLM_BACKEND)
    if backend == "azure":
        deployment, endpoint, api_key, api_version = azure_llm_config(model_name)
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment_name=deployment,
            temperature=temperature,
        )

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set in the environment.")
    return ChatGroq(model_name=model_name, temperature=temperature)


def load_resources(app: FastAPI) -> None:
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


@app.on_event("startup")
def on_startup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_resources(app)


@app.get("/health")
def health() -> Dict[str, object]:
    index = getattr(app.state, "index", None)
    metadata = getattr(app.state, "metadata", None)
    return {
        "status": "ok",
        "index_loaded": index is not None,
        "index_size": int(index.ntotal) if index is not None else 0,
        "metadata_loaded": metadata is not None,
        "metadata_records": len(metadata) if metadata is not None else 0,
        "embedding_model": getattr(app.state, "embedding_model_name", DEFAULT_EMBED_MODEL),
        "index_path": getattr(app.state, "index_path", str(DEFAULT_INDEX_PATH)),
        "metadata_path": getattr(app.state, "metadata_path", str(DEFAULT_META_PATH)),
        "llm_backend": getattr(app.state, "llm_backend", DEFAULT_LLM_BACKEND),
        "default_llm_model": getattr(app.state, "default_llm_model", DEFAULT_LLM_MODEL),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    index = getattr(app.state, "index", None)
    metadata = getattr(app.state, "metadata", None)
    embed_model = getattr(app.state, "embed_model", None)
    llm_backend = (request.backend or getattr(app.state, "llm_backend", DEFAULT_LLM_BACKEND)).lower()
    app.state.llm_backend = llm_backend

    if index is None or metadata is None or embed_model is None:
        raise HTTPException(status_code=500, detail="Resources not loaded.")

    if index.ntotal == 0:
        raise HTTPException(status_code=404, detail="Index is empty.")

    query_vec = embed_query(embed_model, request.query)
    k = min(request.top_k, index.ntotal)
    scores, ids = index.search(query_vec, k)

    retrieved: List[Dict] = []
    for idx, score in zip(ids[0], scores[0]):
        rec = metadata.get(int(idx), {"id": int(idx)})
        rec = rec.copy()
        rec["score"] = float(score)
        retrieved.append(rec)

    try:
        answer = answer_with_llm(request.query, retrieved, request.model, request.temperature)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    contexts = None
    if request.return_contexts:
        contexts = [r.get("text", "") for r in retrieved]

    hits = [
        Hit(
            id=int(r.get("id", -1)),
            score=float(r.get("score", 0.0)),
            text=r.get("text", ""),
            metadata=r.get("metadata", {}),
        )
        for r in retrieved
    ]

    return QueryResponse(
        query=request.query,
        model=request.model,
        top_k=len(hits),
        answer=answer,
        hits=hits,
        index_path=app.state.index_path,
        metadata_path=app.state.metadata_path,
        embedding_model=app.state.embedding_model_name,
        contexts=contexts,
    )
