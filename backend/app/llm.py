from __future__ import annotations

import os
from typing import Dict, List, Tuple

import faiss
import numpy as np
from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_EMBED_MODEL, DEFAULT_LLM_BACKEND


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


def answer_with_llm(
    query: str,
    contexts: List[Dict],
    model_name: str,
    temperature: float,
    backend: str,
    history: List[Dict[str, str]] | None = None,
) -> str:
    system_prompt = (
        "You are a retrieval QA assistant. Use only the provided context snippets to answer the question. "
        "If the context is insufficient, say you do not have enough information. Be concise."
    )
    context_block = build_context(contexts)
    user_prompt = f"Question: {query}\n\nContext:\n{context_block}\n\nAnswer:"

    messages = [SystemMessage(content=system_prompt)]
    if history:
        from langchain_core.messages import AIMessage

        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_prompt))
    llm = get_llm(model_name=model_name, temperature=temperature, backend=backend)
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
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", requested_model)
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


def get_llm(model_name: str, temperature: float, backend: str):
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
