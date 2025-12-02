from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()


def load_metadata(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def normalize(vecs: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vecs)
    return vecs


def retrieve(
    index: faiss.Index, embedder: SentenceTransformer, query: str, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    query_vec = embedder.encode([query], convert_to_numpy=True)
    query_vec = normalize(query_vec)
    scores, indices = index.search(query_vec, k)
    return scores[0], indices[0]


def build_llm(provider: str, model: str):
    if provider == "groq":
        return ChatGroq(model=model, temperature=0)
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=0)
    raise ValueError(f"Unsupported provider: {provider}")


def format_context(metadata_items: List[dict], indices: np.ndarray) -> str:
    parts: List[str] = []
    for rank, idx in enumerate(indices, start=1):
        if idx < 0 or idx >= len(metadata_items):
            continue
        rec = metadata_items[idx]
        meta = rec.get("metadata", {})
        text = rec.get("text", "").strip()
        parts.append(
            f"[{rank}] source_domain={meta.get('source_domain')} "
            f"path={meta.get('source_path')} chunk={meta.get('chunk_index')} "
            f"doc={meta.get('doc_index')}\n{text}"
        )
    return "\n\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query FAISS index with LLM answering.")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("vector_store/faiss/index.faiss"),
        help="Path to FAISS index.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("vector_store/faiss/metadata.jsonl"),
        help="Path to metadata JSONL.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model (must match embed step).",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["groq", "openai"],
        default="groq",
        help="LLM provider for answers.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="LLM model name (provider-specific).",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User question to answer.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of passages to retrieve.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.index.exists() or not args.metadata.exists():
        logging.error("Index or metadata not found; run embed.py first.")
        return

    logging.info("Loading FAISS index from %s", args.index)
    index = faiss.read_index(str(args.index))
    metadata_items = load_metadata(args.metadata)

    logging.info("Loading embedder %s", args.embedding_model)
    embedder = SentenceTransformer(args.embedding_model)

    logging.info("Retrieving top-%d passages", args.top_k)
    scores, indices = retrieve(index, embedder, args.query, args.top_k)

    context = format_context(metadata_items, indices)
    llm = build_llm(args.llm_provider, args.llm_model)

    prompt = (
        "You are a helpful assistant answering questions using the provided context. "
        "Use only the context to answer; if the context is insufficient, say so briefly.\n\n"
        f"Context:\n{context}\n\nQuestion: {args.query}\nAnswer:"
    )

    response = llm.invoke(prompt)
    print("\n=== Answer ===")
    print(response.content.strip())

    print("\n=== Top matches ===")
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        if idx < 0 or idx >= len(metadata_items):
            continue
        rec = metadata_items[idx]
        meta = rec.get("metadata", {})
        preview = rec.get("text", "")[:200].replace("\n", " ")
        print(
            f"[{rank}] score={score:.4f} "
            f"source={meta.get('source_domain')} path={meta.get('source_path')} "
            f"chunk#{meta.get('chunk_index')} doc#{meta.get('doc_index')}"
        )
        print(f"      {preview}")


if __name__ == "__main__":
    main()
