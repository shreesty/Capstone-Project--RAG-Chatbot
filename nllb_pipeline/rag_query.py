from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


def load_metadata(meta_path: Path) -> Dict[int, Dict]:
    """Load id -> record mapping from metadata JSONL."""
    mapping: Dict[int, Dict] = {}
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed JSON line in %s", meta_path)
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


def pick_query(args: argparse.Namespace) -> str:
    if args.query:
        return args.query.strip()
    if args.query_file:
        return args.query_file.read_text(encoding="utf-8").strip()
    raise SystemExit("Provide --query or --query-file")


def build_context(records: List[Dict]) -> str:
    lines: List[str] = []
    for i, rec in enumerate(records, start=1):
        text = rec.get("text", "<missing text>")
        meta = rec.get("metadata", {})
        source = meta.get("source_url") or meta.get("source_path") or meta.get("source_domain") or "unknown"
        lines.append(f"[{i}] {text}\nsource: {source}")
    return "\n\n".join(lines)


def format_hit(idx: int, score: float, rec: Dict) -> str:
    text = rec.get("text", "<missing text>")
    meta = rec.get("metadata", {})
    return f"id={idx} score={score:.4f} source={meta.get('source_url') or meta.get('source_path') or 'n/a'}\n{text}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end RAG query using FAISS retrieval + Llama (Groq).")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("vector_store/faiss_sentences/index.faiss"),
        help="Path to FAISS index file.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("vector_store/faiss_sentences/metadata.jsonl"),
        help="Path to metadata JSONL aligned with the index entries.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name used to embed queries.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbors to return.")
    parser.add_argument("--query", type=str, help="Text query to search for.")
    parser.add_argument(
        "--query-file",
        type=Path,
        help="Read the query from a text file (useful for long prompts).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="Groq Llama model name (e.g., llama-3.3-70b-versatile, llama-3.3-8b-instant).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature for generation.",
    )
    parser.add_argument(
        "--print-context",
        action="store_true",
        help="Print retrieved contexts before the LLM answer.",
    )
    return parser.parse_args()


def answer_with_llm(query: str, contexts: List[Dict], model_name: str, temperature: float) -> str:
    system_prompt = (
        "You are a retrieval QA assistant. Use only the provided context snippets to answer the question. "
        "If the context is insufficient, say you do not have enough information. Be concise."
    )
    context_block = build_context(contexts)
    user_prompt = f"Question: {query}\n\nContext:\n{context_block}\n\nAnswer:"

    llm = ChatGroq(model_name=model_name, temperature=temperature)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # surfacing deprecation/availability errors
        suggestion = (
            "Groq model may be deprecated. Try --model llama-3.3-70b-versatile or --model llama-3.3-8b-instant."
        )
        raise SystemExit(f"LLM call failed ({model_name}): {exc}\n{suggestion}") from exc
    return response.content.strip()


def run() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    query_text = pick_query(args)
    logging.info("Loading index from %s", args.index)
    index = faiss.read_index(str(args.index))

    logging.info("Loading metadata from %s", args.metadata)
    metadata = load_metadata(args.metadata)

    model = SentenceTransformer(args.embedding_model)
    query_vec = embed_query(model, query_text)

    k = min(args.top_k, index.ntotal)
    scores, ids = index.search(query_vec, k)

    retrieved: List[Dict] = []
    for idx, score in zip(ids[0], scores[0]):
        rec = metadata.get(int(idx), {"id": int(idx)})
        rec["score"] = float(score)
        retrieved.append(rec)

    if args.print_context:
        print("\nRetrieved contexts:\n")
        for rec in retrieved:
            print(format_hit(rec.get("id", -1), rec.get("score", 0.0), rec))
            print()

    answer = answer_with_llm(query_text, retrieved, args.model, args.temperature)

    print(f"Query: {query_text}\n")
    print("Answer:\n" + answer)


if __name__ == "__main__":
    run()
