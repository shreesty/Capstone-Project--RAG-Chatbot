from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
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


class AzureLLM:
    """Simple wrapper to configure Azure OpenAI chat."""

    def __init__(
        self,
        deployment: str,
        endpoint: str,
        api_key: str,
        api_version: str = "2024-08-01-preview",
        temperature: float = 0.2,
    ) -> None:
        self.deployment = deployment
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.temperature = temperature
        self.client = AzureChatOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            deployment_name=self.deployment,
            temperature=self.temperature,
        )

    def invoke(self, messages):
        return self.client.invoke(messages)


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
        "--llm-backend",
        type=str,
        choices=["groq", "azure"],
        default="groq",
        help="LLM provider (groq or azure).",
    )
    parser.add_argument(
        "--azure-deployment",
        type=str,
        default=None,
        help="Azure OpenAI deployment name (falls back to AZURE_OPENAI_DEPLOYMENT).",
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


def build_llm(args: argparse.Namespace) -> Tuple[object, str]:
    if args.llm_backend == "azure":
        deployment = args.azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        missing = [name for name, val in (
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
        ) if not val]
        if missing:
            raise SystemExit(f"Missing Azure config: {', '.join(missing)}")
        llm = AzureLLM(
            deployment=deployment,
            endpoint=endpoint,  # type: ignore[arg-type]
            api_key=api_key,  # type: ignore[arg-type]
            api_version=api_version,
            temperature=args.temperature,
        )
        return llm, deployment or "azure"

    llm = ChatGroq(model_name=args.model, temperature=args.temperature)
    return llm, args.model


def answer_with_llm(query: str, contexts: List[Dict], llm: object, model_label: str) -> str:
    system_prompt = (
        "You are a retrieval QA assistant. Use only the provided context snippets to answer the question. "
        "If the context is insufficient, say you do not have enough information. Be concise."
    )
    context_block = build_context(contexts)
    user_prompt = f"Question: {query}\n\nContext:\n{context_block}\n\nAnswer:"

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # surfacing deprecation/availability errors
        suggestion = (
            "LLM call failed. For Groq, try --model llama-3.3-70b-versatile or llama-3.3-8b-instant. "
            "For Azure, verify deployment/endpoint/API key and version."
        )
        raise SystemExit(f"LLM call failed ({model_label}): {exc}\n{suggestion}") from exc
    return response.content.strip()


def run() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    llm, model_label = build_llm(args)

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

    answer = answer_with_llm(query_text, retrieved, llm, model_label)

    print(f"Query: {query_text}\n")
    print("Answer:\n" + answer)


if __name__ == "__main__":
    run()
