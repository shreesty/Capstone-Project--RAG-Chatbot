from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run similarity search over saved FAISS sentence store.")
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
    return parser.parse_args()


def pick_query(args: argparse.Namespace) -> str:
    if args.query:
        return args.query.strip()
    if args.query_file:
        return args.query_file.read_text(encoding="utf-8").strip()
    raise SystemExit("Provide --query or --query-file")


def format_result(rank: int, score: float, record: Dict) -> str:
    text = record.get("text", "<missing text>")
    meta = record.get("metadata", {})
    lines = [f"{rank}. score={score:.4f} id={record.get('id')}", f"   text: {text}"]
    if meta:
        lines.append(f"   meta: {json.dumps(meta, ensure_ascii=False)}")
    return "\n".join(lines)


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

    print(f"\nQuery: {query_text}\n")
    print(f"Top {k} results:\n")
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        rec = metadata.get(int(idx), {"id": int(idx)})
        print(format_result(rank, float(score), rec))


if __name__ == "__main__":
    run()
