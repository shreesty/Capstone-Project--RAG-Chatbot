from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


def load_metadata(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def normalize(vecs: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vecs)
    return vecs


def search(
    index: faiss.Index, query_emb: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    scores, indices = index.search(query_emb, k)
    return scores[0], indices[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run similarity search against the FAISS index."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("vector_store/faiss/index.faiss"),
        help="Path to the FAISS index file.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("vector_store/faiss/metadata.jsonl"),
        help="Path to the metadata JSONL written during embedding.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name (must match embedding build).",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query string to search for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.index.exists() or not args.metadata.exists():
        logging.error("Index or metadata file not found. Run embed.py first.")
        return

    logging.info("Loading index from %s", args.index)
    index = faiss.read_index(str(args.index))
    metadata = load_metadata(args.metadata)

    logging.info("Encoding query with %s", args.embedding_model)
    model = SentenceTransformer(args.embedding_model)
    query_emb = model.encode([args.query], convert_to_numpy=True)
    query_emb = normalize(query_emb)

    scores, indices = search(index, query_emb, args.top_k)

    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        record = metadata[idx]
        text_preview = record.get("text", "")[:240].replace("\n", " ")
        meta = record.get("metadata", {})
        print(
            f"[{rank}] score={score:.4f} id={record.get('id')} "
            f"source={meta.get('source_domain')} path={meta.get('source_path')}"
        )
        print(f"     chunk#{meta.get('chunk_index')} doc#{meta.get('doc_index')}")
        print(f"     text: {text_preview}")


if __name__ == "__main__":
    main()
