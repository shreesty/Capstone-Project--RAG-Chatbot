from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()


def load_processed_documents(path: Path) -> List[Dict]:
    docs: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def chunk_text_for_embeddings(
    text: str, chunk_size: int = 800, chunk_overlap: int = 200
) -> List[str]:
    """Simple character-based chunking with overlap."""
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == length:
            break
        start = end - chunk_overlap
    return [c for c in chunks if c]


def build_embeddings(
    texts: List[str], model_name: str, batch_size: int = 64
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )
    # Normalize for cosine similarity with inner product index.
    faiss.normalize_L2(embeddings)
    return embeddings


def save_faiss(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk processed documents, embed, and store in FAISS."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("processed_data/english_documents.jsonl"),
        help="Path to the processed JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vector_store/faiss"),
        help="Directory to store FAISS index and metadata.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Character overlap between chunks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("Loading processed documents from %s", args.input)
    processed_docs = load_processed_documents(args.input)
    if not processed_docs:
        logging.error("No documents found in %s", args.input)
        return

    texts: List[str] = []
    metadatas: List[Dict] = []

    for doc_idx, doc in enumerate(processed_docs):
        base_meta = {
            "source_path": doc.get("source_path"),
            "source_domain": doc.get("source_domain"),
            "language": doc.get("language"),
            "translated": doc.get("translated"),
            "translation_source": doc.get("translation_source"),
        }
        for chunk_idx, chunk in enumerate(
            chunk_text_for_embeddings(
                doc.get("text", ""), args.chunk_size, args.chunk_overlap
            )
        ):
            texts.append(chunk)
            metadatas.append(
                {
                    **base_meta,
                    "chunk_index": chunk_idx,
                    "doc_index": doc_idx,
                }
            )

    logging.info("Prepared %d chunks from %d documents", len(texts), len(processed_docs))

    embeddings = build_embeddings(texts, args.embedding_model, batch_size=args.batch_size)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = args.output_dir / "index.faiss"
    meta_path = args.output_dir / "metadata.jsonl"
    save_faiss(index, index_path)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        for i, metadata in enumerate(metadatas):
            record = {"id": i, "text": texts[i], "metadata": metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info(
        "Stored %d embeddings (dim=%d) to %s and metadata to %s",
        len(texts),
        dim,
        index_path,
        meta_path,
    )


if __name__ == "__main__":
    main()
