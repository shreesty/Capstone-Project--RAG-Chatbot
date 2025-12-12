from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def iter_translated_sentences(path: Path) -> Iterable[Dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed JSON line in %s", path)
                continue


def pick_text(rec: Dict, prefer_translation: bool = True) -> str:
    if prefer_translation:
        translated = rec.get("translation")
        if translated:
            return translated
    return rec.get("sentence", "") or ""


def build_embeddings(
    texts: List[str], model_name: str, batch_size: int = 64
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    faiss.normalize_L2(embeddings)
    return embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed translated sentences into a FAISS index."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("processed_data/translated_sentences.jsonl"),
        help="Sentence-level translations JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vector_store/faiss_sentences"),
        help="Where to save FAISS index and metadata.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only embed this many sentences (debugging).",
    )
    parser.add_argument(
        "--no-translation",
        action="store_true",
        help="Embed the original sentence text instead of translation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    texts: List[str] = []
    metadatas: List[Dict] = []

    for idx, rec in enumerate(tqdm(iter_translated_sentences(args.input), desc="Loading")):
        if args.limit is not None and idx >= args.limit:
            break
        text = pick_text(rec, prefer_translation=not args.no_translation).strip()
        if not text:
            continue
        texts.append(text)
        metadatas.append(
            {
                "uid": rec.get("uid"),
                "doc_id": rec.get("doc_id"),
                "sentence_index": rec.get("sentence_index"),
                "source_path": rec.get("source_path"),
                "source_domain": rec.get("source_domain"),
                "source_url": rec.get("source_url"),
                "kind": rec.get("kind"),
                "doc_language": rec.get("doc_language"),
                "translated": rec.get("translated"),
                "translator_model": rec.get("translator_model"),
            }
        )

    if not texts:
        logging.error("No sentences loaded from %s", args.input)
        return

    logging.info("Encoding %d sentences with %s", len(texts), args.embedding_model)
    embeddings = build_embeddings(texts, args.embedding_model, batch_size=args.batch_size)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = args.output_dir / "index.faiss"
    meta_path = args.output_dir / "metadata.jsonl"
    faiss.write_index(index, str(index_path))

    with meta_path.open("w", encoding="utf-8") as f:
        for i, meta in enumerate(metadatas):
            record = {"id": i, "text": texts[i], "metadata": meta}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info(
        "Saved %d embeddings (dim=%d) to %s and metadata to %s",
        len(texts),
        dim,
        index_path,
        meta_path,
    )


if __name__ == "__main__":
    main()
