from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm


@dataclass
class SentenceRecord:
    uid: str
    doc_id: str
    sentence_index: int
    source_path: str
    source_domain: str
    source_url: Optional[str]
    kind: str
    doc_language: str
    sentence: str


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[\.\?\!।])\s+")


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    parts = SENTENCE_BOUNDARY_RE.split(normalized)
    return [p.strip() for p in parts if p.strip()]


def iter_sentences(
    input_path: Path, min_chars: int
) -> Iterable[SentenceRecord]:
    with input_path.open(encoding="utf-8") as f:
        for doc_idx, line in enumerate(f):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping bad JSON at line %d", doc_idx + 1)
                continue

            text = doc.get("text", "") or ""
            sentences = split_sentences(text)
            doc_id = doc.get("source_path", f"doc_{doc_idx}")

            for sent_idx, sentence in enumerate(sentences):
                if len(sentence) < min_chars:
                    continue
                yield SentenceRecord(
                    uid=f"{doc_id}#s{sent_idx}",
                    doc_id=doc_id,
                    sentence_index=sent_idx,
                    source_path=doc.get("source_path", ""),
                    source_domain=doc.get("source_domain", ""),
                    source_url=doc.get("source_url"),
                    kind=doc.get("kind", ""),
                    doc_language=doc.get("language", ""),
                    sentence=sentence,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split extracted documents into sentence-level chunks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("processed_data/extracted_documents.jsonl"),
        help="Input JSONL from extract.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed_data/sentences.jsonl"),
        help="Where to write sentence JSONL.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=30,
        help="Drop sentences shorter than this many characters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only emit this many sentences (debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w", encoding="utf-8") as out_f:
        for sentence in tqdm(iter_sentences(args.input, args.min_chars), desc="Splitting"):
            if args.limit is not None and written >= args.limit:
                break
            out_f.write(json.dumps(asdict(sentence), ensure_ascii=False) + "\n")
            written += 1

    logging.info(
        "Saved %d sentences to %s (min_chars=%d, limit=%s)",
        written,
        args.output,
        args.min_chars,
        args.limit,
    )


if __name__ == "__main__":
    main()
