from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from tqdm import tqdm


@dataclass
class SentenceRecord:
    uid: str
    doc_id: str
    sentence_index: int  # chunk index within doc
    source_path: str
    source_domain: str
    source_url: Optional[str]
    kind: str
    doc_language: str
    sentence: str  # chunk text


def simple_recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
) -> List[str]:
    """Minimal recursive splitter that prefers earlier separators and falls back to sliding window."""

    def _split(t: str, seps: List[str]) -> List[str]:
        if len(t) <= chunk_size:
            return [t]
        if not seps:
            # fallback: whitespace sliding window
            words = t.split()
            chunks: List[str] = []
            current: List[str] = []
            current_len = 0
            for w in words:
                wlen = len(w) + 1  # include space
                if current_len + wlen > chunk_size and current:
                    chunk = " ".join(current).strip()
                    chunks.append(chunk)
                    # overlap by reusing tail
                    if chunk_overlap > 0:
                        overlap_words = []
                        total = 0
                        for word in reversed(current):
                            total += len(word) + 1
                            overlap_words.append(word)
                            if total >= chunk_overlap:
                                break
                        current = list(reversed(overlap_words))
                        current_len = sum(len(w) + 1 for w in current)
                    else:
                        current = []
                        current_len = 0
                current.append(w)
                current_len += wlen
            if current:
                chunks.append(" ".join(current).strip())
            return chunks

        sep = seps[0]
        pieces = t.split(sep)
        if len(pieces) == 1:
            return _split(t, seps[1:])

        recombined = []
        for i, piece in enumerate(pieces):
            if not piece:
                continue
            if i < len(pieces) - 1:
                recombined.append(piece + sep)
            else:
                recombined.append(piece)

        out: List[str] = []
        for piece in recombined:
            if len(piece) <= chunk_size:
                out.append(piece.strip())
            else:
                out.extend(_split(piece, seps[1:]))
        return out

    # merge with overlap
    base_chunks = [c.strip() for c in _split(text, separators) if c.strip()]
    if not base_chunks:
        return []

    SENT_ENDINGS = {".", "?", "!"}

    def build_overlap(prev: str, nxt: str) -> str:
        if chunk_overlap <= 0:
            return nxt
        start = max(len(prev) - chunk_overlap, 0)
        tail = prev[start:]
        # try to start overlap at the last sentence boundary before the overlap window
        last_boundary = max(prev.rfind(e, 0, start) for e in SENT_ENDINGS)
        if last_boundary != -1:
            start = last_boundary + 1
            tail = prev[start:].lstrip()
        # if tail ends mid-sentence, extend into next chunk until boundary
        next_boundary = None
        for i, ch in enumerate(nxt):
            if ch in SENT_ENDINGS:
                next_boundary = i
                break
        if next_boundary is not None and tail and tail[-1] not in SENT_ENDINGS:
            tail = (tail + " " + nxt[: next_boundary + 1]).strip()
        return (tail + " " + nxt).strip()

    merged: List[str] = []
    buffer = ""
    for chunk in base_chunks:
        if not buffer:
            buffer = chunk
            continue
        if len(buffer) + 1 + len(chunk) <= chunk_size:
            buffer = (buffer + " " + chunk).strip()
        else:
            merged.append(buffer)
            buffer = build_overlap(buffer, chunk)
    if buffer:
        merged.append(buffer.strip())
    return merged


def iter_sentences(
    input_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
    min_chars: int,
) -> Iterable[SentenceRecord]:
    with input_path.open(encoding="utf-8") as f:
        for doc_idx, line in enumerate(f):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping bad JSON at line %d", doc_idx + 1)
                continue

            text = doc.get("text", "") or ""
            chunks: List[str] = simple_recursive_split(text, chunk_size, chunk_overlap, separators)
            doc_id = doc.get("source_path", f"doc_{doc_idx}")

            for sent_idx, sentence in enumerate(chunks):
                if len(sentence.strip()) < min_chars:
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
        description="Chunk extracted documents with overlap (sentence-biased separators)."
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
        help="Where to write chunked JSONL.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Drop chunks shorter than this many characters.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_100,
        help="Target chunk size (characters).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Character overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--separators",
        type=str,
        default=None,
        help="Comma-separated custom separators (optional). Defaults to a sentence-biased list.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only emit this many chunks (debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.separators:
        separators = [s for s in args.separators.split(",")]
    else:
        separators = [".", "?", "!", "\n\n", "\n", " "]

    written = 0
    with args.output.open("w", encoding="utf-8") as out_f:
        for sentence in tqdm(
            iter_sentences(args.input, args.chunk_size, args.chunk_overlap, separators, args.min_chars),
            desc="Splitting",
        ):
            if args.limit is not None and written >= args.limit:
                break
            out_f.write(json.dumps(asdict(sentence), ensure_ascii=False) + "\n")
            written += 1

    logging.info(
        "Saved %d chunks to %s (min_chars=%d, chunk_size=%d, overlap=%d, limit=%s)",
        written,
        args.output,
        args.min_chars,
        args.chunk_size,
        args.chunk_overlap,
        args.limit,
    )


if __name__ == "__main__":
    main()
