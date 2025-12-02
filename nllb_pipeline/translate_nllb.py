from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm


@dataclass
class TranslatedSentence:
    uid: str
    doc_id: str
    sentence_index: int
    source_path: str
    source_domain: str
    source_url: Optional[str]
    kind: str
    doc_language: str
    sentence: str
    translation: str
    translated: bool
    translator_model: Optional[str]
    source_lang_code: str
    target_lang_code: str


def auto_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def load_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model


def forced_bos_id(tokenizer, target_lang_code: str) -> int:
    if hasattr(tokenizer, "lang_code_to_id"):
        return tokenizer.lang_code_to_id[target_lang_code]
    return tokenizer.convert_tokens_to_ids(target_lang_code)


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_sentences(input_path: Path) -> Iterable[dict]:
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate sentence JSONL (Nepali -> English) using NLLB."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("processed_data/sentences.jsonl"),
        help="Sentence-level JSONL input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed_data/translated_sentences.jsonl"),
        help="Where to write translated JSONL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--source-lang-filter",
        type=str,
        default="ne",
        help="Only translate sentences whose doc_language matches this (langdetect code).",
    )
    parser.add_argument(
        "--src-lang-code",
        type=str,
        default="npi_Deva",
        help="NLLB source language code for Nepali.",
    )
    parser.add_argument(
        "--tgt-lang-code",
        type=str,
        default="eng_Latn",
        help="NLLB target language code.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto|cpu|cuda|mps.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=512,
        help="Max input tokens for generation (truncation).",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=512,
        help="Max output tokens for generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Translate only this many sentences (debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = auto_device(args.device)
    tokenizer, model = load_model(args.model, device)
    bos_id = forced_bos_id(tokenizer, args.tgt_lang_code)

    total_written = 0
    with args.output.open("w", encoding="utf-8") as out_f:
        iterator = iter_sentences(args.input)
        for batch in tqdm(batched(iterator, args.batch_size), desc="Translating"):
            if args.limit is not None and total_written >= args.limit:
                break

            to_translate = []
            passthrough = []

            for rec in batch:
                lang = rec.get("doc_language", "")
                if args.source_lang_filter and lang != args.source_lang_filter:
                    passthrough.append(rec)
                else:
                    to_translate.append(rec)

            translations: list[str] = []
            if to_translate:
                texts = [r.get("sentence", "") for r in to_translate]
                tokenizer.src_lang = args.src_lang_code
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_length,
                ).to(device)

                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=bos_id,
                    max_length=args.max_output_length,
                )
                translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # emit translated records
            for rec, translation in zip(to_translate, translations):
                translated_rec = TranslatedSentence(
                    uid=rec.get("uid", ""),
                    doc_id=rec.get("doc_id", ""),
                    sentence_index=rec.get("sentence_index", 0),
                    source_path=rec.get("source_path", ""),
                    source_domain=rec.get("source_domain", ""),
                    source_url=rec.get("source_url"),
                    kind=rec.get("kind", ""),
                    doc_language=rec.get("doc_language", ""),
                    sentence=rec.get("sentence", ""),
                    translation=translation,
                    translated=True,
                    translator_model=args.model,
                    source_lang_code=args.src_lang_code,
                    target_lang_code=args.tgt_lang_code,
                )
                out_f.write(json.dumps(asdict(translated_rec), ensure_ascii=False) + "\n")
                total_written += 1
                if args.limit is not None and total_written >= args.limit:
                    break

            if args.limit is not None and total_written >= args.limit:
                break

            # emit passthrough records unchanged
            for rec in passthrough:
                translated_rec = TranslatedSentence(
                    uid=rec.get("uid", ""),
                    doc_id=rec.get("doc_id", ""),
                    sentence_index=rec.get("sentence_index", 0),
                    source_path=rec.get("source_path", ""),
                    source_domain=rec.get("source_domain", ""),
                    source_url=rec.get("source_url"),
                    kind=rec.get("kind", ""),
                    doc_language=rec.get("doc_language", ""),
                    sentence=rec.get("sentence", ""),
                    translation=rec.get("sentence", ""),
                    translated=False,
                    translator_model=None,
                    source_lang_code=args.src_lang_code,
                    target_lang_code=args.tgt_lang_code,
                )
                out_f.write(json.dumps(asdict(translated_rec), ensure_ascii=False) + "\n")
                total_written += 1
                if args.limit is not None and total_written >= args.limit:
                    break

    logging.info(
        "Wrote %d sentences to %s (model=%s, device=%s)",
        total_written,
        args.output,
        args.model,
        device,
    )


if __name__ == "__main__":
    main()
