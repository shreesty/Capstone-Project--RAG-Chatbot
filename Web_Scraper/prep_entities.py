#!/usr/bin/env python3
"""
Preprocess your scraped folder tree into ONE FILE PER UNIVERSITY / COLLEGE.

Input tree (example):
Web_Scraper/scraped_data/
  ├── afu.edu.np/
  │   ├── text/*.txt
  │   ├── docs/*.pdf
  │   ├── html/*.html (ignored by default; use text/*.txt instead)
  │   ├── images/*  (optionally OCR later)
  │   └── metadata.jsonl (optional)
  ├── bagmatiuniversity.edu.np/
  └── ...

What this script does:
  • Walks every domain folder under --in-root (default: Web_Scraper/scraped_data)
  • Reads all TXT files, extracts text from PDFs (best‑effort via pdfplumber)
  • Cleans text (Unicode NFC, zero‑width removal, fix ligatures, normalize Nepali digits)
  • Language detection; optional Nepali → English translation (Argos Translate)
  • Concatenates into one JSON (and/or Markdown) per entity in --out (default: ./processed_tree)

Install deps:
    pip install langdetect ftfy pdfplumber
    # Optional for offline translation (Nepali→English):
    pip install argostranslate

Examples:
    # Produce JSON only, no translation
    python prep_from_tree.py --in-root "Web_Scraper/scraped_data" --out processed_tree --format json

    # JSON + Markdown, with Argos translation and 20k char cap per blob
    python prep_from_tree.py --format both --translate-provider argos --translate-char-limit 20000

Notes:
    • PDF extraction is best‑effort; structured tables are not flattened beyond plaintext here.
    • For image OCR, integrate tesseract/pytesseract later.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import ftfy
from langdetect import detect as ld_detect, DetectorFactory
DetectorFactory.seed = 42

import pdfplumber

# --------------------------- CLI --------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess folder tree into one file per university/college")
    p.add_argument("--in-root", type=Path, default=Path("Web_Scraper/scraped_data"), help="Root folder that contains one subfolder per domain")
    p.add_argument("--out", type=Path, default=Path("processed_tree"), help="Output folder")
    p.add_argument("--format", choices=["json", "md", "both"], default="json", help="Output type(s)")
    p.add_argument("--translate-provider", choices=["none", "argos"], default="none", help="Nepali→English translator")
    p.add_argument("--translate-char-limit", type=int, default=15000, help="Max characters to translate per blob")
    p.add_argument("--max-pdf-pages", type=int, default=50, help="Hard page cap per PDF to avoid heavy docs (0 = no cap)")
    p.add_argument("--min-paragraph-len", type=int, default=30, help="Drop tiny fragments below this length")
    return p

# --------------------------- Mappings --------------------------- #
# Map base domains to canonical entity names/types (extend as needed)
DOMAIN_MAP: Dict[str, Tuple[str, str]] = {
    "tu.edu.np": ("Tribhuvan University", "university"),
    "ku.edu.np": ("Kathmandu University", "university"),
    "purbanchaluniversity.edu.np": ("Purbanchal University", "university"),
    "pu.edu.np": ("Pokhara University", "university"),
    "afu.edu.np": ("Agriculture and Forestry University", "university"),
    "fwu.edu.np": ("Far Western University", "university"),
    "mwu.edu.np": ("Mid-Western University", "university"),
    "rju.edu.np": ("Rajshree Janak University", "university"),
    "uon.edu.np": ("University of Nepal", "university"),
    "nou.edu.np": ("Nepal Open University", "university"),
    "mbust.edu.np": ("Madan Bhandari University of Science and Technology", "university"),
    "nsu.edu.np": ("Nepal Sanskrit University", "university"),
    "lbu.edu.np": ("Lumbini Buddhist University", "university"),
    "mtu.edu.np": ("Manmohan Technical University", "university"),
    "gandakiuniversity.edu.np": ("Gandaki University", "university"),
    "mau.edu.np": ("Madhesh University", "university"),
    "ltu.edu.np": ("Lumbini Technical University", "university"),
    "bagmatiuniversity.edu.np": ("Bagmati University", "university"),
    "nams.edu.np": ("National Academy of Medical Sciences", "academy"),
    "bpkihs.edu": ("B. P. Koirala Institute of Health Sciences", "institute"),
    "pahs.edu.np": ("Patan Academy of Health Sciences", "academy"),
    "kahs.edu.np": ("Karnali Academy of Health Sciences", "academy"),
    "ctevt.org.np": ("Council for Technical Education and Vocational Training", "council"),
    "neb.gov.np": ("National Examinations Board", "board"),
    "moest.gov.np": ("Ministry of Education, Science and Technology", "ministry"),
    "ugcnepal.edu.np": ("University Grants Commission, Nepal", "commission"),
    "doe.gov.np": ("Department of Education", "department"),
}

# --------------------------- Text utilities --------------------------- #
NEPALI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
LATIN_LIGATURES = {"ﬀ":"ff","ﬁ":"fi","ﬂ":"fl","ﬃ":"ffi","ﬄ":"ffl","ﬅ":"ft","ﬆ":"st"}
ZERO_WIDTH = re.compile(r"[\u200B-\u200F\uFEFF]")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def normalize_basic(text: str) -> str:
    if not text:
        return text
    t = ftfy.fix_text(text)
    t = nfc(t)
    t = ZERO_WIDTH.sub("", t)
    for k, v in LATIN_LIGATURES.items():
        t = t.replace(k, v)
    t = t.translate(NEPALI_DIGITS)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def detect_lang(text: str) -> str:
    try:
        return ld_detect(text)
    except Exception:
        return "unknown"

# --------------------------- Translation --------------------------- #
class Translator:
    def translate(self, text: str) -> str:
        return text

class ArgosTranslator(Translator):
    def __init__(self) -> None:
        import argostranslate.package as pkg
        import argostranslate.translate as tr
        try:
            pkg.update_package_index()
            pkgs = [p for p in pkg.get_available_packages() if p.from_code=="ne" and p.to_code=="en"]
            if pkgs:
                pkg.install_from_path(pkgs[0].download())
        except Exception:
            pass
        self.tr = tr
    def translate(self, text: str) -> str:
        try:
            return self.tr.translate(text, "ne", "en")
        except Exception:
            return text

# --------------------------- Data model --------------------------- #
@dataclass
class Blob:
    source: str
    kind: str  # "txt" | "pdf"
    lang: str = "unknown"
    text_norm: Optional[str] = None
    text_en: Optional[str] = None

@dataclass
class EntityDoc:
    name: str
    domain: str
    entity_type: str
    files: List[str] = field(default_factory=list)
    blobs: List[Blob] = field(default_factory=list)

# --------------------------- Helpers --------------------------- #

def base_domain(name: str) -> str:
    # strip common prefixes like www., web.
    s = name.lower().strip()
    s = s.replace("https://", "").replace("http://", "").strip("/")
    if s.startswith("www."):
        s = s[4:]
    if s.startswith("web."):
        s = s[4:]
    return s


def guess_entity(domain: str) -> Tuple[str, str]:
    d = base_domain(domain)
    return DOMAIN_MAP.get(d, (d, "organization"))

# --------------------------- Loaders --------------------------- #

def read_text_file(p: Path) -> Optional[str]:
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
        return t if t.strip() else None
    except Exception:
        return None


def extract_pdf_text(pdf_path: Path, page_cap: int) -> Optional[str]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages if page_cap <= 0 else pdf.pages[:page_cap]
            chunks: List[str] = []
            for pg in pages:
                chunks.append(pg.extract_text() or "")
            txt = "\n\n".join(chunks)
            return txt if txt.strip() else None
    except Exception:
        return None

# --------------------------- Core --------------------------- #

def process_domain_folder(folder: Path, *, translator: Translator, char_limit: int, min_para_len: int, page_cap: int) -> Optional[EntityDoc]:
    if not folder.is_dir():
        return None
    domain = base_domain(folder.name)
    name, etype = guess_entity(domain)
    entity = EntityDoc(name=name, domain=domain, entity_type=etype)

    # TXT sources
    for p in sorted((folder / "text").glob("**/*.txt")):
        entity.files.append(str(p))
        raw = read_text_file(p)
        if not raw:
            continue
        norm = normalize_basic(raw)
        if len(norm) < min_para_len:
            continue
        lang = detect_lang(norm) if len(norm) > 20 else "unknown"
        blob = Blob(source=str(p), kind="txt", lang=lang, text_norm=norm)
        if lang.startswith("ne"):
            blob.text_en = translator.translate(norm[:char_limit])
        elif lang.startswith("en"):
            blob.text_en = norm
        entity.blobs.append(blob)

    # PDF sources
    for p in sorted((folder / "docs").glob("**/*.pdf")):
        entity.files.append(str(p))
        raw = extract_pdf_text(p, page_cap)
        if not raw:
            continue
        norm = normalize_basic(raw)
        if len(norm) < min_para_len:
            continue
        lang = detect_lang(norm) if len(norm) > 20 else "unknown"
        blob = Blob(source=str(p), kind="pdf", lang=lang, text_norm=norm)
        if lang.startswith("ne"):
            blob.text_en = translator.translate(norm[:char_limit])
        elif lang.startswith("en"):
            blob.text_en = norm
        entity.blobs.append(blob)

    # If nothing collected, skip
    if not entity.blobs:
        return None

    # Lightweight dedupe by normalized paragraph hash
    seen: set[str] = set()
    uniq: List[Blob] = []
    for b in entity.blobs:
        key = (b.text_en or b.text_norm or "").strip()
        key = re.sub(r"\s+", " ", key)
        if len(key) < min_para_len:
            continue
        h = hash(key)
        if h in seen:
            continue
        seen.add(h)
        uniq.append(b)
    entity.blobs = uniq
    return entity

# --------------------------- Writers --------------------------- #

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "entity"


def to_markdown(e: EntityDoc) -> str:
    lines = [
        f"# {e.name}",
        "",
        f"Domain: {e.domain}",
        f"Type: {e.entity_type}",
        "",
        "## Sources",
    ]
    for f in sorted(set(e.files)):
        lines.append(f"- {f}")
    lines += ["", "## Collected Text (English where available)"]
    for i, b in enumerate(e.blobs, 1):
        txt = (b.text_en or b.text_norm or "").strip()
        if len(txt) > 1200:
            txt = txt[:1200] + " …"
        lines.append(f"\n### Blob {i} ({b.kind})\nSource: {b.source}\nLang: {b.lang}\n\n{txt}\n")
    return "\n".join(lines)


def write_entity(out_dir: Path, e: EntityDoc, fmt: str) -> None:
    (out_dir / "entities").mkdir(parents=True, exist_ok=True)
    slug = slugify(e.name)
    if fmt in ("json", "both"):
        data = asdict(e)
        # convert dataclasses inside
        data["blobs"] = [asdict(b) for b in e.blobs]
        (out_dir / "entities" / f"{slug}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if fmt in ("md", "both"):
        (out_dir / "entities" / f"{slug}.md").write_text(to_markdown(e), encoding="utf-8")

# --------------------------- Main --------------------------- #

def main() -> None:
    args = build_argparser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Translator
    translator: Translator
    if args.translate_provider == "argos":
        try:
            translator = ArgosTranslator()
        except Exception:
            translator = Translator()
    else:
        translator = Translator()

    count = 0
    for folder in sorted(args.in_root.iterdir()):
        if not folder.is_dir():
            continue
        ent = process_domain_folder(folder, translator=translator, char_limit=args.translate_char_limit, min_para_len=args.min_paragraph_len, page_cap=args.max_pdf_pages)
        if not ent:
            continue
        write_entity(args.out, ent, fmt=args.format)
        count += 1
    print(f"✓ Wrote {count} entity files to {args.out / 'entities'}")


if __name__ == "__main__":
    main()
