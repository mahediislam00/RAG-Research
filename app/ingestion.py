"""Ingestion: turn an uploaded file into clean, citable chunks.

Each chunk keeps its source filename and page range so the LLM can cite
"[solicitation.pdf p.14]" — essential when the output feeds a proposal or SOW
and every claim has to be traceable to the solicitation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import CHUNK, ChunkConfig


# --- text cleaning ------------------------------------------------------
# Repair the most common "UTF-8 decoded as Windows-1252" mojibake, which shows
# up in some PDFs/exports as e.g. "Respondentâ€™s" or "Respondentâs" for a
# right single quote. Conservative: only touches known bad sequences.
_MOJIBAKE = {
    "\u00e2\u20ac\u2122": "\u2019",  # ’
    "\u00e2\u20ac\u0153": "\u201c",  # “
    "\u00e2\u20ac\u009d": "\u201d",  # ”
    "\u00e2\u20ac\u201c": "\u2013",  # –
    "\u00e2\u20ac\u201d": "\u2014",  # —
    "\u00e2\u20ac\u2039": "\u2018",  # ‘
    "\u00e2\u20ac\u00a6": "\u2026",  # …
    "\u00e2\u0080\u0099": "\u2019",  # ’ (C1 control variant)
    "\u00e2\u0080\u009c": "\u201c",
    "\u00e2\u0080\u009d": "\u201d",
    "\u00e2\u0080\u0093": "\u2013",
    "\u00e2\u0080\u0094": "\u2014",
}


def clean_text(text: str) -> str:
    if not text:
        return text
    for bad, good in _MOJIBAKE.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


@dataclass
class Chunk:
    text: str
    filename: str
    doc_id: str
    page_start: int
    page_end: int
    section: str          # best-effort nearest heading
    chunk_index: int

    def to_dict(self) -> dict:
        return asdict(self)


# --- token estimation ---------------------------------------------------
# Prefer a real tokenizer when tiktoken is installed; otherwise approximate.
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # pragma: no cover - fallback path
    def count_tokens(text: str) -> int:
        # ~0.75 words per token for English prose.
        return int(len(text.split()) / 0.75) + 1


# --- text extraction ----------------------------------------------------
@dataclass
class Page:
    number: int
    text: str


def extract_pages(path: Path) -> list[Page]:
    """Return text split into pages. Non-paged formats yield a single page."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix in (".docx",):
        return _extract_docx(path)
    if suffix in (".txt", ".md", ".text"):
        return [Page(1, clean_text(path.read_text(encoding="utf-8", errors="ignore")))]
    raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(path: Path) -> list[Page]:
    import fitz  # PyMuPDF

    pages: list[Page] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            # "text" keeps reading order; good enough for most solicitations.
            pages.append(Page(i, clean_text(page.get_text("text"))))
    return pages


def _extract_docx(path: Path) -> list[Page]:
    from docx import Document

    doc = Document(str(path))
    parts: list[str] = []
    for para in doc.paragraphs:
        parts.append(para.text)
    # docx has no hard pages; treat the whole document as one logical page.
    return [Page(1, clean_text("\n".join(parts)))]


# --- chunking -----------------------------------------------------------
_HEADING_RE = re.compile(
    r"^\s*("
    r"(?:SECTION\s+[A-Z0-9.\-]+(?:\s*[-—]\s*[A-Z][^\n]{0,80})?)"  # SECTION L - INSTRUCTIONS
    r"|(?:[A-Z]\.\d+(?:\.\d+)*(?:\s+[A-Z][^\n]{0,80})?)"   # C.3.1  /  C.3.1 Reporting
    r"|(?:\d+(?:\.\d+){0,3}\s+[A-Z][^\n]{0,80})"           # 3.2 Scope of Work
    r"|(?:[A-Z][A-Z \-/]{4,60})"                           # ALL CAPS HEADINGS
    r")\s*$"
)


def _detect_heading(line: str) -> str | None:
    line = line.rstrip()
    if not line or len(line) > 90:
        return None
    m = _HEADING_RE.match(line)
    return line.strip() if m else None


def _split_sentences(text: str) -> list[str]:
    # Lightweight splitter that respects common abbreviations and clause codes.
    text = re.sub(r"[ \t]+", " ", text)
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    return [p.strip() for p in pieces if p.strip()]


def chunk_pages(pages: list[Page], filename: str, doc_id: str,
                chunk_config: ChunkConfig | None = None) -> list[Chunk]:
    """Pack page text into ~target_tokens chunks with overlap, tracking the
    page range and nearest heading for each chunk.

    `chunk_config` overrides the module-level default `CHUNK` for this call
    only (used by the Evaluation Lab to rebuild the same document under a
    different chunk size/overlap without mutating global config, which
    would not be safe under concurrent runs)."""
    cfg = chunk_config or CHUNK
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    buf_page_start = pages[0].number if pages else 1
    buf_page_end = buf_page_start
    current_section = ""
    idx = 0

    def flush():
        nonlocal buf, buf_tokens, idx, buf_page_start, buf_page_end
        if not buf:
            return
        text = " ".join(buf).strip()
        if count_tokens(text) >= cfg.min_tokens:
            chunks.append(
                Chunk(
                    text=text,
                    filename=filename,
                    doc_id=doc_id,
                    page_start=buf_page_start,
                    page_end=buf_page_end,
                    section=current_section,
                    chunk_index=idx,
                )
            )
            idx += 1
        buf = []
        buf_tokens = 0

    def seed_overlap(emitted: list[str], page_number: int):
        """Start the next chunk with a tail of the previous one for continuity."""
        nonlocal buf, buf_tokens, buf_page_start, buf_page_end
        tail, tail_tokens = [], 0
        for s in reversed(emitted):
            t = count_tokens(s)
            if tail_tokens + t > cfg.overlap_tokens:
                break
            tail.insert(0, s)
            tail_tokens += t
        buf = tail
        buf_tokens = tail_tokens
        buf_page_start = page_number
        buf_page_end = page_number

    for page in pages:
        for raw_line in page.text.split("\n"):
            heading = _detect_heading(raw_line)
            if heading:
                # A heading marks a logical boundary: flush so sections don't
                # bleed together, then remember it for chunk metadata.
                flush()
                current_section = heading
                buf_page_start = page.number
                buf_page_end = page.number
                continue

            for sentence in _split_sentences(raw_line):
                buf.append(sentence)
                buf_tokens += count_tokens(sentence)
                buf_page_end = page.number
                if buf_tokens >= cfg.target_tokens:
                    emitted = list(buf)
                    flush()
                    seed_overlap(emitted, page.number)

    flush()
    return chunks


def ingest_file(path: Path, doc_id: str,
                chunk_config: ChunkConfig | None = None) -> list[Chunk]:
    pages = extract_pages(path)
    # Drop pages that are effectively empty (scanned image pages with no OCR).
    pages = [p for p in pages if p.text and p.text.strip()]
    if not pages:
        raise ValueError(
            "No extractable text found. If this is a scanned PDF, run OCR first "
            "(e.g. `ocrmypdf in.pdf out.pdf`) and re-upload."
        )
    return chunk_pages(pages, filename=path.name, doc_id=doc_id, chunk_config=chunk_config)
