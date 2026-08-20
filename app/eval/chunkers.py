"""Chunking strategies for the Evaluation Lab.

Four strategies, selectable per run:

- **fixed**       — the production strategy (`ingestion.chunk_pages`):
                     packs raw page text into ~target_tokens windows with a
                     token-bounded overlap tail. Doesn't respect sentence or
                     paragraph boundaries.
- **sentence**     — packs whole sentences up to ~target_tokens; a chunk
                     never ends mid-sentence.
- **recursive**     — LangChain-style recursive splitting: tries paragraph
                     breaks first, then line breaks, then sentence breaks,
                     then word breaks, only recursing into a separator when
                     the current piece is still over budget. Small adjacent
                     pieces are then merged back up to ~target_tokens.
- **semantic**     — embeds each sentence and inserts a chunk boundary
                     wherever similarity to the next sentence drops below
                     the configured percentile of all consecutive
                     similarities (the "percentile breakpoint" method used
                     by LlamaIndex's SemanticSplitterNodeParser and
                     LangChain's SemanticChunker). Requires an embedding
                     model, so it is more expensive to build than the other
                     three.

All four converge on the same `Chunk` shape as `ingestion.py`'s production
chunker (text, filename, doc_id, page_start, page_end, section, chunk_index),
so nothing downstream has to know which strategy produced a given chunk.
"""
from __future__ import annotations

import re

import numpy as np

from ..ingestion import Chunk, Page, chunk_pages, count_tokens, _detect_heading
from ..config import ChunkConfig
from . import embed_cache

CHUNKING_STRATEGIES = [
    {"id": "fixed", "label": "Fixed-size (production default)"},
    {"id": "sentence", "label": "Sentence-packed"},
    {"id": "recursive", "label": "Recursive (paragraph→line→sentence→word)"},
    {"id": "semantic", "label": "Semantic (embedding breakpoints)"},
]

_SENT_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\(\[])')
_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " "]


# ==========================================================================
# Build one offset-tracked document string from a document's pages, so any
# character span in it can be mapped back to a page range and nearest
# heading -- the strategies below all operate on this flat string rather
# than page-by-page, which is what lets a "chunk" span a page boundary or
# respect a paragraph break that a fixed page-buffer approach would miss.
# ==========================================================================

def _build_document_text(pages: list[Page]) -> tuple[str, list[tuple[int, int, int]], list[tuple[int, str]]]:
    parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []
    headings: list[tuple[int, str]] = []
    cursor = 0
    for page in pages:
        text = page.text
        start = cursor
        line_offset = 0
        for line in text.splitlines(keepends=True):
            heading = _detect_heading(line.rstrip("\n"))
            if heading:
                headings.append((cursor + line_offset, heading))
            line_offset += len(line)
        parts.append(text)
        cursor += len(text)
        page_spans.append((start, cursor, page.number))
        parts.append("\n\n")
        cursor += 2
    return "".join(parts), page_spans, headings


def _page_range_for_span(page_spans: list[tuple[int, int, int]], start: int, end: int) -> tuple[int, int]:
    matched = [pn for (s, e, pn) in page_spans if e > start and s < end]
    if matched:
        return min(matched), max(matched)
    for s, e, pn in page_spans:
        if s <= start < e:
            return pn, pn
    if page_spans:
        return page_spans[0][2], page_spans[-1][2]
    return 1, 1


def _heading_before(headings: list[tuple[int, str]], pos: int) -> str:
    result = ""
    for hpos, h in headings:
        if hpos <= pos:
            result = h
        else:
            break
    return result


# ==========================================================================
# Atomic span generators (one per strategy) -- these decide WHERE it is
# acceptable to cut. Packing them up to target_tokens is a separate,
# shared step (`_pack_spans`) so the merge/overlap logic only has to be
# written once.
# ==========================================================================

def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        end = m.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans or ([(0, len(text))] if text.strip() else [])


def _recursive_split_spans(text: str, target_tokens: int,
                           seps: list[str] | None = None) -> list[tuple[int, int]]:
    seps = _RECURSIVE_SEPARATORS if seps is None else seps
    if not text.strip():
        return []
    if count_tokens(text) <= target_tokens or not seps:
        return [(0, len(text))]
    sep = seps[0]
    pieces: list[tuple[int, int]] = []
    cursor = 0
    for part in text.split(sep):
        piece_start = cursor
        piece_end = cursor + len(part)
        if part.strip():
            if count_tokens(part) > target_tokens and len(seps) > 1:
                for s, e in _recursive_split_spans(part, target_tokens, seps[1:]):
                    pieces.append((piece_start + s, piece_start + e))
            else:
                pieces.append((piece_start, piece_end))
        cursor = piece_end + len(sep)
    return pieces


def _semantic_chunk_spans(text: str, embed_model: str, target_tokens: int,
                          breakpoint_percentile: float = 25.0) -> list[tuple[int, int]]:
    sent_spans = _sentence_spans(text)
    if len(sent_spans) <= 1:
        return sent_spans

    sentences = [text[s:e] for s, e in sent_spans]
    vecs = embed_cache.embed_passages(embed_model, sentences)  # already L2-normalized
    if len(vecs) < 2:
        return sent_spans
    sims = [float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(vecs) - 1)]
    threshold = float(np.percentile(sims, breakpoint_percentile))

    groups: list[tuple[int, int]] = []
    seg_start = 0
    for i, sim in enumerate(sims):
        if sim < threshold:
            groups.append((seg_start, i))
            seg_start = i + 1
    groups.append((seg_start, len(sent_spans) - 1))

    raw = [(sent_spans[a][0], sent_spans[b][1]) for a, b in groups]

    # Safety net: a semantic group with no low-similarity breakpoints inside
    # it (e.g. a long, topically uniform passage) can still come out larger
    # than the target. Re-split anything well over budget with the recursive
    # splitter rather than emitting one oversized chunk.
    refined: list[tuple[int, int]] = []
    for s, e in raw:
        piece = text[s:e]
        if count_tokens(piece) > target_tokens * 1.6:
            for ss, ee in _recursive_split_spans(piece, target_tokens):
                refined.append((s + ss, s + ee))
        else:
            refined.append((s, e))
    return refined


# ==========================================================================
# Shared packer: merge adjacent atomic spans up to ~target_tokens, carrying
# a token-bounded overlap tail into the next chunk. Mirrors the overlap
# strategy in ingestion.chunk_pages, generalized to arbitrary atomic spans
# instead of page buffering.
# ==========================================================================

def _pack_spans(full_text: str, atomic_spans: list[tuple[int, int]],
                target_tokens: int, overlap_tokens: int, min_tokens: int) -> list[tuple[int, int]]:
    if not atomic_spans:
        return []
    chunks: list[tuple[int, int]] = []
    current: list[tuple[int, int]] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if not current:
            return
        start, end = current[0][0], current[-1][1]
        text = full_text[start:end].strip()
        if text and (count_tokens(text) >= min_tokens or not chunks):
            chunks.append((start, end))
        tail: list[tuple[int, int]] = []
        tail_tokens = 0
        for span in reversed(current):
            t = count_tokens(full_text[span[0]:span[1]])
            if tail_tokens + t > overlap_tokens:
                break
            tail.insert(0, span)
            tail_tokens += t
        current = tail
        current_tokens = tail_tokens

    for span in atomic_spans:
        span_tokens = count_tokens(full_text[span[0]:span[1]])
        if current and current_tokens + span_tokens > target_tokens:
            flush()
        current.append(span)
        current_tokens += span_tokens
        if current_tokens >= target_tokens:
            flush()
    if current:
        start, end = current[0][0], current[-1][1]
        text = full_text[start:end].strip()
        if text and (count_tokens(text) >= min_tokens or not chunks):
            chunks.append((start, end))
    return chunks


# ==========================================================================
# Public entry point
# ==========================================================================

def chunk_document(pages: list[Page], filename: str, doc_id: str,
                   chunk_config: ChunkConfig, strategy: str = "fixed",
                   embed_model: str | None = None) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_pages(pages, filename=filename, doc_id=doc_id, chunk_config=chunk_config)

    full_text, page_spans, headings = _build_document_text(pages)
    if not full_text.strip():
        return []

    if strategy == "sentence":
        atomic = _sentence_spans(full_text)
    elif strategy == "recursive":
        atomic = _recursive_split_spans(full_text, chunk_config.target_tokens)
    elif strategy == "semantic":
        if not embed_model:
            raise ValueError("Semantic chunking requires an embedding model.")
        atomic = _semantic_chunk_spans(full_text, embed_model, chunk_config.target_tokens)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    packed = _pack_spans(full_text, atomic, chunk_config.target_tokens,
                         chunk_config.overlap_tokens, chunk_config.min_tokens)

    chunks: list[Chunk] = []
    for i, (s, e) in enumerate(packed):
        text = full_text[s:e].strip()
        if not text:
            continue
        page_start, page_end = _page_range_for_span(page_spans, s, e)
        section = _heading_before(headings, s)
        chunks.append(Chunk(text=text, filename=filename, doc_id=doc_id,
                            page_start=page_start, page_end=page_end,
                            section=section, chunk_index=i))
    return chunks
