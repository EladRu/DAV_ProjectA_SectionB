"""Corpus preprocessing and chunking into small, sentence-aware passages.

Design rationale: the queries read as if written from a single sentence of the
answer page, so embedding short (roughly sentence-scale) passages rather than
large fixed-width blocks gives both the dense retriever and the cross-encoder a
much tighter target. Each chunk is the page TITLE followed by a window of
consecutive sentences totalling about TARGET_WORDS words, with a one-sentence
overlap between consecutive chunks so a sentence on a boundary is never stripped
of its neighbouring context.

This same chunking is reused to build the chunk-level BM25 index, so the dense
index and the chunk-BM25 index share an identical chunk ordering and page map.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

TARGET_WORDS = 45            # approximate words per chunk (~1-2 sentences)
MAX_CHUNKS_PER_PAGE = 80     # safety cap so a very long page cannot explode

# Names kept for backward compatibility with index.py's import; they are only
# recorded in the index metadata (the actual splitting below is sentence-based).
CHUNK_WORDS = TARGET_WORDS
OVERLAP_WORDS = 0            # overlap is measured in sentences, not words

_SENT = re.compile(r"(?<=[.!?])\s+")   # split after sentence-ending punctuation


@dataclass
class Chunk:
    """One indexed passage: its source page, its order on the page, and text."""
    page_id: int
    chunk_id: int
    text: str


def _sentences(text: str) -> List[str]:
    """Collapse whitespace and split text into a list of non-empty sentences."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s for s in _SENT.split(text) if s]


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus page into title-prefixed, sentence-aware chunks.

    A page with no content collapses to a single title-only chunk. Otherwise we
    accumulate whole sentences until the running word count reaches TARGET_WORDS,
    prepend the title to each chunk, then begin the next chunk one sentence back
    (the overlap) while always advancing.
    """
    page_id = int(record["page_id"])
    title = (record.get("title") or "").strip()
    sents = _sentences(record.get("content") or "")
    if not sents:
        return [Chunk(page_id=page_id, chunk_id=0, text=title or str(page_id))]

    chunks: List[Chunk] = []
    i, cid = 0, 0
    n = len(sents)
    while i < n:
        # Grow the current chunk one sentence at a time up to the word target.
        cur: List[str] = []
        wc, j = 0, i
        while j < n and (wc < TARGET_WORDS or not cur):
            cur.append(sents[j])
            wc += len(sents[j].split())
            j += 1
        body = " ".join(cur)
        text = f"{title}. {body}" if title else body
        chunks.append(Chunk(page_id=page_id, chunk_id=cid, text=text))
        cid += 1
        if j >= n or cid >= MAX_CHUNKS_PER_PAGE:
            break
        # Step back one sentence for overlap, but never stall on the same index.
        i = j - 1 if (j - 1) > i else j
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    """Chunk every record in the corpus, preserving input order."""
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
