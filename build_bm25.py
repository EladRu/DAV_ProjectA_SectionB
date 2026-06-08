"""Offline page-level BM25 index builder (pure numpy + standard library).

Why a lexical index at all: the dense MiniLM retriever misses the exact tokens
many queries depend on -- specific numbers (e.g. "1,456,779"), years ("1820s"),
and rare phrases. BM25 matches those directly, so fusing it with the dense
retriever at query time lifts recall before reranking. BM25 is a lexical signal,
not an embedding, so it does not conflict with the requirement that the initial
retrieval embeddings come from MiniLM.

Output artifacts (written to ARTIFACTS_DIR):
  bm25_postings.npz  -- CSR postings (docs, tfs, offsets) + idf + doc_len
                        + page_ids + [k1, b, avgdl]
  bm25_vocab.json    -- {term: term_id}

Run once, offline (not part of the timed grading run):
  python build_bm25.py
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

K1 = 1.5        # BM25 term-frequency saturation
B = 0.75        # BM25 length-normalisation strength
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase and split text into alphanumeric tokens.

    Commas are stripped first so a figure like "1,456,779" becomes the single
    token "1456779", matching how the same number is written in the queries.
    This tokenizer is shared by the query-time BM25 classes for consistency.
    """
    return _TOKEN.findall(text.lower().replace(",", ""))


def build_bm25(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> None:
    """Tokenise every page, build CSR postings + idf, and write the artifact."""
    out_dir = artifacts_dir or ensure_artifacts_dir()

    t0 = time.perf_counter()
    page_ids: List[int] = []
    doc_len: List[int] = []
    post_docs: dict = defaultdict(list)   # term -> [doc_idx, ...]
    post_tfs: dict = defaultdict(list)    # term -> [term frequency in that doc, ...]

    # Pass 1: tokenise each page and record per-term (doc index, frequency).
    for doc_idx, rec in enumerate(iter_entries(entries_dir)):
        title = (rec.get("title") or "").strip()
        content = (rec.get("content") or "").strip()
        toks = tokenize(title + "\n" + content)
        page_ids.append(int(rec["page_id"]))
        doc_len.append(len(toks))
        tf: dict = {}
        for tk in toks:
            tf[tk] = tf.get(tk, 0) + 1
        for tk, c in tf.items():
            post_docs[tk].append(doc_idx)
            post_tfs[tk].append(c)

    n_docs = len(page_ids)
    avgdl = float(np.mean(doc_len)) if n_docs else 0.0
    print(f"[bm25] {n_docs} pages tokenized in {time.perf_counter()-t0:.1f}s  "
          f"vocab={len(post_docs)}  avgdl={avgdl:.1f}")

    # Pass 2: flatten postings into CSR arrays with a stable, sorted vocab id space.
    terms = sorted(post_docs.keys())
    vocab = {t: i for i, t in enumerate(terms)}
    offsets = np.zeros(len(terms) + 1, dtype=np.int64)
    total = sum(len(post_docs[t]) for t in terms)
    docs_arr = np.empty(total, dtype=np.int32)
    tfs_arr = np.empty(total, dtype=np.int32)
    idf = np.empty(len(terms), dtype=np.float32)

    pos = 0
    for i, t in enumerate(terms):
        d = post_docs[t]
        f = post_tfs[t]
        m = len(d)
        docs_arr[pos:pos + m] = d
        tfs_arr[pos:pos + m] = f
        offsets[i + 1] = pos + m
        df = m                                                  # document frequency
        idf[i] = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))  # BM25 idf
        pos += m

    np.savez(
        out_dir / "bm25_postings.npz",
        docs=docs_arr,
        tfs=tfs_arr,
        offsets=offsets,
        idf=idf,
        doc_len=np.asarray(doc_len, dtype=np.int32),
        page_ids=np.asarray(page_ids, dtype=np.int64),
        params=np.asarray([K1, B, avgdl], dtype=np.float64),
    )
    (out_dir / "bm25_vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    print(f"[bm25] postings={total}  wrote artifact to {out_dir} "
          f"({time.perf_counter()-t0:.1f}s total)")


if __name__ == "__main__":
    build_bm25()
