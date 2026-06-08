"""Offline CHUNK-level BM25 index builder (pure numpy + standard library).

Page-level BM25 dilutes a query whose terms all co-occur in a single sentence of
a long page. Building BM25 over the small chunks instead lets that one passage
score highly; at query time chunk scores are max-aggregated back to pages. The
chunks come from the SAME chunk_corpus used to build the dense index, so chunk
ordering and the chunk->page mapping stay consistent across both indexes.

Output artifacts (written to ARTIFACTS_DIR):
  bm25c_postings.npz  -- CSR postings + idf + chunk_len + chunk_page_ids
                         + [k1, b, avgdl]
  bm25c_vocab.json    -- {term: term_id}

Run offline:  python build_bm25_chunks.py
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries
from chunk import chunk_corpus
from build_bm25 import tokenize, K1, B


def build_bm25_chunks(*, entries_dir: Optional[Path] = None,
                      artifacts_dir: Optional[Path] = None) -> None:
    """Chunk the corpus, build CSR postings over chunks, and write the artifact."""
    out_dir = artifacts_dir or ensure_artifacts_dir()
    t0 = time.perf_counter()

    # Reuse the dense index's chunking so chunk indices line up across artifacts.
    records = list(iter_entries(entries_dir))
    chunks = chunk_corpus(records)
    chunk_page_ids = [int(c.page_id) for c in chunks]

    # Tokenise each chunk and accumulate per-term postings (chunk index + tf).
    chunk_len = []
    post_docs: dict = defaultdict(list)
    post_tfs: dict = defaultdict(list)
    for ci, c in enumerate(chunks):
        toks = tokenize(c.text)
        chunk_len.append(len(toks))
        tf: dict = {}
        for tk in toks:
            tf[tk] = tf.get(tk, 0) + 1
        for tk, cnt in tf.items():
            post_docs[tk].append(ci)
            post_tfs[tk].append(cnt)

    n = len(chunks)
    avgdl = float(np.mean(chunk_len)) if n else 0.0
    print(f"[bm25c] {n} chunks tokenized in {time.perf_counter()-t0:.1f}s  "
          f"vocab={len(post_docs)}  avgdl={avgdl:.1f}")

    # Flatten to CSR with a sorted vocab id space (chunk tfs are small -> int16).
    terms = sorted(post_docs)
    vocab = {t: i for i, t in enumerate(terms)}
    offsets = np.zeros(len(terms) + 1, dtype=np.int64)
    total = sum(len(post_docs[t]) for t in terms)
    docs_arr = np.empty(total, dtype=np.int32)
    tfs_arr = np.empty(total, dtype=np.int16)
    idf = np.empty(len(terms), dtype=np.float32)
    pos = 0
    for i, t in enumerate(terms):
        d = post_docs[t]
        m = len(d)
        docs_arr[pos:pos + m] = d
        tfs_arr[pos:pos + m] = post_tfs[t]
        offsets[i + 1] = pos + m
        idf[i] = np.log(1.0 + (n - m + 0.5) / (m + 0.5))   # BM25 idf over chunks
        pos += m

    np.savez(out_dir / "bm25c_postings.npz",
             docs=docs_arr, tfs=tfs_arr, offsets=offsets, idf=idf,
             chunk_len=np.asarray(chunk_len, dtype=np.int32),
             chunk_page_ids=np.asarray(chunk_page_ids, dtype=np.int64),
             params=np.asarray([K1, B, avgdl], dtype=np.float64))
    (out_dir / "bm25c_vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    print(f"[bm25c] postings={total}  wrote artifact ({time.perf_counter()-t0:.1f}s total)")


if __name__ == "__main__":
    build_bm25_chunks()
