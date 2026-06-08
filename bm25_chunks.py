"""Query-time CHUNK-level BM25: score every chunk, then max-aggregate to pages.

Page-level BM25 dilutes a query whose terms all land in a single sentence of a
long page. Scoring each short chunk instead lets that one passage spike, and
taking each page's best chunk score recovers a strong page ranking. This index
gives the highest candidate recall and is fused with the dense retriever to form
the candidate pool in retrieve.py.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from utils import ARTIFACTS_DIR
from build_bm25 import tokenize


class ChunkBM25:
    """Loads bm25c_postings.npz / bm25c_vocab.json and scores queries over chunks."""

    def __init__(self, artifacts_dir: Optional[Path] = None):
        """Load chunk postings and precompute per-chunk length normalisation."""
        root = artifacts_dir or ARTIFACTS_DIR
        z = np.load(root / "bm25c_postings.npz")
        self.docs = z["docs"]
        self.tfs = z["tfs"].astype(np.float32)
        self.offsets = z["offsets"]
        self.idf = z["idf"]
        self.chunk_len = z["chunk_len"].astype(np.float32)
        self.chunk_page_ids = z["chunk_page_ids"]
        self.k1, self.b, self.avgdl = (float(x) for x in z["params"])
        self.vocab = json.loads((root / "bm25c_vocab.json").read_text(encoding="utf-8"))
        self._lennorm = self.k1 * (1.0 - self.b + self.b * self.chunk_len / self.avgdl)
        # Map each chunk to a compact page index so chunk scores can be
        # max-aggregated to pages without a Python loop (np.maximum.at below).
        self.unique_pages, self._page_idx = np.unique(self.chunk_page_ids, return_inverse=True)

    def chunk_scores(self, query: str) -> np.ndarray:
        """Return a BM25 score for every chunk, aligned to self.chunk_page_ids."""
        out = np.zeros(self.chunk_page_ids.shape[0], dtype=np.float32)
        seen = set()
        for tk in tokenize(query):
            if tk in seen:
                continue
            seen.add(tk)
            tid = self.vocab.get(tk)
            if tid is None:
                continue
            s, e = int(self.offsets[tid]), int(self.offsets[tid + 1])
            docs = self.docs[s:e]
            tf = self.tfs[s:e]
            out[docs] += self.idf[tid] * (tf * (self.k1 + 1.0)) / (tf + self._lennorm[docs])
        return out

    def page_scores(self, query: str):
        """Aggregate chunk scores to pages by taking each page's max chunk score.

        Returns (unique_page_ids, page_score), two arrays aligned to each other.
        """
        cs = self.chunk_scores(query)
        page_score = np.zeros(self.unique_pages.shape[0], dtype=np.float32)
        np.maximum.at(page_score, self._page_idx, cs)
        return self.unique_pages, page_score

    def ranked_pages(self, query: str, top: int) -> List[int]:
        """Return the top-`top` page_ids by max-chunk BM25 score, highest first."""
        pages, sc = self.page_scores(query)
        top = min(top, sc.shape[0])
        idx = np.argpartition(-sc, top - 1)[:top]
        idx = idx[np.argsort(-sc[idx])]
        return [int(pages[i]) for i in idx]
