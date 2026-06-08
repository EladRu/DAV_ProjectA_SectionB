"""Query-time page-level BM25: load the offline artifact and score quickly.

The heavy work (tokenising the corpus and building postings) happens offline in
build_bm25.py. Here we memory-load the CSR postings and, for a query, compute a
BM25 score per page with one vectorised accumulation per query term. In the
retrieval pipeline this index is used mainly for its idf table, which weights
query-term overlap when choosing a page's most on-topic chunk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from utils import ARTIFACTS_DIR
from build_bm25 import tokenize


class BM25:
    """Loads bm25_postings.npz / bm25_vocab.json and scores queries over pages."""

    def __init__(self, artifacts_dir: Optional[Path] = None):
        """Load CSR postings, idf, document lengths and BM25 params from disk."""
        root = artifacts_dir or ARTIFACTS_DIR
        z = np.load(root / "bm25_postings.npz")
        self.docs = z["docs"]
        self.tfs = z["tfs"].astype(np.float32)
        self.offsets = z["offsets"]
        self.idf = z["idf"]
        self.doc_len = z["doc_len"].astype(np.float32)
        self.page_ids = z["page_ids"]
        self.k1, self.b, self.avgdl = (float(x) for x in z["params"])
        self.vocab = json.loads((root / "bm25_vocab.json").read_text(encoding="utf-8"))
        # Per-document length normalisation, precomputed once: it depends only on
        # document length, k1 and b, not on the query.
        self._lennorm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)

    def scores(self, query: str) -> np.ndarray:
        """Return a BM25 score for every page, aligned to self.page_ids.

        For each unique query term we read its postings (the pages that contain
        it) and add that term's BM25 contribution to exactly those pages.
        """
        out = np.zeros(self.page_ids.shape[0], dtype=np.float32)
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
            contrib = self.idf[tid] * (tf * (self.k1 + 1.0)) / (tf + self._lennorm[docs])
            out[docs] += contrib
        return out

    def ranked_pages(self, query: str, top: int) -> List[int]:
        """Return the top-`top` page_ids by BM25 score, highest first."""
        sc = self.scores(query)
        top = min(top, sc.shape[0])
        idx = np.argpartition(-sc, top - 1)[:top]
        idx = idx[np.argsort(-sc[idx])]
        return [int(self.page_ids[i]) for i in idx]
