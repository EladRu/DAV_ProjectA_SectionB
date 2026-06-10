"""Query-time retrieval pipeline for Section B.

Turns a batch of queries into a ranked list of page_ids per query. This is the
only query-time code path the grader exercises (through main.run); the index is
built offline and merely loaded here from artifacts/.

The pipeline has three stages, settled on empirically (dense retrieval alone
plateaued well below the hybrid-plus-rerank configuration):

  Stage 1 - Recall.  Two complementary retrievers vote on candidate pages:
            dense MiniLM similarity (strong on paraphrase / meaning) and
            chunk-level BM25 (strong on the exact rare tokens the queries hinge
            on). Their rankings are merged with Reciprocal Rank Fusion (RRF) and
            the top POOL_PAGES pages are kept.
  Stage 2 - Chunk selection.  Pages are long, so each candidate page is reduced
            to one passage: the chunk with the highest idf-weighted overlap with
            the query terms, so the reranker sees the most on-topic text.
  Stage 3 - Precision.  An L-12 cross-encoder scores each (query, chunk) pair,
            and the final page ranking is taken from those scores.

Heavy objects (vectors, text store, BM25 indexes, cross-encoder) are loaded once
and cached in module-level globals, so repeated calls within a run are cheap.
"""
from __future__ import annotations

import collections
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from embed import embed_queries
from index import load_index, load_texts
from utils import K_EVAL
from bm25 import BM25
from bm25_chunks import ChunkBM25
from build_bm25 import tokenize

POOL_PAGES = 50         # candidates kept after fusion; NDCG flat 70..150
RETRIEVE_DEPTH = 500      # how deep each retriever's ranking is read before fusion
RRF_K = 60                # reciprocal-rank-fusion damping constant
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
CE_BATCH = 64             # cross-encoder prediction batch size

# Cached singletons, populated by _ensure_loaded on the first query of a run.
_VECTORS = _PAGE_IDS = _TEXTS = _CE = _BM25 = _CBM = None
_PAGE_TO_CHUNKS: Optional[Dict[int, np.ndarray]] = None
_IDF: Optional[Dict[str, float]] = None


def _ensure_loaded(artifacts_dir: Optional[Path]) -> None:
    """Load and cache every artifact and model once; later calls are no-ops."""
    global _VECTORS, _PAGE_IDS, _TEXTS, _CE, _BM25, _CBM, _PAGE_TO_CHUNKS, _IDF
    if _VECTORS is None:
        vecs, pids = load_index(artifacts_dir)
        _VECTORS = vecs
        _PAGE_IDS = np.asarray(pids, dtype=np.int64)
        _TEXTS = load_texts(artifacts_dir)
        # Group chunk indices by their page so a page's chunks can be looked up.
        groups: Dict[int, list] = collections.defaultdict(list)
        for i, pid in enumerate(_PAGE_IDS.tolist()):
            groups[pid].append(i)
        _PAGE_TO_CHUNKS = {pid: np.asarray(ix, dtype=np.int64) for pid, ix in groups.items()}
    if _BM25 is None:
        _BM25 = BM25(artifacts_dir)
        # Per-term idf, reused to weight query-term overlap when picking a chunk.
        _IDF = {t: float(_BM25.idf[i]) for t, i in _BM25.vocab.items()}
    if _CBM is None:
        _CBM = ChunkBM25(artifacts_dir)
    if _CE is None:
        _CE = CrossEncoder(RERANK_MODEL)


def _dense_ranked(row: np.ndarray, top: int) -> List[int]:
    """Rank pages by dense similarity, keeping one entry per page.

    `row` holds a similarity score per chunk; a page can own several chunks, so
    we walk chunks best-first and keep each page's first (best) appearance.
    """
    order = np.argsort(-row)
    seen: set = set(); ranked: List[int] = []
    for idx in order:
        pid = int(_PAGE_IDS[idx])
        if pid in seen:
            continue
        seen.add(pid); ranked.append(pid)
        if len(ranked) >= top:
            break
    return ranked


def _rrf(*lists: List[int], K: int) -> List[int]:
    """Reciprocal Rank Fusion: merge several rankings into one.

    Each list contributes 1/(K + rank) to every page it contains, so a page
    ranked highly by either retriever rises, and pages found by both rise most.
    """
    sc: dict = collections.defaultdict(float)
    for lst in lists:
        for r, p in enumerate(lst):
            sc[p] += 1.0 / (K + r + 1)
    return sorted(sc, key=lambda p: -sc[p])


def _query_terms(query: str):
    """Compile a regex of the query's known terms plus their idf weights.

    Returns (pattern, {term: idf}), or (None, None) if no query token is in the
    BM25 vocabulary. Consumed by _pick_chunk to score query-term overlap.
    """
    terms = {t for t in tokenize(query) if t in _IDF}
    if not terms:
        return None, None
    pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b")
    return pat, {t: _IDF[t] for t in terms}


def _pick_chunk(pid: int, row: np.ndarray, pat, idf_map) -> int:
    """Pick the single most on-topic chunk of a page for the cross-encoder.

    Preferred: the chunk with the largest idf-weighted query-term overlap (rare
    matched terms count more). If no chunk matches any query term, fall back to
    the chunk with the highest dense similarity.
    """
    chunks = _PAGE_TO_CHUNKS[pid]
    if pat is not None:
        best_i, best_s = -1, 0.0
        for ci in chunks.tolist():
            matched = set(pat.findall(_TEXTS[ci].lower()))
            if matched:
                s = sum(idf_map[t] for t in matched)
                if s > best_s:
                    best_s, best_i = s, ci
        if best_i >= 0:
            return best_i
    return int(chunks[int(np.argmax(row[chunks]))])


def search_batch(queries, *, top_k: int = K_EVAL, artifacts_dir=None):
    """Rank pages for a batch of queries; one list of page_ids per query.

    Runs all three stages. The cross-encoder is called a single time over every
    (query, chunk) pair in the whole batch (for speed); `layout` records which
    slice of the score array belongs to which query so results can be split back
    out and ranked per query.
    """
    _ensure_loaded(artifacts_dir)
    qv = embed_queries(queries)
    if qv.size == 0:
        return [[] for _ in queries]
    dense_scores = qv @ _VECTORS.T

    # Stages 1-2: build each query's candidate pool and queue its CE pairs.
    all_pairs: List[list] = []
    layout: List[tuple] = []
    for qi, row in enumerate(dense_scores):
        d = _dense_ranked(row, RETRIEVE_DEPTH)
        bc = _CBM.ranked_pages(queries[qi], RETRIEVE_DEPTH)
        pool = _rrf(d, bc, K=RRF_K)[:POOL_PAGES]
        pat, idf_map = _query_terms(queries[qi])
        start = len(all_pairs)
        for pid in pool:
            all_pairs.append([queries[qi], _TEXTS[_pick_chunk(pid, row, pat, idf_map)]])
        layout.append((start, pool))

    # Stage 3: one batched cross-encoder pass over every queued pair.
    ce = (np.asarray(_CE.predict(all_pairs, batch_size=CE_BATCH, show_progress_bar=False))
          if all_pairs else np.zeros(0))

    # Split scores back per query and rank that query's pool by CE score.
    ranked: List[List[int]] = []
    for start, pool in layout:
        s = ce[start:start + len(pool)]
        order = np.argsort(-s)
        ranked.append([pool[j] for j in order[:top_k]])
    return ranked
