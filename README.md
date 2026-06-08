# Project A — Section B: Wikipedia Passage Retrieval

End-to-end retrieval over a ~27,000-page synthetic Wikipedia corpus. Given a
natural-language query, `run()` returns the 10 most relevant page IDs. The
system is evaluated by mean **NDCG@10** over a set of hidden queries, under a
**60-second** budget for one batched call on the evaluation GPU (Tesla M60).

## Approach

A three-stage pipeline, arrived at empirically — dense retrieval alone plateaued
well below this configuration:

**1. Recall — hybrid candidate retrieval.** Two complementary retrievers are
merged with Reciprocal Rank Fusion (RRF):

- **Dense:** `sentence-transformers/all-MiniLM-L6-v2` embeddings over small,
  sentence-aware chunks. Strong on paraphrase and meaning ("physicist" ≈
  "scientist").
- **Lexical:** chunk-level BM25. Strong on the exact rare tokens many queries
  hinge on — specific numbers, years, and proper nouns — which dense embeddings
  tend to blur.

The two have opposite blind spots, so fusing them lifts candidate recall above
either alone (dense recall@100 ≈ 0.54; the fused pool ≈ 0.80).

**2. Chunk selection.** Pages are long (median ~1,200 words), so each candidate
page is reduced to a single passage: the chunk with the highest idf-weighted
overlap with the query terms. This gives the reranker the most on-topic text
instead of an arbitrary slice of a long page.

**3. Precision — cross-encoder rerank.** `cross-encoder/ms-marco-MiniLM-L-12-v2`
scores each (query, chunk) pair, and the final page ranking is taken from those
scores.

## Result

| Metric | Value |
| --- | --- |
| Mean NDCG@10 (public queries) | **0.3066** |
| Query time, 50 queries (Tesla M60) | ≈ 50–55 s (within the 60 s budget) |

## Repository layout

```
main.py                 Entry point. run(queries) (called by the grader) + build_offline_index()
retrieve.py             Query-time pipeline: hybrid recall -> chunk pick -> cross-encoder rerank
chunk.py                Sentence-aware chunking of corpus pages
embed.py                MiniLM embedding wrapper (queries and chunks)
index.py                Dense index build + load; memory-mapped chunk-text store
bm25.py                 Query-time page-level BM25 (loads artifact; supplies idf)
bm25_chunks.py          Query-time chunk-level BM25 (loads artifact; the high-recall retriever)
build_bm25.py           Offline page-BM25 builder; shared tokenizer
build_bm25_chunks.py    Offline chunk-BM25 builder
utils.py                Paths, corpus iteration, query loading
eval.py                 NDCG@10 metric (provided)
scripts/build_index.py  Build entry point (provided)
scripts/eval_public.py  Public-query evaluation (provided)
artifacts/              Prebuilt index, committed via Git LFS (see below)
data/public_queries.json  Public evaluation queries
```

## Setup

The prebuilt index in `artifacts/` is stored with **Git LFS**, so Git LFS must
be installed before cloning, or the large files will arrive as pointer stubs:

```bash
git lfs install
git clone https://github.com/EladRu/DAV_ProjectA_SectionB.git
cd DAV_ProjectA_SectionB
pip install -r requirements.txt
```

After cloning, confirm the index downloaded fully (this should be ~790 MB, not a
few bytes):

```bash
ls -lh artifacts/index_vectors.npy
```

## Running the evaluation (no rebuild required)

The index is prebuilt and committed, so evaluation runs directly against the
artifacts — no indexing step is needed:

```bash
python scripts/eval_public.py
```

This prints the mean NDCG@10 over the public queries and the total query time.

## Rebuilding the index from scratch (optional)

Only needed to regenerate the artifacts. Requires the corpus under
`data/Wikipedia Entries/` (not committed to the repo). Run from the repo root:

```bash
python scripts/build_index.py   # dense MiniLM index + memory-mapped text store
python build_bm25.py            # page-level BM25 artifact
python build_bm25_chunks.py     # chunk-level BM25 artifact
```

All three index over the same chunking, so they must be rebuilt together.

## Requirements

Python 3.10 and the packages pinned in `requirements.txt`: PyTorch (CUDA 12.1
build), `sentence-transformers`, `transformers`, `numpy`, `scipy`, and
`faiss-cpu`. A CUDA-capable GPU is used for embedding and cross-encoder
inference.

## Authors

- Naama Inbar
- Elad Rubani
