"""Offline dense-index build and load (the build step is not timed at grading).

Builds the MiniLM chunk-embedding index plus its supporting on-disk artifacts,
and provides the loaders the query-time pipeline relies on:
  index_vectors.npy      (num_chunks, 384) float16 chunk embeddings
  index_meta.json        page_id per chunk + build parameters
  index_texts.bin        all chunk texts concatenated as UTF-8 bytes
  index_texts_off.npy    int64 byte offsets, length num_chunks + 1

Chunk texts are stored as a single blob plus an offset array and read through a
memory-mapped TextStore, so the whole corpus text is never held in RAM at once.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from chunk import CHUNK_WORDS, OVERLAP_WORDS, Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

INDEX_VECTORS_NAME = "index_vectors.npy"      # (num_chunks, 384) float16
INDEX_META_NAME = "index_meta.json"           # page_id per chunk + params
TEXT_BLOB_NAME = "index_texts.bin"            # all chunk texts, utf-8 concatenated
TEXT_OFF_NAME = "index_texts_off.npy"         # int64 offsets, len num_chunks+1
INDEX_TEXTS_JSON = "index_texts.json"         # legacy fallback


class TextStore:
    """Lazy, memory-mapped access to chunk texts by index (no full RAM load)."""

    def __init__(self, blob_path: Path, off_path: Path):
        """Memory-map the text blob and load its offset array."""
        self._blob = np.memmap(blob_path, dtype=np.uint8, mode="r")
        self._off = np.load(off_path)

    def __getitem__(self, i: int) -> str:
        """Return chunk i, decoded from the blob slice between its offsets."""
        a, b = int(self._off[i]), int(self._off[i + 1])
        return bytes(self._blob[a:b]).decode("utf-8")

    def __len__(self) -> int:
        """Number of stored chunks."""
        return len(self._off) - 1


def _write_text_blob(out_dir: Path, texts: List[str]) -> None:
    """Write all chunk texts as one UTF-8 blob with a matching offsets array."""
    offs = np.zeros(len(texts) + 1, dtype=np.int64)
    pos = 0
    with open(out_dir / TEXT_BLOB_NAME, "wb") as f:
        for i, t in enumerate(texts):
            b = t.encode("utf-8")
            f.write(b)
            pos += len(b)
            offs[i + 1] = pos
    np.save(out_dir / TEXT_OFF_NAME, offs)


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Chunk the corpus, embed every chunk with MiniLM, and persist all artifacts.

    Returns the embedding matrix (float32 in memory, stored as float16) and the
    per-chunk page_id list. Run once offline; the grader loads the result.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()

    t0 = time.perf_counter()
    records = list(iter_entries(entries_dir))
    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    page_ids = [c.page_id for c in chunks]
    print(f"[build] {len(records)} pages -> {len(chunks)} chunks "
          f"({time.perf_counter()-t0:.1f}s to chunk)")

    t0 = time.perf_counter()
    vectors = embed_texts(texts, batch_size=256)
    print(f"[build] embedded {len(texts)} chunks in {time.perf_counter()-t0:.1f}s")

    # Persist vectors as float16 to roughly halve the artifact size on disk.
    np.save(out_dir / INDEX_VECTORS_NAME, vectors.astype(np.float16))
    (out_dir / INDEX_META_NAME).write_text(json.dumps({
        "page_ids": page_ids,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
        "chunk_words": CHUNK_WORDS,
        "overlap_words": OVERLAP_WORDS,
    }), encoding="utf-8")
    _write_text_blob(out_dir, texts)
    print(f"[build] wrote artifacts to {out_dir}")
    return vectors, page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load chunk vectors (restored to float32) and the chunk->page_id list."""
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME).astype(np.float32)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids


def load_texts(artifacts_dir: Optional[Path] = None) -> Union[TextStore, List[str]]:
    """Return a memory-mapped TextStore, or a plain list from the legacy JSON."""
    root = artifacts_dir or ARTIFACTS_DIR
    if (root / TEXT_BLOB_NAME).exists() and (root / TEXT_OFF_NAME).exists():
        return TextStore(root / TEXT_BLOB_NAME, root / TEXT_OFF_NAME)
    return json.loads((root / INDEX_TEXTS_JSON).read_text(encoding="utf-8"))
