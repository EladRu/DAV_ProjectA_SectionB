"""Section B entry point.

The autograder imports this module and calls run(queries) exactly once with the
full batch of evaluation queries (50 hidden queries at grading time). All query
embedding and retrieval for that call must complete within the time limit, with
a GPU available. The index itself is built separately and offline through
build_offline_index, so it is not part of the timed run.
"""
from __future__ import annotations

from typing import List

from index import build_index
from retrieve import search_batch


def run(queries: List[str]) -> List[List[int]]:
    """Rank corpus pages for each query.

    Parameters
    ----------
    queries : list[str]
        Batch of query strings (e.g. the 50 hidden queries at grading time).

    Returns
    -------
    list[list[int]]
        One ranked list of page_id per query, most relevant first. Only the
        first 10 ids of each list are scored.
    """
    return search_batch(queries)


def build_offline_index() -> None:
    """Build artifacts/ once locally (not part of the timed grading run)."""
    build_index()


if __name__ == "__main__":
    build_offline_index()
    print("Index built under artifacts/. Run: python scripts/eval_public.py")
