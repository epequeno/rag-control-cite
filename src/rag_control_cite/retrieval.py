"""Stage 1 — naive baseline: embed → top-k.

Dense retrieval over the in-memory index (numpy cosine). Ends in:
recall@10, groundedness, p50/p95, $/1k queries.
"""

from __future__ import annotations

from rag_control_cite.models import Retrieved

__all__ = ["retrieve"]


def retrieve(
    index: object,
    query_vec: object,
    k: int = 10,
) -> list[Retrieved]:
    """Dense top-k retrieval from a pre-built index.

    Args:
        index: an ``Index`` with ``chunks`` and ``retrieve(query_vec, k)``.
        query_vec: L2-normalized query embedding (numpy array).
        k: number of candidates to retrieve.

    Returns:
        List of Retrieved, 1-based rank, sorted by score descending.
    """
    hits = index.retrieve(query_vec, k)  # type: ignore[attr-defined]
    return [
        Retrieved(chunk=index.chunks[i], score=score, rank=rank + 1, source="dense")  # type: ignore[attr-defined]
        for rank, (i, score) in enumerate(hits)
    ]
