"""Stage 2 — hybrid retrieval: dense + BM25, reciprocal rank fusion (RRF).

The exact-term query set writes itself from OSCAL control IDs — AC-2,
AU-6(1), SC-8 — which is the codes-and-IDs case dense search misses.
BM25 wins on exact lexical match; RRF fuses the two legs by rank.

Scores from different retrievers are not comparable; ranks are. Ties
break on ``chunk_id`` so the output is deterministic across runs.

Ends in: Δ recall@10 on the exact-term query set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

from rag_control_cite.models import Retrieved

__all__ = ["BM25Retriever", "RRFFusion", "hybrid_retrieve", "rrf_fuse"]


def rrf_fuse(
    legs: list[list[Retrieved]],
    k_rrf: int = 60,
    weights: list[float] | None = None,
    k: int | None = None,
) -> list[Retrieved]:
    """Reciprocal rank fusion over already-retrieved legs.

    One implementation shared by the measured path (``RRFFusion``) and the
    served path (``store.VectorStore.retrieve_hybrid``), so a parity test
    is a tautology by construction rather than a hope. Ties break on
    ``chunk_id``.
    """
    if weights is None:
        weights = [1.0] * len(legs)
    if len(weights) != len(legs):
        raise ValueError(f"weights ({len(weights)}) must match legs ({len(legs)})")

    fused: dict[str, tuple[float, Retrieved]] = {}
    for leg, weight in zip(legs, weights, strict=True):
        for item in leg:
            gain = weight / (k_rrf + item.rank)
            key = item.chunk.chunk_id
            prev = fused.get(key)
            if prev is None:
                fused[key] = (gain, item)
            else:
                fused[key] = (prev[0] + gain, prev[1])

    ranked = sorted(fused.items(), key=lambda kv: (-kv[1][0], kv[0]))
    if k is not None:
        ranked = ranked[:k]
    return [
        Retrieved(chunk=item.chunk, score=score, rank=rank, source="fused")
        for rank, (_, (score, item)) in enumerate(ranked, start=1)
    ]


# Token pattern that keeps OSCAL control IDs as single tokens.
# Default bm25s pattern r"(?u)\b\w\w+\b" splits "AC-1" → "ac" (drops "1"
# because it's a single char). This pattern matches control IDs like
# ac-1, au-6, ac-2.1, au-6(1) as atomic tokens, then falls back to
# regular 2+ char words for everything else.
_CONTROL_ID_PATTERN = (
    r"(?u)"
    r"\b[a-z]{2}-\d+(?:\.\d+)?(?:\(\d+\))?"  # control ID: ac-1, au-6.1, au-6(1)
    r"|"
    r"\b\w\w+\b"  # regular word: policy, procedures
)


class _Retriever(Protocol):
    def retrieve(self, query: str, *, k_recall: int) -> list[Retrieved]: ...


@dataclass
class BM25Retriever:
    """Lexical retrieval (bm25s) over the same Index's chunks.

    The index's vectors are unused here; the shared index exists so
    hybrid fusion can address identical chunks by ``chunk_id``.

    ``token_pattern`` keeps control IDs ("ac-1", "au-6(1)") as single
    tokens so BM25 can match them exactly, instead of splitting into
    common subtokens ("ac", "1") that flood the candidate set.
    """

    chunks: list  # list[Chunk] — typed loose to avoid circular import
    k1: float = 1.5
    b: float = 0.75
    stopwords: str | None = "en"
    token_pattern: str = _CONTROL_ID_PATTERN
    source: str = "bm25"

    def __post_init__(self) -> None:
        import bm25s

        self._bm25s = bm25s
        corpus_tokens = bm25s.tokenize(
            [chunk.text for chunk in self.chunks],
            stopwords=self.stopwords,
            token_pattern=self.token_pattern,
            show_progress=False,
        )
        self._bm25 = bm25s.BM25(k1=self.k1, b=self.b)
        self._bm25.index(corpus_tokens, show_progress=False)

    def retrieve(self, query: str, *, k_recall: int) -> list[Retrieved]:
        """BM25 top-k retrieval."""
        chunks = self.chunks
        k = min(k_recall, len(chunks))
        query_tokens = self._bm25s.tokenize(
            query,
            stopwords=self.stopwords,
            token_pattern=self.token_pattern,
            show_progress=False,
        )
        results, scores = self._bm25.retrieve(query_tokens, k=k, show_progress=False)

        out: list[Retrieved] = []
        for ix, score in zip(results[0], scores[0], strict=True):
            chunk = chunks[int(ix)]
            out.append(
                Retrieved(
                    chunk=chunk,
                    score=float(score),
                    rank=len(out) + 1,
                    source=self.source,
                )
            )
        return out


@dataclass
class RRFFusion:
    """Hybrid retrieval as reciprocal rank fusion over N legs.

    Each leg retrieves ``k_recall`` candidates; RRF scores each by
    ``weight / (k_rrf + rank)`` and sums across legs. ``weights``
    biases toward stronger legs — e.g. [2, 1] gives the first leg 2x
    the influence of the second. The fused list is sorted by total
    RRF score, ties broken on ``chunk_id``.
    """

    retrievers: list[_Retriever]
    k_rrf: int = 60
    weights: list[float] | None = None

    def __post_init__(self) -> None:
        if len(self.retrievers) < 2:
            raise ValueError("RRF needs at least two legs")
        if self.weights is None:
            self.weights = [1.0] * len(self.retrievers)
        if len(self.weights) != len(self.retrievers):
            raise ValueError(
                f"weights ({len(self.weights)}) must match retrievers ({len(self.retrievers)})"
            )

    def retrieve(self, query: str, *, k_recall: int) -> list[Retrieved]:
        """Fuse retrieval legs by reciprocal rank."""
        legs = [leg.retrieve(query, k_recall=k_recall) for leg in self.retrievers]
        return rrf_fuse(legs, k_rrf=self.k_rrf, weights=self.weights, k=k_recall)


class _CachedBM25:
    """Memoize the BM25 index per chunk set.

    ``hybrid_retrieve`` used to build a fresh BM25 index on every call — a
    measurement over N queries rebuilt the 2k-document index N times. Same
    build, reused. Keyed on the identity and length of the chunk list so a
    changed corpus rebuilds.
    """

    _cache: ClassVar[dict[tuple[int, int], BM25Retriever]] = {}

    @classmethod
    def get(cls, chunks: list) -> BM25Retriever:
        key = (id(chunks), len(chunks))
        retriever = cls._cache.get(key)
        if retriever is None:
            retriever = BM25Retriever(chunks=chunks)
            cls._cache[key] = retriever
        return retriever


class _DenseAdapter:
    """Wrap (index, embedder) to satisfy the _Retriever protocol."""

    def __init__(self, index: object, embedder: object) -> None:
        self._index = index
        self._embedder = embedder
        self.source = "dense"

    def retrieve(self, query: str, *, k_recall: int) -> list[Retrieved]:
        from rag_control_cite.retrieval import retrieve

        q_vec = self._embedder.embed([query])[0]
        return retrieve(self._index, q_vec, k=k_recall)


def hybrid_retrieve(
    index: object,
    embedder: object,
    query: str,
    k: int = 10,
    k_rrf: int = 60,
    weights: list[float] | None = None,
    k_leg: int | None = None,
) -> list[Retrieved]:
    """Hybrid dense + BM25 with reciprocal rank fusion.

    Args:
        index: an ``Index`` with ``chunks`` and ``retrieve(query_vec, k)``.
        embedder: an object with ``embed(texts) -> ndarray``.
        query: the query string.
        k: number of candidates to return.
        k_rrf: RRF smoothing constant.
        weights: per-leg weights. None = equal weight. [2, 1] biases
                 toward the dense leg.

    Returns:
        Fused Retrieved list, 1-based rank, sorted by RRF score.
    """
    dense = _DenseAdapter(index, embedder)
    bm25 = _CachedBM25.get(index.chunks)  # type: ignore[attr-defined]
    fusion = RRFFusion(retrievers=[dense, bm25], k_rrf=k_rrf, weights=weights)
    fused = fusion.retrieve(query, k_recall=k_leg or k)
    return fused[:k]
