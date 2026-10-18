"""Stage 1/2/3 measurements — recall@k, nDCG@k, latency, cost, CIs.

The numbers the README leads with: recall@10, nDCG@5, groundedness (with
a per-query bootstrap 95% CI), p50/p95 latency, and $/1k at a fixed
quality floor. Cost comes from the one price table (``pricing.py``), not
constants inline here.

Deltas (hybrid vs dense, rerank vs none) use paired bootstrap intervals,
so a comparison carries its own uncertainty rather than a bare point
estimate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag_control_cite.models import Retrieved
from rag_control_cite.pricing import UnknownModelError, cost_usd

__all__ = [
    "Stage1Report",
    "Stage3Report",
    "bootstrap_ci",
    "measure_hybrid_retrieval",
    "measure_rerank",
    "measure_retrieval",
    "ndcg_at_k",
    "paired_bootstrap_delta",
    "recall_at_k",
    "rrf_weight_sweep",
]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def bootstrap_ci(
    values: list[float], confidence: float = 0.95, n_resamples: int = 10_000
) -> tuple[float, float]:
    """Per-query bootstrap 95% CI for a mean. Returns (lo, hi)."""
    import numpy as np

    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(42)
    means = np.array(
        [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)]
    )
    alpha = (1 - confidence) / 2
    return (
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1 - alpha))),
    )


def paired_bootstrap_delta(
    a: list[float], b: list[float], confidence: float = 0.95, n_resamples: int = 10_000
) -> tuple[float, float, float]:
    """Paired bootstrap CI for mean(a) - mean(b). Returns (delta, lo, hi).

    ``a`` and ``b`` are per-query scores in the same query order.
    """
    import numpy as np

    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and equal length")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(42)
    means = np.array(
        [np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(n_resamples)]
    )
    alpha = (1 - confidence) / 2
    return (
        float(np.mean(diff)),
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1 - alpha))),
    )


@dataclass
class Stage1Report:
    """Stage 1/2 metrics — recall@k, latency, cost."""

    n_queries: int = 0
    recall_scores: list[float] = field(default_factory=list)
    groundedness_scores: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def recall_at_10(self) -> float:
        if not self.recall_scores:
            return 0.0
        return sum(self.recall_scores) / len(self.recall_scores)

    @property
    def groundedness(self) -> float:
        if not self.groundedness_scores:
            return 0.0
        return sum(self.groundedness_scores) / len(self.groundedness_scores)

    @property
    def p50_ms(self) -> float:
        return _percentile(self.latencies_ms, 50)

    @property
    def p95_ms(self) -> float:
        return _percentile(self.latencies_ms, 95)

    @property
    def cost_per_1k(self) -> float:
        """Estimated $/1k queries from recorded usage at the model's price."""
        if self.n_queries == 0 or not self.model:
            return 0.0
        try:
            est_total = cost_usd(self.prompt_tokens, self.completion_tokens, self.model)
        except UnknownModelError:
            return 0.0
        return est_total / self.n_queries * 1000

    def as_dict(self) -> dict[str, float]:
        recall_lo, recall_hi = bootstrap_ci(self.recall_scores)
        return {
            "n_queries": self.n_queries,
            "recall@10": round(self.recall_at_10, 4),
            "recall@10_ci_lo": round(recall_lo, 4),
            "recall@10_ci_hi": round(recall_hi, 4),
            "groundedness": round(self.groundedness, 4),
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "$/1k": round(self.cost_per_1k, 4),
        }


def recall_at_k(retrieved: list[Retrieved], gold_ids: frozenset[str], k: int = 10) -> float:
    """Recall@k: fraction of gold chunks in the top-k retrieved."""
    if not gold_ids:
        return 0.0
    top_k_ids = {r.chunk.chunk_id for r in retrieved[:k]}
    hits = len(top_k_ids & gold_ids)
    return hits / len(gold_ids)


def _log2(x: float) -> float:
    import math

    return math.log2(x)


def ndcg_at_k(retrieved: list[Retrieved], gold_ids: frozenset[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain @ k (binary relevance)."""
    if not gold_ids or not retrieved:
        return 0.0
    rels = [1.0 if r.chunk.chunk_id in gold_ids else 0.0 for r in retrieved[:k]]
    dcg = sum(rel / _log2(rank + 1) for rank, rel in enumerate(rels, start=1))
    n_relevant = min(len(gold_ids), k)
    idcg = sum(1.0 / _log2(rank + 1) for rank in range(1, n_relevant + 1))
    return dcg / idcg if idcg > 0 else 0.0


def measure_retrieval(
    index: object,
    embedder: object,
    queries: list[object],
    k: int = 10,
) -> Stage1Report:
    """Run dense retrieval queries and measure recall@k + latency."""
    report = Stage1Report()
    for q in queries:
        t0 = time.perf_counter()
        q_vec = embedder.embed([q.query])[0]  # type: ignore[attr-defined]
        retrieved = _retrieve(index, q_vec, k)
        t1 = time.perf_counter()

        report.n_queries += 1
        report.latencies_ms.append((t1 - t0) * 1000)
        report.recall_scores.append(
            recall_at_k(retrieved, q.gold_chunk_ids, k)  # type: ignore[attr-defined]
        )
    return report


def _retrieve(index: object, query_vec: object, k: int) -> list[Retrieved]:
    from rag_control_cite.retrieval import retrieve

    return retrieve(index, query_vec, k)


def measure_hybrid_retrieval(
    index: object,
    embedder: object,
    queries: list[object],
    k: int = 10,
    k_rrf: int = 60,
    weights: list[float] | None = None,
) -> Stage1Report:
    """Run hybrid (dense + BM25, RRF) queries and measure recall@k + latency."""
    from rag_control_cite.hybrid import hybrid_retrieve

    report = Stage1Report()
    for q in queries:
        t0 = time.perf_counter()
        retrieved = hybrid_retrieve(index, embedder, q.query, k=k, k_rrf=k_rrf, weights=weights)
        t1 = time.perf_counter()

        report.n_queries += 1
        report.latencies_ms.append((t1 - t0) * 1000)
        report.recall_scores.append(
            recall_at_k(retrieved, q.gold_chunk_ids, k)  # type: ignore[attr-defined]
        )
    return report


def rrf_weight_sweep(
    index: object,
    embedder: object,
    queries: list[object],
    weights: list[list[float]],
    k_leg: int = 50,
    k: int = 10,
) -> dict[tuple[float, ...], float]:
    """Recall@k for each RRF weight setting, with legs returning k_leg.

    ``k_leg=50`` (not 10): at k_leg=10 and dense weight ≥ 1.15, BM25-only
    candidates cannot enter the top 10 with k_rrf=60, so any weight ≥ 2
    collapses to the dense baseline by arithmetic. A sweep that cannot
    move the output proves nothing.
    """
    from rag_control_cite.hybrid import BM25Retriever, RRFFusion, _DenseAdapter

    dense = _DenseAdapter(index, embedder)
    bm25 = BM25Retriever(chunks=index.chunks)  # type: ignore[attr-defined]
    out: dict[tuple[float, ...], float] = {}
    for weight in weights:
        fusion = RRFFusion(retrievers=[dense, bm25], k_rrf=60, weights=list(weight))
        scores: list[float] = []
        for q in queries:
            fused = fusion.retrieve(q.query, k_recall=k_leg)[:k]
            scores.append(recall_at_k(fused, q.gold_chunk_ids, k))  # type: ignore[attr-defined]
        out[tuple(weight)] = sum(scores) / len(scores) if scores else 0.0
    return out


@dataclass
class Stage3Report:
    """Stage 3 metrics — nDCG@5, latency, cost."""

    n_queries: int = 0
    ndcg_scores: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    groundedness_scores: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def ndcg_at_5(self) -> float:
        if not self.ndcg_scores:
            return 0.0
        return sum(self.ndcg_scores) / len(self.ndcg_scores)

    @property
    def groundedness(self) -> float:
        if not self.groundedness_scores:
            return 0.0
        return sum(self.groundedness_scores) / len(self.groundedness_scores)

    @property
    def p50_ms(self) -> float:
        return _percentile(self.latencies_ms, 50)

    @property
    def p95_ms(self) -> float:
        return _percentile(self.latencies_ms, 95)

    @property
    def cost_per_1k(self) -> float:
        if self.n_queries == 0 or not self.model:
            return 0.0
        try:
            est_total = cost_usd(self.prompt_tokens, self.completion_tokens, self.model)
        except UnknownModelError:
            return 0.0
        return est_total / self.n_queries * 1000

    def as_dict(self) -> dict[str, float]:
        lo, hi = bootstrap_ci(self.ndcg_scores)
        return {
            "n_queries": self.n_queries,
            "ndcg@5": round(self.ndcg_at_5, 4),
            "ndcg@5_ci_lo": round(lo, 4),
            "ndcg@5_ci_hi": round(hi, 4),
            "groundedness": round(self.groundedness, 4),
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "$/1k": round(self.cost_per_1k, 4),
        }


def measure_rerank(
    index: object,
    embedder: object,
    queries: list[object],
    k_recall: int = 50,
    k_final: int = 5,
    weights: list[float] | None = None,
) -> Stage3Report:
    """Hybrid retrieval (top-50) → cross-encoder rerank (top-5) → measure.

    Reranking runs under the evaluation tenant (unrestricted) — this path
    measures ranking quality, not access. The same candidate pool feeds
    the no-rerank baseline (hybrid top-50 truncated to 5) so the
    comparison is on equal pools.
    """
    from rag_control_cite.hybrid import hybrid_retrieve
    from rag_control_cite.rerank import CrossEncoderReranker
    from rag_control_cite.tenants import EVAL_TENANT

    reranker = CrossEncoderReranker()
    report = Stage3Report()

    for q in queries:
        t0 = time.perf_counter()
        candidates = hybrid_retrieve(
            index,
            embedder,
            q.query,
            k=k_recall,
            weights=weights,  # type: ignore[attr-defined]
        )
        reranked = reranker.rerank(q.query, candidates, EVAL_TENANT, k_final=k_final)
        t1 = time.perf_counter()

        report.n_queries += 1
        report.latencies_ms.append((t1 - t0) * 1000)
        report.ndcg_scores.append(
            ndcg_at_k(reranked, q.gold_chunk_ids, k_final)  # type: ignore[attr-defined]
        )
    return report
