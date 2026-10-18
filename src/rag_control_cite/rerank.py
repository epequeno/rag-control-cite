"""Stage 3 — retrieve-then-rerank: cross-encoder, top-50 → top-5.

A cross-encoder adds +33–40% retrieval accuracy for ~120 ms. The
reranker reorders the hybrid retrieval output (top-50 candidates) and
truncates to top-5, preserving the ``source`` from the retrieval leg.

Ends in: Δ nDCG@5, added p95 ms, Δ $/1k.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag_control_cite.models import Retrieved, Tenant

__all__ = ["CrossEncoderReranker", "rerank"]


@dataclass
class CrossEncoderReranker:
    """Cross-encoder reranker via sentence-transformers.

    ``score`` is the model's raw logit (not a probability and not
    comparable across models) — only the induced order is meaningful.
    ``source`` is preserved from the retrieval leg so the reranked
    output still says which leg found the chunk.
    """

    model_name: str = ""
    batch_size: int = 32
    device: str | None = None

    def __post_init__(self) -> None:
        from sentence_transformers import CrossEncoder

        if not self.model_name:
            from rag_control_cite.config import settings

            self.model_name = settings.reranker_model
        self._model = CrossEncoder(self.model_name, device=self.device)

    @property
    def id(self) -> str:
        return f"ce:{self.model_name}"

    def rerank(
        self,
        query: str,
        candidates: list[Retrieved],
        tenant: Tenant,
        k_final: int = 5,
    ) -> list[Retrieved]:
        """Rerank candidates by cross-encoder score, truncate to k_final.

        The tenant is required and enforcement runs here, so a caller
        cannot rerank an unfiltered candidate set — the reranker's output
        is always a subset of the tenant's entitled chunks.
        """
        from rag_control_cite.acl import enforce_acl

        entitled = enforce_acl(candidates, tenant)
        if not entitled:
            return []
        scores = np.asarray(
            self._model.predict(
                [(query, item.chunk.text) for item in entitled],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        )
        order = np.argsort(-scores, kind="stable")[:k_final]
        return [
            Retrieved(
                chunk=entitled[int(i)].chunk,
                score=float(scores[int(i)]),
                rank=rank,
                source=entitled[int(i)].source,
            )
            for rank, i in enumerate(order, start=1)
        ]


def rerank(
    query: str,
    candidates: list[Retrieved],
    tenant: Tenant,
    top_k: int = 5,
    model_name: str = "",
) -> list[Retrieved]:
    """Cross-encoder rerank: top-50 → top-5, tenant-enforced.

    Args:
        query: the query string.
        candidates: retrieved candidates (typically top-50 from hybrid).
        tenant: the caller's tenant — enforcement runs inside the reranker.
        top_k: number of candidates to return after reranking.
        model_name: cross-encoder model. Defaults to config.

    Returns:
        Reranked Retrieved list, 1-based rank, sorted by cross-encoder score.
    """
    reranker = CrossEncoderReranker(model_name=model_name)
    return reranker.rerank(query, candidates, tenant, k_final=top_k)
