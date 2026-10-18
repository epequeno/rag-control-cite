"""ACL placement: pre-filter SQL, fused parity, reranker enforcement."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from rag_control_cite.acl import can_access
from rag_control_cite.hybrid import rrf_fuse
from rag_control_cite.models import Chunk, Retrieved, Tenant
from rag_control_cite.store import VectorStore, acl_sql


def _chunk(cid: str, tags: frozenset[str]) -> Chunk:
    return Chunk(
        chunk_id=cid,
        source_uri=f"t://{cid}",
        text=f"{cid}: text",
        start_char=0,
        end_char=5,
        chunk_index=0,
        doc_type="x",
        access_tags=tags,
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        control_id=cid,
    )


def _retrieved(cid: str, rank: int, tags: frozenset[str] | None = None) -> Retrieved:
    return Retrieved(chunk=_chunk(cid, tags or frozenset()), score=1.0 / rank, rank=rank)


def test_acl_sql_restricted_dims() -> None:
    tenant = Tenant(
        "t",
        frozenset({"sp-800-53"}),
        None,  # family wildcard → omitted
        frozenset({"low"}),
        frozenset({"security"}),
    )
    where, params = acl_sql(tenant)
    assert where.count("access_tags && %s::text[]") == 3  # framework, baseline, audience
    assert ["framework:sp-800-53"] in params
    assert ["baseline:low"] in params
    assert all("family:" not in p[0] for p in params)


def test_acl_sql_wildcard_all() -> None:
    where, _params = acl_sql(Tenant("t", None, None, None, None))
    assert where == "TRUE"
    assert _params == []


def test_acl_sql_matches_python_policy_for_fixture() -> None:
    """The SQL clauses encode exactly the same rule as can_access."""
    tenant = Tenant(
        "t",
        frozenset({"sp-800-53"}),
        frozenset({"ac"}),
        frozenset({"low"}),
        frozenset({"security"}),
    )
    where, _params = acl_sql(tenant)
    assert where.count("&&") == 4  # four restricted dimensions

    allowed = [frozenset({"framework:sp-800-53", "family:ac", "baseline:low", "audience:security"})]
    blocked = [
        frozenset({"framework:sp-800-171", "family:ac", "baseline:low", "audience:security"}),
        frozenset({"framework:sp-800-53", "family:au", "baseline:low", "audience:security"}),
        frozenset({"framework:sp-800-53", "family:ac", "baseline:high", "audience:security"}),
        frozenset({"framework:sp-800-53", "family:ac", "baseline:low", "audience:privacy"}),
        frozenset({"framework:sp-800-53"}),  # missing dims
        frozenset(),
    ]
    assert all(can_access(tags, tenant) for tags in allowed)
    assert not any(can_access(tags, tenant) for tags in blocked)


def test_retrieve_hybrid_uses_shared_fuse() -> None:
    """Served path fuses by the same rrf_fuse the measured path uses."""
    store = object.__new__(VectorStore)  # no DB connection
    dense = [_retrieved("a", 1), _retrieved("b", 2), _retrieved("c", 3)]
    bm25 = [_retrieved("c", 1), _retrieved("d", 2)]
    store.retrieve_dense = lambda qv, k=50, tenant=None: list(dense)
    store.retrieve_bm25 = lambda q, k=50, tenant=None: list(bm25)

    got = store.retrieve_hybrid("q", None, k=10)
    expected = rrf_fuse([dense, bm25], k_rrf=60, k=10)
    assert [r.chunk.chunk_id for r in got] == [r.chunk.chunk_id for r in expected]
    assert [r.rank for r in got] == [r.rank for r in expected]


def test_reranker_requires_tenant() -> None:
    from rag_control_cite.rerank import CrossEncoderReranker

    reranker = object.__new__(CrossEncoderReranker)
    with pytest.raises(TypeError):
        reranker.rerank("q", [_retrieved("a", 1)])  # type: ignore[call-arg]


def test_reranker_output_is_entitled_only() -> None:
    from rag_control_cite.rerank import CrossEncoderReranker

    tenant = Tenant(
        "t", frozenset({"sp-800-53"}), None, frozenset({"low"}), frozenset({"security"})
    )

    reranker = object.__new__(CrossEncoderReranker)
    reranker.batch_size = 32

    class _Model:
        def predict(self, pairs, batch_size, show_progress_bar):
            return np.asarray([len(p[1]) for p in pairs], dtype=float)

    reranker._model = _Model()

    candidates = [
        _retrieved(
            "ok", 1, frozenset({"framework:sp-800-53", "baseline:low", "audience:security"})
        ),
        _retrieved(
            "leak", 2, frozenset({"framework:sp-800-171", "baseline:high", "audience:privacy"})
        ),
    ]
    out = reranker.rerank("q", candidates, tenant, k_final=5)
    assert [r.chunk.chunk_id for r in out] == ["ok"]
    assert all(can_access(r.chunk.access_tags, tenant) for r in out)
