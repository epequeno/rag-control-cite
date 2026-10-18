"""Served path (store) and measured path (RRFFusion) rank identically."""

from __future__ import annotations

from datetime import UTC, datetime

from rag_control_cite.hybrid import RRFFusion, rrf_fuse
from rag_control_cite.models import Chunk, Retrieved
from rag_control_cite.store import VectorStore


def _r(cid: str, rank: int) -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=cid,
            source_uri=f"t://{cid}",
            text=cid,
            start_char=0,
            end_char=1,
            chunk_index=0,
            doc_type="x",
            access_tags=frozenset({"framework:sp-800-53"}),
            last_modified=datetime(2026, 1, 1, tzinfo=UTC),
            control_id=cid,
        ),
        score=1.0 / rank,
        rank=rank,
    )


class _Leg:
    def __init__(self, items: list[Retrieved]) -> None:
        self.items = items

    def retrieve(self, query: str, *, k_recall: int) -> list[Retrieved]:
        return self.items[:k_recall]


def test_rrf_fusion_equals_shared_fuse() -> None:
    leg_a = [_r("a", 1), _r("b", 2), _r("c", 3)]
    leg_b = [_r("c", 1), _r("d", 2), _r("a", 3)]
    fusion = RRFFusion(retrievers=[_Leg(leg_a), _Leg(leg_b)])
    got = fusion.retrieve("q", k_recall=10)
    expected = rrf_fuse([leg_a, leg_b], k_rrf=60, k=10)
    assert [(x.chunk.chunk_id, x.rank) for x in got] == [
        (x.chunk.chunk_id, x.rank) for x in expected
    ]


def test_store_and_measured_path_agree_on_fixtures() -> None:
    leg_a = [_r("a", 1), _r("b", 2), _r("c", 3)]
    leg_b = [_r("c", 1), _r("d", 2), _r("a", 3)]

    store = object.__new__(VectorStore)
    store.retrieve_dense = lambda qv, k=50, tenant=None: list(leg_a)
    store.retrieve_bm25 = lambda q, k=50, tenant=None: list(leg_b)
    served = store.retrieve_hybrid("q", None, k=10)

    fusion = RRFFusion(retrievers=[_Leg(leg_a), _Leg(leg_b)])
    measured = fusion.retrieve("q", k_recall=10)  # k_leg=10 for the fixture

    assert [(x.chunk.chunk_id, x.rank) for x in served] == [
        (x.chunk.chunk_id, x.rank) for x in measured
    ]


def test_rrf_weights_have_effect_at_large_k_leg() -> None:
    """A weight change can reorder — at k_leg=10 it cannot (the A2-7 artefact)."""
    leg_dense = [_r(f"d{i}", i) for i in range(1, 11)]
    leg_bm25 = [_r(f"b{i}", i) for i in range(1, 11)]
    equal = rrf_fuse([leg_dense, leg_bm25], k_rrf=60, weights=[1.0, 1.0], k=10)
    dense_heavy = rrf_fuse([leg_dense, leg_bm25], k_rrf=60, weights=[3.0, 1.0], k=10)
    assert [x.chunk.chunk_id for x in equal] != [x.chunk.chunk_id for x in dense_heavy]
