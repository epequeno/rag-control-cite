"""Smoke test: the package and every stage module imports."""

import importlib

import pytest

STAGE_MODULES = [
    "rag_control_cite",
    "rag_control_cite.cli",
    "rag_control_cite.config",
    "rag_control_cite.models",
    "rag_control_cite.tenants",
    "rag_control_cite.ingestion",
    "rag_control_cite.retrieval",
    "rag_control_cite.hybrid",
    "rag_control_cite.rerank",
    "rag_control_cite.acl",
    "rag_control_cite.generation",
    "rag_control_cite.eval_gate",
    "rag_control_cite.observability",
    "rag_control_cite.deploy",
    "rag_control_cite.embedder",
    "rag_control_cite.index",
    "rag_control_cite.query_set",
    "rag_control_cite.judge",
    "rag_control_cite.measure",
]


@pytest.mark.parametrize("mod_name", STAGE_MODULES)
def test_imports(mod_name: str) -> None:
    """Every stage module imports without error."""
    importlib.import_module(mod_name)


def test_tenants_load() -> None:
    """Synthetic tenants load with the expected dimensions."""
    from rag_control_cite.tenants import TENANTS

    assert len(TENANTS) >= 3
    for t in TENANTS.values():
        assert t.tenant_id
        assert t.frameworks
        assert t.baseline_tiers
        assert t.audiences


def test_cli_dispatches() -> None:
    """All CLI subcommands are implemented."""
    from rag_control_cite import cli

    assert callable(cli._ingest)
    assert callable(cli._demo)
    assert callable(cli._eval)


def test_ingest_stage0() -> None:
    """Stage 0 ingestion: parse → chunk → measure against the OSCAL corpus."""
    from pathlib import Path

    from rag_control_cite.ingestion import ingest

    corpus = Path("data/corpus/oscal")
    if not corpus.is_dir():
        pytest.skip("OSCAL corpus not fetched — run: bash scripts/fetch_corpus.sh")

    chunks, report = ingest()

    assert report.files_attempted == 3
    assert report.files_failed == 0
    assert report.parse_failure_rate == 0.0
    assert report.total_controls > 0
    assert report.table_controls > 0
    assert report.total_documents > 0
    assert len(chunks) > 0
    assert report.char_sanity_total == len(chunks)
    assert report.char_sanity_passed == report.char_sanity_total
    assert report.char_sanity_pass_pct == 1.0

    # Every chunk carries provenance
    for ch in chunks[:20]:
        assert ch.chunk_id
        assert ch.source_uri
        assert ch.doc_type == "oscal-json"
        assert ch.access_tags
        assert ch.last_modified
        assert ch.start_char >= 0
        assert ch.end_char > ch.start_char


def test_chunker_char_sanity() -> None:
    """Chunker preserves text exactly: chunk.text == body[start:end]."""
    from datetime import UTC, datetime

    from rag_control_cite.ingestion import RecursiveChunker
    from rag_control_cite.models import Document

    doc = Document(
        doc_id="test-1",
        source_uri="test://1",
        text="A. " * 600,  # 2400 chars — forces multiple chunks
        title="Test",
        doc_type="oscal-json",
        access_tags=frozenset({"framework:test"}),
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
    )
    chunker = RecursiveChunker(chunk_size=256, chunk_overlap=32)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    for ch in chunks:
        assert ch.text == doc.text[ch.start_char : ch.end_char]


def test_baseline_tier_tags() -> None:
    """SP 800-53 controls get baseline tier access_tags from profiles."""
    from pathlib import Path

    from rag_control_cite.ingestion import ingest

    corpus = Path("data/corpus/oscal")
    if not corpus.is_dir():
        pytest.skip("OSCAL corpus not fetched")

    chunks, _ = ingest()
    sp53 = [ch for ch in chunks if "framework:sp-800-53" in ch.access_tags]
    assert len(sp53) > 0

    # AC-2 (Account Management) is in all baselines
    ac2 = [ch for ch in sp53 if "family:ac" in ch.access_tags]
    assert len(ac2) > 0

    # At least one chunk should have a high baseline tag
    high = [ch for ch in sp53 if "baseline:high" in ch.access_tags]
    assert len(high) > 0


def test_query_set_generation() -> None:
    """Exact-term query set generates from OSCAL control IDs."""
    from pathlib import Path

    from rag_control_cite.ingestion import ingest
    from rag_control_cite.query_set import generate_exact_term_queries

    corpus = Path("data/corpus/oscal")
    if not corpus.is_dir():
        pytest.skip("OSCAL corpus not fetched")

    chunks, _ = ingest()
    queries = generate_exact_term_queries(chunks)

    assert len(queries) > 0
    categories = {q.category for q in queries}
    assert "by_id" in categories
    assert "by_name" in categories
    assert "by_family" in categories

    for q in queries:
        assert q.query
        assert q.gold_chunk_ids
        assert q.control_id


def test_index_build_and_retrieve() -> None:
    """Index builds, caches, and retrieves by cosine similarity."""
    import tempfile
    from datetime import UTC, datetime

    # Use a dummy embedder that returns deterministic vectors
    import numpy as np

    from rag_control_cite.index import build_or_load_index
    from rag_control_cite.models import Chunk
    from rag_control_cite.retrieval import retrieve

    class DummyEmbedder:
        id = "dummy:test"

        def embed(self, texts: list[str]) -> np.ndarray:
            # Simple deterministic embedding: hash of text → 8-dim vector
            import hashlib

            rows = []
            for t in texts:
                h = hashlib.md5(t.encode()).hexdigest()
                vec = [int(h[i * 2 : i * 2 + 2], 16) / 255 for i in range(8)]
                rows.append(vec)
            vec = np.asarray(rows, dtype=np.float32)
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            return vec / np.maximum(norms, 1e-12)

    chunks = [
        Chunk(
            chunk_id=f"test-{i}",
            source_uri=f"test://{i}",
            text=f"AC-{i}: Account Management test number {i}",
            start_char=0,
            end_char=40,
            chunk_index=i,
            doc_type="oscal-json",
            access_tags=frozenset({"framework:sp-800-53", "family:ac"}),
            last_modified=datetime(2026, 1, 1, tzinfo=UTC),
            title=f"AC-{i}",
        )
        for i in range(5)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        index = build_or_load_index(chunks, DummyEmbedder(), cache_dir=tmp)
        assert index.vectors.shape == (5, 8)

        # Retrieve with the same dummy embedder
        embedder = DummyEmbedder()
        q_vec = embedder.embed([chunks[0].text])[0]
        results = retrieve(index, q_vec, k=3)
        assert len(results) == 3
        assert results[0].rank == 1
        # The exact match should be in top results
        top_ids = {r.chunk.chunk_id for r in results[:2]}
        assert "test-0" in top_ids


def test_ndcg_at_k() -> None:
    """nDCG@5: perfect ranking scores 1.0, mixed ranking scores between 0 and 1."""
    from datetime import UTC, datetime

    from rag_control_cite.measure import ndcg_at_k
    from rag_control_cite.models import Chunk, Retrieved

    def make_retrieved(cid: str) -> Retrieved:
        return Retrieved(
            chunk=Chunk(
                chunk_id=cid,
                source_uri=f"test://{cid}",
                text=cid,
                start_char=0,
                end_char=10,
                chunk_index=0,
                doc_type="oscal-json",
                access_tags=frozenset(),
                last_modified=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            score=1.0,
            rank=1,
        )

    gold = frozenset({"a", "b"})

    # Perfect: both gold at top
    perfect = [
        make_retrieved("a"),
        make_retrieved("b"),
        make_retrieved("c"),
        make_retrieved("d"),
        make_retrieved("e"),
    ]
    assert ndcg_at_k(perfect, gold, k=5) == 1.0

    # Worst: no gold in top-5
    worst = [
        make_retrieved("c"),
        make_retrieved("d"),
        make_retrieved("e"),
        make_retrieved("f"),
        make_retrieved("g"),
    ]
    assert ndcg_at_k(worst, gold, k=5) == 0.0

    # Mixed: one gold at rank 1, one at rank 3
    mixed = [
        make_retrieved("a"),
        make_retrieved("c"),
        make_retrieved("b"),
        make_retrieved("d"),
        make_retrieved("e"),
    ]
    score = ndcg_at_k(mixed, gold, k=5)
    assert 0.0 < score < 1.0

    # Empty
    assert ndcg_at_k([], gold, k=5) == 0.0
    assert ndcg_at_k(perfect, frozenset(), k=5) == 0.0


def test_acl_can_access() -> None:
    """can_access: tenant sees allowed tags, blocked by disallowed dimensions."""
    from rag_control_cite.acl import can_access
    from rag_control_cite.models import Tenant

    tenant = Tenant(
        tenant_id="test",
        frameworks=frozenset({"sp-800-53"}),
        control_families=frozenset({"ac", "au"}),
        baseline_tiers=frozenset({"low"}),
        audiences=frozenset({"security"}),
    )

    # Allowed: framework matches, family matches, baseline matches, audience matches
    allowed_tags = frozenset(
        {
            "framework:sp-800-53",
            "family:ac",
            "baseline:low",
            "audience:security",
        }
    )
    assert can_access(allowed_tags, tenant) is True

    # Blocked: wrong framework
    assert can_access(frozenset({"framework:sp-800-171", "audience:security"}), tenant) is False

    # Blocked: wrong baseline
    assert (
        can_access(frozenset({"framework:sp-800-53", "baseline:high", "audience:security"}), tenant)
        is False
    )

    # Blocked: wrong audience
    assert can_access(frozenset({"framework:sp-800-53", "audience:privacy"}), tenant) is False

    # Blocked: family not in tenant's set
    assert (
        can_access(frozenset({"framework:sp-800-53", "family:ia", "audience:security"}), tenant)
        is False
    )

    # Wildcard family (None): any family passes, but restricted dimensions
    # (baseline, audience) stay fail-closed — a missing tag is denied.
    wildcard = Tenant(
        tenant_id="wild",
        frameworks=frozenset({"sp-800-53"}),
        control_families=None,
        baseline_tiers=frozenset({"low"}),
        audiences=frozenset({"security"}),
    )
    assert (
        can_access(
            frozenset(
                {
                    "framework:sp-800-53",
                    "family:zz",
                    "baseline:low",
                    "audience:security",
                }
            ),
            wildcard,
        )
        is True
    )
    # Missing baseline tag for a restricted baseline dimension → denied.
    assert (
        can_access(frozenset({"framework:sp-800-53", "family:zz", "audience:security"}), wildcard)
        is False
    )
    # Zero-tag chunk → denied for any finite-dimension tenant.
    assert can_access(frozenset(), wildcard) is False


def test_acl_enforce_filters() -> None:
    """enforce_acl: filters out unauthorized chunks and re-ranks."""
    from datetime import UTC, datetime

    from rag_control_cite.acl import enforce_acl
    from rag_control_cite.models import Chunk, Retrieved, Tenant

    tenant = Tenant(
        tenant_id="test",
        frameworks=frozenset({"sp-800-53"}),
        control_families=None,
        baseline_tiers=frozenset({"low"}),
        audiences=frozenset({"security"}),
    )

    def make_chunk(cid: str, tags: frozenset[str]) -> Chunk:
        return Chunk(
            chunk_id=cid,
            source_uri=f"test://{cid}",
            text=cid,
            start_char=0,
            end_char=10,
            chunk_index=0,
            doc_type="oscal-json",
            access_tags=tags,
            last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        )

    retrieved = [
        Retrieved(
            chunk=make_chunk(
                "ok",
                frozenset(
                    {
                        "framework:sp-800-53",
                        "baseline:low",
                        "audience:security",
                    }
                ),
            ),
            score=0.9,
            rank=1,
        ),
        Retrieved(
            chunk=make_chunk(
                "leak",
                frozenset(
                    {
                        "framework:sp-800-53",
                        "baseline:high",
                        "audience:security",
                    }
                ),
            ),
            score=0.8,
            rank=2,
        ),
    ]

    filtered = enforce_acl(retrieved, tenant)
    assert len(filtered) == 1
    assert filtered[0].chunk.chunk_id == "ok"
    assert filtered[0].rank == 1  # re-ranked


def test_eval_baseline_roundtrip() -> None:
    """EvalBaseline saves and loads round-trip."""
    import tempfile
    from pathlib import Path

    from rag_control_cite.eval_gate import EvalBaseline

    baseline = EvalBaseline(
        metrics={"recall@10": 0.70, "ndcg@5": 0.60, "groundedness": 1.0},
        ci_lower={"recall@10": 0.66, "ndcg@5": 0.55, "groundedness": 0.9},
        results={"by_id:ac-1": True, "by_name:au-2": False},
    )

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        path = Path(f.name)

    try:
        baseline.save(path)
        loaded = EvalBaseline.load(path)
        assert loaded is not None
        assert loaded.metrics == baseline.metrics
        assert loaded.ci_lower == baseline.ci_lower
        assert loaded.results == baseline.results
    finally:
        path.unlink(missing_ok=True)


def test_gate_config_defaults() -> None:
    """GateConfig has sensible default thresholds."""
    from rag_control_cite.eval_gate import GateConfig

    config = GateConfig()
    assert "recall@10" in config.thresholds
    assert "ndcg@5" in config.thresholds
    assert "groundedness" in config.thresholds
    # None = evaluate the whole query set (the reported count is the real one).
    assert config.n_eval_queries is None


def test_trace_recorder() -> None:
    """TraceRecorder accumulates stages and computes summary."""
    from rag_control_cite.observability import TraceRecorder, stage_span

    recorder = TraceRecorder()
    qid = "test-0"

    with recorder.trace(qid):
        with stage_span("retrieve", recorder, qid):
            pass
        with stage_span("generate", recorder, qid) as span:
            span.add_tokens(100, 200)

    assert recorder.n_queries == 1
    assert recorder.n_fully_traced == 1
    assert recorder.pct_fully_traced == 1.0

    summary = recorder.summary()
    assert "retrieve" in summary
    assert "generate" in summary
    assert summary["generate"]["prompt_tokens"] == 100
    assert summary["generate"]["completion_tokens"] == 200
    assert summary["retrieve"]["prompt_tokens"] == 0

    # Partial trace — missing "generate" → not fully traced
    recorder2 = TraceRecorder()
    with recorder2.trace("test-1"), stage_span("retrieve", recorder2, "test-1"):
        pass

    assert recorder2.n_fully_traced == 0
