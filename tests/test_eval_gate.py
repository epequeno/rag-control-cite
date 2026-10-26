"""Eval gate: baseline immutability, regression vs failure, CI thresholds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import rag_control_cite.hybrid as hybrid_mod
from rag_control_cite.eval_gate import (
    EvalBaseline,
    GateConfig,
    accept_baseline,
    run_eval_gate,
)
from rag_control_cite.judge import Judgement
from rag_control_cite.models import Chunk, QueryResult, Retrieved, Usage


@dataclass
class _Q:
    query: str
    gold_chunk_ids: frozenset[str]
    control_id: str
    category: str = "by_id"


def _retrieved(cid: str) -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=cid,
            source_uri="t://x",
            text=cid,
            start_char=0,
            end_char=1,
            chunk_index=0,
            doc_type="x",
            access_tags=frozenset(),
            last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        score=1.0,
        rank=1,
    )


class _Generator:
    def generate(self, query, retrieved, tenant_id=""):
        return QueryResult(query=query, answer="answer", retrieved=retrieved)


class _Judge:
    def judge_groundedness(self, result, max_chunks=5):
        return Judgement(score=1.0, usage=Usage(0, 0, "m"))


@pytest.fixture
def gate(monkeypatch):
    """Retrieval returns gold for the 'pass' query, nothing for 'fail'."""

    def fake_retrieve(index, embedder, query, k=10, k_rrf=60, weights=None, k_leg=None):
        if query == "pass":
            return [_retrieved("g1")]
        return []

    monkeypatch.setattr(hybrid_mod, "hybrid_retrieve", fake_retrieve)
    yield


def _config(tmp_path: Path) -> GateConfig:
    return GateConfig(
        baseline_path=tmp_path / "baseline.json",
        candidate_path=tmp_path / "candidate.json",
    )


def _queries() -> list[_Q]:
    return [
        _Q("pass", frozenset({"g1"}), "passctrl"),
        _Q("fail", frozenset({"g2"}), "failctrl"),
    ]


def test_failing_run_never_writes_baseline(gate, tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"metrics": {}, "ci_lower": {}, "results": {}}))
    before = baseline_path.read_bytes()

    result = run_eval_gate(None, None, _queries(), _config(tmp_path), _Generator(), _Judge())

    assert result.passed is False
    assert result.candidate_written is True
    assert baseline_path.read_bytes() == before  # byte-identical
    assert (tmp_path / "candidate.json").exists()


def test_regression_vs_known_failure(gate, tmp_path) -> None:
    cfg = _config(tmp_path)
    # Baseline: every query passed before.
    EvalBaseline(
        metrics={"recall@10": 1.0},
        ci_lower={"recall@10": 1.0},
        results={"by_id:passctrl": True, "by_id:failctrl": True},
    ).save(cfg.baseline_path)

    result = run_eval_gate(None, None, _queries(), cfg, _Generator(), _Judge())
    assert result.n_failures == 1
    assert result.n_regressions == 1  # failctrl passed before, fails now
    assert result.n_known_failures == 0

    # Baseline: failctrl was already failing → known, not a regression.
    EvalBaseline(
        metrics={"recall@10": 1.0},
        ci_lower={"recall@10": 1.0},
        results={"by_id:passctrl": True, "by_id:failctrl": False},
    ).save(cfg.baseline_path)
    result2 = run_eval_gate(None, None, _queries(), cfg, _Generator(), _Judge())
    assert result2.n_regressions == 0
    assert result2.n_known_failures == 1


def test_reported_count_matches_evaluated(gate, tmp_path) -> None:
    queries = [_Q(f"pass{i % 2}", frozenset({f"g{i}"}), f"c{i}") for i in range(6)]
    cfg = _config(tmp_path)
    all_run = run_eval_gate(None, None, queries, cfg, _Generator(), _Judge())
    assert all_run.n_queries == 6  # n_eval_queries=None → whole set

    cfg.n_eval_queries = 2
    sampled = run_eval_gate(None, None, queries, cfg, _Generator(), _Judge())
    assert sampled.n_queries == 2


def test_ci_threshold_blocks_low_bound(gate, tmp_path) -> None:
    cfg = _config(tmp_path)
    # Only one of two queries passes → mean 0.5, CI lower far below 0.60.
    result = run_eval_gate(None, None, _queries(), cfg, _Generator(), _Judge())
    assert result.ci_lower["recall@10"] < cfg.thresholds["recall@10"]
    assert result.passed is False


def test_passing_run_writes_baseline(gate, tmp_path) -> None:
    cfg = _config(tmp_path)
    queries = [_Q("pass", frozenset({"g1"}), "only")]

    def fake_retrieve(index, embedder, query, k=10, k_rrf=60, weights=None, k_leg=None):
        return [_retrieved("g1")]

    import rag_control_cite.hybrid as h

    h.hybrid_retrieve = fake_retrieve  # type: ignore[assignment]
    result = run_eval_gate(None, None, queries, cfg, _Generator(), _Judge())
    assert result.passed is True
    assert cfg.baseline_path.exists()
    assert not result.candidate_written


def test_accept_promotes_candidate(gate, tmp_path) -> None:
    cfg = _config(tmp_path)
    run_eval_gate(None, None, _queries(), cfg, _Generator(), _Judge())
    assert cfg.candidate_path.exists()
    assert accept_baseline(cfg.candidate_path, cfg.baseline_path) is True
    assert cfg.baseline_path.exists()


@pytest.mark.parametrize(("passed", "exit_code"), [(False, 1), (True, None)])
def test_cli_eval_exit_code_follows_gate(monkeypatch, passed, exit_code) -> None:
    """The workflow gates on the exit code: a failing gate must fail the step."""
    import rag_control_cite.embedder as embedder_mod
    import rag_control_cite.eval_gate as eval_gate_mod
    import rag_control_cite.generation as generation_mod
    import rag_control_cite.index as index_mod
    import rag_control_cite.ingestion as ingestion_mod
    import rag_control_cite.judge as judge_mod
    import rag_control_cite.query_set as query_set_mod
    from rag_control_cite import cli

    report = {
        "passed": passed,
        "n_queries": 1,
        "metrics": {},
        "ci_lower": {},
        "n_failures": 0 if passed else 1,
        "n_regressions": 0,
        "n_known_failures": 0,
        "baseline_loaded": True,
        "candidate_written": not passed,
        "failed_queries": [],
    }
    fake_index = SimpleNamespace(vectors=SimpleNamespace(shape=(1, 1)))
    monkeypatch.setattr(ingestion_mod, "ingest", lambda: ([object()], None))
    monkeypatch.setattr(embedder_mod, "OpenRouterEmbedder", lambda: object())
    monkeypatch.setattr(index_mod, "build_or_load_index", lambda chunks, embedder: fake_index)
    monkeypatch.setattr(query_set_mod, "generate_exact_term_queries", lambda chunks: [])
    monkeypatch.setattr(query_set_mod, "split_queries", lambda queries, split: [])
    monkeypatch.setattr(generation_mod, "Generator", lambda: object())
    monkeypatch.setattr(judge_mod, "Judge", lambda: object())
    monkeypatch.setattr(
        eval_gate_mod,
        "run_eval_gate",
        lambda *args, **kwargs: SimpleNamespace(n_queries=1, as_dict=lambda: report),
    )
    monkeypatch.setattr(cli, "_write_result", lambda kind, payload: None)

    if exit_code is None:
        cli._eval([])
    else:
        with pytest.raises(SystemExit) as excinfo:
            cli._eval([])
        assert excinfo.value.code == exit_code
