"""Stage 5 — an eval gate that gates.

Runs hybrid retrieval → generate → judge on the query set and measures the
RAG triad per query (recall@10, nDCG@5, groundedness). The gate:

  1. evaluates the whole set (or a stratified sample) — the reported count
     is the count actually evaluated;
  2. compares each metric's *bootstrap 95% CI lower bound* to its
     threshold, not a point estimate;
  3. distinguishes a *failure* (a query below its floor) from a
     *regression* (a query that passed in the baseline and fails now);
  4. never overwrites the baseline on a failing run — a candidate is
     written and promoted only with an explicit ``accept_baseline``.

The baseline is tracked in ``eval/baseline.json`` (dev/test split lives in
the query set). It is a local, tested implementation — the sibling
``rag-eval-project`` harness is not imported (its own metrics are under
audit; re-adoption is deferred).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "EvalBaseline",
    "GateConfig",
    "GateResult",
    "accept_baseline",
    "estimate_false_block_rate",
    "run_eval_gate",
]

# Default thresholds — compared against the CI lower bound, not the mean.
DEFAULT_THRESHOLDS = {
    "recall@10": 0.60,
    "ndcg@5": 0.50,
    "groundedness": 0.80,
}

# A query fails if any listed metric is at or below its floor.
DEFAULT_PER_QUERY_FLOOR = {"recall@10": 0.0, "groundedness": 0.0}

BASELINE_PATH = Path("eval/baseline.json")
CANDIDATE_PATH = Path("results/eval_candidate.json")


@dataclass
class GateConfig:
    """Configuration for the eval gate."""

    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    per_query_floor: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PER_QUERY_FLOOR))
    n_eval_queries: int | None = None  # None = evaluate the whole set
    k_recall: int = 50
    k_final: int = 5
    baseline_tolerance: float = 0.02
    baseline_path: Path = field(default_factory=lambda: BASELINE_PATH)
    candidate_path: Path = field(default_factory=lambda: CANDIDATE_PATH)


@dataclass
class GateResult:
    """Outcome of running the eval gate."""

    metrics: dict[str, float] = field(default_factory=dict)
    ci_lower: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    n_queries: int = 0
    n_failures: int = 0
    n_regressions: int = 0
    n_known_failures: int = 0
    n_false_blocks: int = 0
    failed_queries: list[dict[str, Any]] = field(default_factory=list)
    baseline_loaded: bool = False
    candidate_written: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_queries": self.n_queries,
            "metrics": self.metrics,
            "ci_lower": self.ci_lower,
            "n_failures": self.n_failures,
            "n_regressions": self.n_regressions,
            "n_known_failures": self.n_known_failures,
            "n_false_blocks": self.n_false_blocks,
            "baseline_loaded": self.baseline_loaded,
            "candidate_written": self.candidate_written,
            "failed_queries": self.failed_queries,
        }


@dataclass
class EvalBaseline:
    """Stored baseline metrics + per-query pass/fail for regression checks."""

    metrics: dict[str, float] = field(default_factory=dict)
    ci_lower: dict[str, float] = field(default_factory=dict)
    results: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> EvalBaseline | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(
            metrics=data.get("metrics", {}),
            ci_lower=data.get("ci_lower", {}),
            results={str(k): bool(v) for k, v in data.get("results", {}).items()},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "metrics": self.metrics,
                    "ci_lower": self.ci_lower,
                    "results": dict(sorted(self.results.items())),
                },
                indent=2,
            )
        )


def accept_baseline(
    candidate_path: Path = CANDIDATE_PATH,
    baseline_path: Path = BASELINE_PATH,
) -> bool:
    """Promote a candidate to the baseline. Returns False if none exists."""
    if not candidate_path.exists():
        return False
    baseline = EvalBaseline.load(candidate_path)
    if baseline is None:
        return False
    baseline.save(baseline_path)
    return True


def _query_id(q: object) -> str:
    """Stable ID for a query — used for regression tracking."""
    return f"{q.category}:{q.control_id}"  # type: ignore[attr-defined]


def _stratified_sample(queries: list[object], n: int) -> list[object]:
    """Take up to ``n`` queries, spread across categories."""
    by_category: dict[str, list[object]] = {}
    for q in queries:
        by_category.setdefault(q.category, []).append(q)  # type: ignore[attr-defined]
    if not by_category:
        return []
    picked: list[object] = []
    # Round-robin across categories so the sample is stratified.
    while len(picked) < n and any(by_category.values()):
        for cat in sorted(by_category):
            bucket = by_category[cat]
            if bucket and len(picked) < n:
                picked.append(bucket.pop(0))
    return picked


def run_eval_gate(
    index: object,
    embedder: object,
    queries: list[object],
    config: GateConfig | None = None,
    generator: object | None = None,
    judge: object | None = None,
) -> GateResult:
    """Run the eval gate: RAG triad + regression check."""
    import time

    from rag_control_cite.hybrid import hybrid_retrieve
    from rag_control_cite.measure import bootstrap_ci, ndcg_at_k, recall_at_k

    if config is None:
        config = GateConfig()

    result = GateResult()
    baseline = EvalBaseline.load(config.baseline_path)
    result.baseline_loaded = baseline is not None

    eval_queries = (
        list(queries)
        if config.n_eval_queries is None
        else _stratified_sample(list(queries), config.n_eval_queries)
    )
    result.n_queries = len(eval_queries)

    if generator is None:
        from rag_control_cite.generation import Generator

        generator = Generator()
    if judge is None:
        from rag_control_cite.judge import Judge

        judge = Judge()

    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    groundedness_scores: list[float] = []
    per_query: dict[str, bool] = {}
    failed: list[dict[str, Any]] = []

    for q in eval_queries:
        t0 = time.perf_counter()
        retrieved = hybrid_retrieve(index, embedder, q.query, k=config.k_final * 10, weights=None)  # type: ignore[attr-defined]
        rec = recall_at_k(retrieved, q.gold_chunk_ids, 10)  # type: ignore[attr-defined]
        ndcg = ndcg_at_k(retrieved, q.gold_chunk_ids, config.k_final)  # type: ignore[attr-defined]

        gen_result = generator.generate(q.query, retrieved)  # type: ignore[attr-defined]
        judgement = judge.judge_groundedness(gen_result)  # type: ignore[attr-defined]
        grounded = float(judgement.score)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        recall_scores.append(rec)
        ndcg_scores.append(ndcg)
        groundedness_scores.append(grounded)

        qid = _query_id(q)
        metrics_now = {"recall@10": rec, "groundedness": grounded}
        failed_now = any(
            metrics_now.get(m, 1.0) <= floor for m, floor in config.per_query_floor.items()
        )
        per_query[qid] = not failed_now

        if failed_now:
            was_passing = baseline is not None and baseline.results.get(qid, False)
            failed.append(
                {
                    "query_id": qid,
                    "query": q.query,
                    "recall@10": rec,
                    "ndcg@5": ndcg,
                    "groundedness": grounded,
                    "latency_ms": round(elapsed_ms, 1),
                    "is_regression": was_passing,
                }
            )

    result.metrics = {
        "recall@10": round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0,
        "ndcg@5": round(sum(ndcg_scores) / len(ndcg_scores), 4) if ndcg_scores else 0.0,
        "groundedness": round(sum(groundedness_scores) / len(groundedness_scores), 4)
        if groundedness_scores
        else 0.0,
    }
    result.ci_lower = {
        "recall@10": round(bootstrap_ci(recall_scores)[0], 4),
        "ndcg@5": round(bootstrap_ci(ndcg_scores)[0], 4),
        "groundedness": round(bootstrap_ci(groundedness_scores)[0], 4),
    }

    result.n_failures = len(failed)
    result.n_regressions = sum(1 for f in failed if f["is_regression"])
    result.n_known_failures = sum(1 for f in failed if not f["is_regression"])
    result.failed_queries = failed

    threshold_pass = all(result.ci_lower.get(m, 0.0) >= t for m, t in config.thresholds.items())
    baseline_pass = True
    if baseline and baseline.ci_lower:
        for metric, base_lo in baseline.ci_lower.items():
            if result.ci_lower.get(metric, 0.0) < base_lo - config.baseline_tolerance:
                baseline_pass = False

    result.passed = threshold_pass and baseline_pass and result.n_regressions == 0

    candidate = EvalBaseline(
        metrics=result.metrics,
        ci_lower=result.ci_lower,
        results=per_query,
    )
    if result.passed:
        candidate.save(config.baseline_path)
    else:
        candidate.save(config.candidate_path)
        result.candidate_written = True

    return result


def estimate_false_block_rate(run_once: Any, repeats: int = 5) -> float:
    """False-block rate from repeated runs of the *same* code.

    A "false block" is a non-zero exit from a run that changes nothing.
    ``run_once`` returns True when the gate blocked. Rate = blocks / repeats.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    blocks = sum(1 for _ in range(repeats) if run_once())
    return blocks / repeats
