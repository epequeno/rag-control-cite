"""CLI entry point — `rcc <command>`.

Subcommands:
  ingest  — Stage 0: parse → chunk → metadata
  seed    — ingest → embed → upsert into pgvector (local or RDS)
  demo    — dense vs hybrid vs rerank, then the ACL audit (Stages 1-4)
  eval    — Stage 5: eval gate (RAG triad + regression)
  accept  — promote a passing candidate baseline to eval/baseline.json
"""

from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("usage: rcc <ingest|seed|demo|eval|accept>")
        raise SystemExit(1)
    cmd, rest = args[0], args[1:]
    {
        "ingest": _ingest,
        "seed": _seed,
        "demo": _demo,
        "eval": _eval,
        "accept": _accept,
    }.get(cmd, _unknown)(rest)


def _seed(args: list[str]) -> None:
    """Ingest → embed → upsert into pgvector. Idempotent (upsert on chunk_id)."""
    import time

    from rag_control_cite.embedder import OpenRouterEmbedder
    from rag_control_cite.ingestion import ingest
    from rag_control_cite.store import VectorStore

    print("Stage 0+1 — seed pgvector (parse → embed → upsert)")

    print("  ingesting…  ", end="", flush=True)
    chunks, report = ingest()
    print(f"{len(chunks)} chunks ({report.parse_failure_rate:.0%} parse failures)")

    store = VectorStore()
    print("  applying schema…  ", end="", flush=True)
    store.apply_schema()
    print("done")

    print("  embedding…  ", end="", flush=True)
    t0 = time.perf_counter()
    embedder = OpenRouterEmbedder()
    vectors = embedder.embed([c.text for c in chunks])
    print(f"{vectors.shape[0]} vectors in {time.perf_counter() - t0:.1f}s")

    print("  upserting…  ", end="", flush=True)
    t0 = time.perf_counter()
    store.upsert_chunks(chunks, vectors)
    n = store.count_chunks()
    print(f"{n} rows in DB ({time.perf_counter() - t0:.1f}s)")
    store.close()
    print("seed complete.")


def _ingest(args: list[str]) -> None:
    from rag_control_cite.ingestion import ingest

    _chunks, report = ingest()
    r = report.as_dict()
    print("Stage 0 — ingestion")
    print(f"  files attempted:     {r['files_attempted']}")
    print(f"  files failed:        {r['files_failed']}")
    print(f"  parse-failure rate:  {r['parse_failure_rate']:.2%}")
    print(f"  controls:            {r['total_controls']}")
    print(
        f"  table-extraction:    {r['table_extraction_rate']:.2%}  ({r['table_controls']}/{r['total_controls']})"
    )
    print(f"  documents:           {r['total_documents']}")
    print(f"  chunks:              {r['total_chunks']}")
    print(
        f"  char-sanity pass:    {r['char_sanity_pass_pct']:.2%}  ({r['char_sanity_passed']}/{r['char_sanity_total']})"
    )
    _write_result("ingest", {"command": "uv run rcc ingest", "date": _utc_now_iso(), **r})


def _demo(args: list[str]) -> None:
    """Stages 1-4: dense vs hybrid vs rerank, then the ACL audit."""
    import time

    from rag_control_cite.embedder import OpenRouterEmbedder
    from rag_control_cite.generation import Generator
    from rag_control_cite.index import build_or_load_index
    from rag_control_cite.ingestion import ingest
    from rag_control_cite.judge import Judge
    from rag_control_cite.measure import (
        measure_hybrid_retrieval,
        measure_rerank,
        measure_retrieval,
        ndcg_at_k,
    )
    from rag_control_cite.observability import TraceRecorder, stage_span
    from rag_control_cite.query_set import generate_exact_term_queries, split_queries

    print("Stage 1+2+3 — dense vs hybrid vs hybrid+rerank")
    print()

    print("  ingesting…  ", end="", flush=True)
    chunks, _ = ingest()
    print(f"{len(chunks)} chunks")

    print("  embedding…  ", end="", flush=True)
    t0 = time.perf_counter()
    embedder = OpenRouterEmbedder()
    index = build_or_load_index(chunks, embedder)
    print(
        f"done in {time.perf_counter() - t0:.1f}s  ({index.vectors.shape[0]} vectors, dim {index.vectors.shape[1]})"
    )

    print("  generating queries…  ", end="", flush=True)
    all_queries = generate_exact_term_queries(chunks)
    queries = split_queries(all_queries, "test")
    print(f"{len(queries)} test queries ({len(all_queries)} total)")

    print("  dense retrieval…  ", end="", flush=True)
    dense_report = measure_retrieval(index, embedder, queries, k=10)
    print(
        f"recall@10: {dense_report.recall_at_10:.2%}, p50: {dense_report.p50_ms:.0f}ms, p95: {dense_report.p95_ms:.0f}ms"
    )

    print("  hybrid retrieval…  ", end="", flush=True)
    hybrid_report = measure_hybrid_retrieval(index, embedder, queries, k=10, weights=None)
    print(
        f"recall@10: {hybrid_report.recall_at_10:.2%}, p50: {hybrid_report.p50_ms:.0f}ms, p95: {hybrid_report.p95_ms:.0f}ms"
    )

    from rag_control_cite.hybrid import hybrid_retrieve

    # Equal pools: the no-rerank baseline is hybrid top-50 truncated to 5 —
    # the same candidate pool the reranker sees — not a separate k=5 retrieval.
    hybrid_ndcg5_scores: list[float] = []
    for q in queries:
        hybrid_top50 = hybrid_retrieve(index, embedder, q.query, k=50, weights=None)
        hybrid_ndcg5_scores.append(ndcg_at_k(hybrid_top50[:5], q.gold_chunk_ids, 5))
    hybrid_ndcg5 = (
        sum(hybrid_ndcg5_scores) / len(hybrid_ndcg5_scores) if hybrid_ndcg5_scores else 0.0
    )
    print(f"  nDCG@5 (hybrid top-5): {hybrid_ndcg5:.2%}")

    # RRF weight sweep at k_leg=50, so a weight can actually change the top-10.
    from rag_control_cite.measure import rrf_weight_sweep

    print("  RRF weight sweep (k_leg=50)…  ", end="", flush=True)
    sweep = rrf_weight_sweep(
        index, embedder, queries, [[w, 1.0] for w in (0.5, 1.0, 1.5, 2.0, 3.0)], k_leg=50
    )
    print("done")
    for weights_key, value in sweep.items():
        print(f"    dense×{weights_key[0]:g} / bm25×{weights_key[1]:g}: recall@10 {value:.2%}")

    print("  loading cross-encoder…  ", end="", flush=True)
    t0 = time.perf_counter()
    rerank_report = measure_rerank(index, embedder, queries, k_recall=50, k_final=5, weights=None)
    print(f"done in {time.perf_counter() - t0:.1f}s")
    print(
        f"  nDCG@5: {rerank_report.ndcg_at_5:.2%}, p50: {rerank_report.p50_ms:.0f}ms, p95: {rerank_report.p95_ms:.0f}ms"
    )
    print(f"  Δ nDCG@5: {rerank_report.ndcg_at_5 - hybrid_ndcg5:+.2%}")

    # One demo query end-to-end (traced)
    recorder = TraceRecorder()
    if queries:
        from rag_control_cite.tenants import EVAL_TENANT

        q = queries[0]
        qid = "demo-0"
        print()
        print(f"  demo query: {q.query}")

        with recorder.trace(qid) as trace:
            with stage_span("retrieve", recorder, qid):
                candidates = hybrid_retrieve(index, embedder, q.query, k=50, weights=None)
            with stage_span("acl", recorder, qid):
                from rag_control_cite.acl import enforce_acl

                candidates = enforce_acl(candidates, EVAL_TENANT)
            with stage_span("generate", recorder, qid) as span:
                generator = Generator()
                result = generator.generate(q.query, candidates)
                if result.usage is not None:
                    span.add_tokens(
                        result.usage.prompt_tokens,
                        result.usage.completion_tokens,
                        result.usage.model,
                    )
            with stage_span("judge", recorder, qid) as span:
                judge = Judge()
                judgement = judge.judge_groundedness(result)
                span.add_tokens(
                    judgement.usage.prompt_tokens,
                    judgement.usage.completion_tokens,
                    judgement.usage.model,
                )
            trace.outcome = 200
        print(f"  answer: {result.answer[:200]}…")
        print(f"  citations: {len(result.citations)}")
        if result.usage is not None:
            rerank_report.prompt_tokens += (
                result.usage.prompt_tokens + judgement.usage.prompt_tokens
            )
            rerank_report.completion_tokens += (
                result.usage.completion_tokens + judgement.usage.completion_tokens
            )
        print(f"  grounded: {'yes' if judgement.grounded else 'no'}")

    print()
    for name, report in (
        ("dense (Stage 1)", dense_report),
        ("hybrid (Stage 2)", hybrid_report),
        ("hybrid+rerank (Stage 3)", rerank_report),
    ):
        d = report.as_dict()
        print(f"  --- {name} ---")
        for key in (
            "recall@10",
            "ndcg@5",
            "p50_ms",
            "p95_ms",
            "recall@10_ci_lo",
            "recall@10_ci_hi",
            "ndcg@5_ci_lo",
            "ndcg@5_ci_hi",
        ):
            if key in d:
                print(f"    {key}: {d[key]}")

    # ACL audit (Stage 4) — chunk-unit leakage over real retrievals
    from rag_control_cite.acl import (
        EntitlementOracle,
        planted_leak_chunks,
        run_acl_audit,
    )
    from rag_control_cite.hybrid import BM25Retriever
    from rag_control_cite.retrieval import retrieve as dense_retrieve
    from rag_control_cite.tenants import TENANTS

    print()
    print("  ACL audit…  ", end="", flush=True)
    oracle = EntitlementOracle.from_corpus()
    bm25 = BM25Retriever(chunks=index.chunks)

    class _CachingEmbedder:
        """Reuse one query embedding across the tenant/leg cross-product.

        Same vectors (deterministic), far fewer API calls — the audit is a
        tensor product of legs x tenants x queries, but the query set is
        small and each query vector is computed once.
        """

        def __init__(self, inner: object) -> None:
            self._inner = inner
            self._cache: dict[tuple[str, ...], object] = {}

        @property
        def id(self) -> str:
            return self._inner.id  # type: ignore[attr-defined]

        def embed(self, texts: list[str]):
            key = tuple(texts)
            if key not in self._cache:
                self._cache[key] = self._inner.embed(texts)  # type: ignore[attr-defined]
            return self._cache[key]

    cached = _CachingEmbedder(embedder)
    retrievers = {
        "dense": lambda query, k: dense_retrieve(index, cached.embed([query])[0], k),
        "bm25": lambda query, k: bm25.retrieve(query, k_recall=k),
        "hybrid": lambda query, k: hybrid_retrieve(index, cached, query, k=k),
    }
    audit = run_acl_audit(
        retrievers,
        [q.query for q in queries],
        TENANTS,
        oracle,
        planted=planted_leak_chunks(TENANTS["eng-low"]),
        k=10,
    )
    ar = audit.as_dict()
    print(f"{ar['n_combos']} combos, {ar['n_checked']} chunk checks, {ar['n_leaked']} leaked")
    print(
        f"  prevented: {ar['n_prevented']}  ({ar['prevented_rate']:.2%} of unauthorized candidates)"
    )

    # Observability (Stage 6)
    print()
    obs = recorder.as_dict()
    print("  --- observability (Stage 6) ---")
    print(f"    queries traced: {obs['n_queries']}   fully traced: {obs['pct_fully_traced']:.0%}")
    for stage_name, s in obs["stages"].items():
        print(f"    {stage_name}: p50 {s['p50_ms']:.0f} ms  p95 {s['p95_ms']:.0f} ms")

    payload = {
        "command": "uv run rcc demo",
        "date": _utc_now_iso(),
        "n_chunks": len(chunks),
        "n_test_queries": len(queries),
        "dense": dense_report.as_dict(),
        "hybrid": hybrid_report.as_dict(),
        "hybrid_ndcg5": round(hybrid_ndcg5, 4),
        "rrf_sweep": {f"{k[0]:g},{k[1]:g}": round(v, 4) for k, v in sweep.items()},
        "rerank": rerank_report.as_dict(),
        "acl": ar,
        "observability": obs,
    }
    _write_result("demo", payload)


def _eval(args: list[str]) -> None:
    """Stage 5: eval gate — RAG triad + regression check."""
    from rag_control_cite.embedder import OpenRouterEmbedder
    from rag_control_cite.eval_gate import GateConfig, accept_baseline, run_eval_gate
    from rag_control_cite.generation import Generator
    from rag_control_cite.index import build_or_load_index
    from rag_control_cite.ingestion import ingest
    from rag_control_cite.judge import Judge
    from rag_control_cite.query_set import generate_exact_term_queries, split_queries

    print("Stage 5 — eval gate (RAG triad + regression)")
    print()

    print("  ingesting…  ", end="", flush=True)
    chunks, _ = ingest()
    print(f"{len(chunks)} chunks")

    print("  embedding…  ", end="", flush=True)
    embedder = OpenRouterEmbedder()
    index = build_or_load_index(chunks, embedder)
    print(f"done  ({index.vectors.shape[0]} vectors)")

    print("  generating queries…  ", end="", flush=True)
    queries = split_queries(generate_exact_term_queries(chunks), "test")
    print(f"{len(queries)} test queries")

    print("  running eval gate…  ", end="", flush=True)
    result = run_eval_gate(
        index, embedder, queries, GateConfig(), generator=Generator(), judge=Judge()
    )
    print(f"done  ({result.n_queries} queries)")

    r = result.as_dict()
    print()
    print(f"  gate: {'PASS' if r['passed'] else 'FAIL'}")
    print(f"  queries evaluated: {r['n_queries']}")
    for metric, value in r["metrics"].items():
        threshold = DEFAULT_THRESHOLD(metric)
        ci_lo = r["ci_lower"].get(metric, 0.0)
        status = "✓" if ci_lo >= threshold else "✗"
        print(
            f"    {status} {metric}: {value:.2%}  (CI lower {ci_lo:.2%}, threshold {threshold:.2%})"
        )
    print(
        f"  failures: {r['n_failures']}  regressions: {r['n_regressions']}  known: {r['n_known_failures']}"
    )
    print(f"  baseline loaded: {r['baseline_loaded']}  candidate written: {r['candidate_written']}")

    if r["failed_queries"]:
        print()
        print("  failed queries (first 5):")
        for fq in r["failed_queries"][:5]:
            print(
                f"    [{fq['query_id']}] recall={fq['recall@10']:.0%} "
                f"groundedness={fq['groundedness']:.0%} "
                f"{'regression' if fq['is_regression'] else 'known'}"
            )

    _write_result(
        "eval",
        {"command": "uv run rcc eval", "date": _utc_now_iso(), **r},
    )

    if "--accept" in args:
        if accept_baseline():
            print("  baseline promoted from candidate.")
        else:
            print("  no candidate to promote.")

    # The workflow gates on this exit code. A failing gate that exits 0 is a
    # green CI run with a red verdict — the gate must block the build.
    if not r["passed"]:
        raise SystemExit(1)


def _accept(args: list[str]) -> None:
    from rag_control_cite.eval_gate import accept_baseline

    print("baseline promoted." if accept_baseline() else "no candidate to promote.")


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_result(kind: str, payload: dict) -> None:
    """Persist a run's numbers to results/<kind>_<ts>.json — the source of RESULTS.md."""
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"{kind}_{ts}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {path}")


def DEFAULT_THRESHOLD(metric: str) -> float:
    from rag_control_cite.eval_gate import DEFAULT_THRESHOLDS

    return DEFAULT_THRESHOLDS.get(metric, 0.0)


def _unknown(args: list[str]) -> None:
    print("unknown command", file=sys.stderr)
    raise SystemExit(2)
