# Agent Guidelines

## Project Goal

A multi-tenant document Q&A service with enforced access control over public regulatory
frameworks (OSCAL-first). Answers cite exact spans, return nothing the caller is not
entitled to, and ship with an eval gate in CI and per-stage traces.

## Agent Role

Agent-written, owner-directed. Agents implement, test, run, and document. The owner sets
scope, makes decisions, and reviews.

## Staged ladder

Eight stages (0–7), each ending in a number:

0. Ingestion: parse → chunk → metadata
1. Naive baseline (embed → top-k)
2. Hybrid retrieval (dense + BM25, RRF)
3. Retrieve-then-rerank (cross-encoder, top-50 → top-5)
4. ACL enforcement across cached and reranked paths
5. Eval gate in CI (RAG triad + regression set)
6. Observability — OTEL span per stage
7. Deploy + teardown

Stages are implemented in order; each stage's number is recorded in `RESULTS.md`.
The full plan is kept privately and is not part of this repository; `README.md` and
`RESULTS.md` are the durable record of what was built and measured.

## Repo conventions

- `just`, not `make`. `just down` tears everything down for cents.
- One idea per repo.
- README: problem → number → diagram → `just up` / `just demo` → what broke → date.
- `RESULTS.md` carries the number, the exact command, and the date.
- Synthetic tenants and invented data are fine — name the seam in one line.

## Documentation

Long-form plans live in the research vault; the repository carries the code and the README.