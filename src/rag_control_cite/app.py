"""FastAPI app — the serving surface.

``POST /query``: authenticate → embed → hybrid retrieve (pgvector + BM25,
tenant-filtered) → ACL filter (defence in depth) → generate → return the
answer with exact-span citations. Every stage is traced (Stage 6).

``GET /healthz``: readiness for the container/platform probes.
``GET /observability``: per-stage trace summary, operator-token gated.

Authentication: the tenant is derived from a bearer token, not the body.
A body ``tenant_id`` that disagrees with the token is rejected. When no
tokens are configured the service runs unauthenticated — local only.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from rag_control_cite.config import settings
from rag_control_cite.models import QueryResult, Tenant
from rag_control_cite.observability import TraceRecorder, configure_otel, stage_span

recorder = TraceRecorder()


class QueryRequest(BaseModel):
    query: str
    # Optional: when present it must match the token's tenant, else 403.
    tenant_id: str | None = None


class CitationOut(BaseModel):
    chunk_id: str
    source_uri: str
    start_char: int
    end_char: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieved: int
    trace_id: str


def _get_tenant(tenant_id: str) -> Tenant | None:
    from rag_control_cite.tenants import TENANTS

    return TENANTS.get(tenant_id)


@lru_cache(maxsize=1)
def _get_embedder() -> Any:
    from rag_control_cite.embedder import OpenRouterEmbedder

    return OpenRouterEmbedder()


@lru_cache(maxsize=1)
def _get_generator() -> Any:
    from rag_control_cite.generation import Generator

    return Generator()


@lru_cache(maxsize=1)
def _get_store() -> Any:
    from rag_control_cite.store import VectorStore

    store = VectorStore()  # resolves DATABASE_URL / RDS_* from environment
    store.load_corpus()
    return store


@lru_cache(maxsize=1)
def _seeded() -> bool:
    """Whether the store has rows — checked once per process, not per request."""
    return _get_store().count_chunks() > 0


def _authorize(authorization: str | None, body_tenant: str | None) -> Tenant:
    """Resolve the caller's tenant from the bearer token.

    Empty token config = authentication disabled (local runs): the body
    ``tenant_id`` is trusted. Otherwise the token names the tenant and a
    disagreeing body ``tenant_id`` is a 403.
    """
    from rag_control_cite.tenants import load_tenant_tokens

    tokens = load_tenant_tokens()
    if not tokens:
        if not body_tenant:
            raise HTTPException(status_code=400, detail="tenant_id is required")
        tenant = _get_tenant(body_tenant)
        if tenant is None:
            raise HTTPException(status_code=403, detail=f"unknown tenant: {body_tenant}")
        return tenant

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    token_tenant = tokens.get(token)
    if token_tenant is None:
        raise HTTPException(status_code=401, detail="invalid bearer token")
    if body_tenant and body_tenant != token_tenant:
        raise HTTPException(status_code=403, detail="tenant_id does not match token")
    tenant = _get_tenant(token_tenant)
    if tenant is None:
        raise HTTPException(status_code=403, detail=f"unknown tenant: {token_tenant}")
    return tenant


def _authorize_operator(authorization: str | None) -> None:
    """Gate /observability: an operator token, or local-only open access."""
    from rag_control_cite.tenants import load_operator_token, load_tenant_tokens

    operator = load_operator_token()
    if operator:
        if authorization != f"Bearer {operator}":
            raise HTTPException(status_code=403, detail="operator token required")
        return
    if load_tenant_tokens():
        raise HTTPException(status_code=403, detail="operator token not configured")
    # No tokens configured anywhere — local run, allow.


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Boot: connect + load corpus once; configure OTEL if an endpoint is set."""
    endpoint = settings.otel_exporter_otlp_endpoint
    if endpoint:
        configure_otel(endpoint=endpoint)
    _get_store()
    yield


app = FastAPI(title="rag-control-cite", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    authorization: str | None = Header(default=None),
) -> QueryResponse:
    """Answer a tenant-scoped query with citations.

    Sync handler on purpose: the pipeline calls blocking IO (psycopg,
    openai, httpx). A sync handler runs in FastAPI's threadpool.
    """
    trace_id = uuid.uuid4().hex[:16]
    from rag_control_cite.acl import enforce_acl

    with recorder.trace(trace_id) as trace:
        try:
            tenant = _authorize(authorization, req.tenant_id)
        except HTTPException as exc:
            trace.outcome = exc.status_code
            raise

        # Embed
        with stage_span("embed", recorder, trace_id):
            embedder = _get_embedder()
            q_vec = embedder.embed([req.query])[0]

        if not _seeded():
            trace.outcome = 503
            raise HTTPException(
                status_code=503,
                detail="store not seeded yet — run: uv run rcc seed",
            )

        # Hybrid retrieve — the entitlement filter is pushed into both legs
        store = _get_store()
        with stage_span("retrieve", recorder, trace_id):
            retrieved = store.retrieve_hybrid(req.query, q_vec, k=10, tenant=tenant)

        # ACL filter — defence in depth on the fused set
        with stage_span("acl", recorder, trace_id):
            retrieved = enforce_acl(retrieved, tenant)

        if not retrieved:
            # The tenant's entitled set is empty for this query.
            trace.outcome = 404
            raise HTTPException(
                status_code=404,
                detail="no accessible documents for this tenant",
            )

        # Generate + citations
        with stage_span("generate", recorder, trace_id) as span:
            generator = _get_generator()
            result: QueryResult = generator.generate(req.query, retrieved, tenant.tenant_id)
            if result.usage is not None:
                span.add_tokens(
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    result.usage.model,
                )

        trace.outcome = 200

    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationOut(
                chunk_id=c.chunk_id,
                source_uri=c.source_uri,
                start_char=c.start_char,
                end_char=c.end_char,
                text=c.text,
            )
            for c in result.citations
        ],
        retrieved=len(retrieved),
        trace_id=trace_id,
    )


@app.get("/observability")
async def observability(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Per-stage trace summary for the service lifetime (operator only)."""
    _authorize_operator(authorization)
    return recorder.as_dict()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "rag_control_cite.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
