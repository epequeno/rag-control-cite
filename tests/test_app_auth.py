"""Serving surface: 401/403 behaviour, token-derived tenant, /observability."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_control_cite import app as appmod
from rag_control_cite.models import Chunk, QueryResult, Retrieved, Usage


def _retrieved() -> list[Retrieved]:
    chunk = Chunk(
        chunk_id="c-ac-2",
        source_uri="oscal://cat",
        text="AC-2: Account Management",
        start_char=0,
        end_char=10,
        chunk_index=0,
        doc_type="oscal-json",
        access_tags=frozenset(
            {"framework:sp-800-53", "family:ac", "baseline:low", "audience:security"}
        ),
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        title="Account Management",
        control_id="ac-2",
    )
    return [Retrieved(chunk=chunk, score=0.9, rank=1)]


class _Embedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 4), dtype=np.float32)


class _Store:
    def __init__(self, retrieved: list[Retrieved]) -> None:
        self._retrieved = retrieved

    def retrieve_hybrid(self, query, query_vec, k=10, tenant=None, k_leg=None):
        return list(self._retrieved)


class _Generator:
    def generate(self, query, retrieved, tenant_id=""):
        return QueryResult(
            query=query,
            answer="AC-2 requires account management.",
            citations=[],
            retrieved=retrieved,
            tenant_id=tenant_id,
            usage=Usage(prompt_tokens=10, completion_tokens=5, model="openai/gpt-4o-mini"),
        )


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(appmod, "_get_embedder", lambda: _Embedder())
    monkeypatch.setattr(appmod, "_get_store", lambda: _Store(_retrieved()))
    monkeypatch.setattr(appmod, "_get_generator", lambda: _Generator())
    monkeypatch.setattr(appmod, "_seeded", lambda: True)
    return TestClient(appmod.app)


@pytest.fixture
def auth_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "RCC_TENANT_TOKENS",
        json.dumps({"eng-low": "tok-a", "sec-high": "tok-b"}),
    )
    monkeypatch.setenv("RCC_OPERATOR_TOKEN", "op-token")


def test_missing_token_401(client, auth_env) -> None:
    resp = client.post("/query", json={"query": "q"})
    assert resp.status_code == 401


def test_invalid_token_401(client, auth_env) -> None:
    resp = client.post("/query", json={"query": "q"}, headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


def test_body_tenant_override_403(client, auth_env) -> None:
    resp = client.post(
        "/query",
        json={"query": "q", "tenant_id": "sec-high"},
        headers={"Authorization": "Bearer tok-a"},
    )
    assert resp.status_code == 403


def test_valid_token_succeeds(client, auth_env) -> None:
    resp = client.post("/query", json={"query": "q"}, headers={"Authorization": "Bearer tok-a"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["trace_id"]


def test_matching_body_tenant_ok(client, auth_env) -> None:
    resp = client.post(
        "/query",
        json={"query": "q", "tenant_id": "eng-low"},
        headers={"Authorization": "Bearer tok-a"},
    )
    assert resp.status_code == 200


def test_observability_operator_gated(client, auth_env) -> None:
    assert client.get("/observability").status_code == 403
    assert (
        client.get("/observability", headers={"Authorization": "Bearer wrong"}).status_code == 403
    )
    ok = client.get("/observability", headers={"Authorization": "Bearer op-token"})
    assert ok.status_code == 200


def test_auth_disabled_local(client, monkeypatch) -> None:
    monkeypatch.delenv("RCC_TENANT_TOKENS", raising=False)
    monkeypatch.delenv("RCC_OPERATOR_TOKEN", raising=False)
    ok = client.post("/query", json={"query": "q", "tenant_id": "eng-low"})
    assert ok.status_code == 200
    assert client.get("/observability").status_code == 200
    bad = client.post("/query", json={"query": "q", "tenant_id": "ghost"})
    assert bad.status_code == 403


def test_empty_retrieval_404(client, auth_env, monkeypatch) -> None:
    monkeypatch.setattr(appmod, "_get_store", lambda: _Store([]))
    resp = client.post("/query", json={"query": "q"}, headers={"Authorization": "Bearer tok-a"})
    assert resp.status_code == 404


def test_usage_recorded_per_request(client, auth_env) -> None:
    before = appmod.recorder.stage_tokens("generate")
    client.post("/query", json={"query": "q"}, headers={"Authorization": "Bearer tok-a"})
    after = appmod.recorder.stage_tokens("generate")
    assert after[0] - before[0] == 10
    assert after[1] - before[1] == 5
