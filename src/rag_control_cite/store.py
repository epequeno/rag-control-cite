"""pgvector-backed store — the production retrieval surface.

Dense retrieval runs natively in Postgres (cosine operator). BM25 stays
in-process: the corpus is loaded from the DB at startup and the bm25s
index rebuilt (2k chunks ≈ 1.1 MB — sub-second).

ACL is enforced in two places: pushed into the candidate legs (a SQL
``WHERE`` on the dense leg, an allow-mask on BM25) so a small-entitlement
tenant is not starved by a post-retrieval filter, then applied again after
fusion as defence in depth. The two legs are fused by the same
``rrf_fuse`` the measured path uses, so served and measured rankings agree
by construction.

Connections come from a pool; the row count is not queried per request.
"""

from __future__ import annotations

from typing import Any

from rag_control_cite.models import Chunk, Retrieved, Tenant

__all__ = ["VectorStore", "database_url_from_environment"]

# RRF constant and leg weights — kept identical to Stage 2 defaults
K_RRF = 60

# Dimensions an entitlement clause is built from, in tag-prefix order.
_ACL_DIMENSIONS = (
    ("framework:", "frameworks"),
    ("family:", "control_families"),
    ("baseline:", "baseline_tiers"),
    ("audience:", "audiences"),
)


def acl_sql(tenant: Tenant | None) -> tuple[str, list[list[str]]]:
    """Build the fail-closed SQL predicate + params for a tenant.

    Each restricted dimension becomes one ``access_tags && %s`` clause
    (array overlap); a wildcard (``None``) dimension contributes nothing.
    A chunk missing tags for a restricted dimension fails overlap.
    """
    if tenant is None:
        return "TRUE", []
    clauses: list[str] = []
    params: list[list[str]] = []
    for prefix, attr in _ACL_DIMENSIONS:
        allowed = getattr(tenant, attr)
        if allowed is None:
            continue
        clauses.append("access_tags && %s::text[]")
        params.append(sorted(f"{prefix}{value}" for value in allowed))
    return (" AND ".join(clauses) if clauses else "TRUE"), params


def database_url_from_environment() -> str:
    """Build DATABASE_URL from ECS-injected RDS_* env vars, or fall back.

    In the container, CDK injects ``RDS_CREDENTIALS`` (Secrets Manager
    JSON with username/password) plus ``RDS_HOST``/``RDS_PORT``/
    ``RDS_DB_NAME`` env vars. Locally, plain ``DATABASE_URL`` is used.
    """
    import json as _json
    import os

    url = os.environ.get("DATABASE_URL", "")
    if url:
        return url

    creds_json = os.environ.get("RDS_CREDENTIALS", "")
    host = os.environ.get("RDS_HOST", "")
    if creds_json and host:
        creds = _json.loads(creds_json)
        port = os.environ.get("RDS_PORT", "5432")
        dbname = os.environ.get("RDS_DB_NAME", "rag")
        # RDS rejects unencrypted transport; local compose has no TLS.
        return (
            f"postgresql://{creds['username']}:{creds['password']}"
            f"@{host}:{port}/{dbname}?sslmode=require"
        )

    # Local default (settings also reads .env)
    from rag_control_cite.config import settings

    return settings.database_url


class VectorStore:
    """pgvector store: schema apply, chunk upsert, hybrid retrieval."""

    def __init__(self, database_url: str | None = None, max_pool_size: int = 4) -> None:
        from psycopg_pool import ConnectionPool

        if database_url is None:
            database_url = database_url_from_environment()

        def configure(conn: Any) -> None:
            from pgvector.psycopg import register_vector

            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)

        self._pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=max_pool_size,
            kwargs={"autocommit": True},
            configure=configure,
            open=True,
        )
        self._bm25: Any = None  # built on load_corpus
        self._chunks_by_id: dict[str, Chunk] = {}
        self._chunks: list[Chunk] = []

    # ------------------------------------------------------------------ #
    # Schema + writes
    # ------------------------------------------------------------------ #

    def apply_schema(self) -> None:
        """Create the extension + tables + indexes if missing."""
        from rag_control_cite.schema import SCHEMA_SQL

        with self._pool.connection() as conn:
            conn.execute(SCHEMA_SQL)

    def upsert_chunks(self, chunks: list[Chunk], vectors: Any) -> None:
        """Insert or update chunks with their embedding vectors."""
        import numpy as np

        assert len(chunks) == vectors.shape[0]
        with self._pool.connection() as conn, conn.cursor() as cur:
            for chunk, vec in zip(chunks, vectors, strict=True):
                cur.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, source_uri, text, start_char, end_char,
                        chunk_index, doc_type, access_tags, last_modified,
                        author, title, control_id, framework, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        access_tags = EXCLUDED.access_tags,
                        last_modified = EXCLUDED.last_modified
                    """,
                    (
                        chunk.chunk_id,
                        chunk.source_uri,
                        chunk.text,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.chunk_index,
                        chunk.doc_type,
                        sorted(chunk.access_tags),
                        chunk.last_modified,
                        chunk.author,
                        chunk.title,
                        chunk.control_id,
                        chunk.framework,
                        np.asarray(vec, dtype=np.float32),
                    ),
                )

    def count_chunks(self) -> int:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------ #
    # Boot: load corpus + rebuild BM25
    # ------------------------------------------------------------------ #

    def load_corpus(self) -> list[Chunk]:
        """Load all chunks from the DB and rebuild the in-process BM25 index.

        Tolerates an unseeded DB (missing/empty table) — the app boots
        healthy and serves 503s until ``rcc seed`` populates the store.
        """
        import bm25s

        try:
            with self._pool.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT chunk_id, source_uri, text, start_char, end_char,
                           chunk_index, doc_type, access_tags, last_modified,
                           author, title, control_id, framework
                    FROM chunks
                    """
                ).fetchall()
        except Exception as exc:
            if "does not exist" in str(exc):
                return []
            raise
        chunks = [
            Chunk(
                chunk_id=r[0],
                source_uri=r[1],
                text=r[2],
                start_char=r[3],
                end_char=r[4],
                chunk_index=r[5],
                doc_type=r[6],
                access_tags=frozenset(r[7]),
                last_modified=r[8],
                author=r[9],
                title=r[10],
                control_id=r[11],
                framework=r[12],
            )
            for r in rows
        ]
        self._chunks = chunks
        self._chunks_by_id = {c.chunk_id: c for c in chunks}

        # Rebuild in-process BM25 — same control-ID tokenizer as Stage 2
        from rag_control_cite.hybrid import _CONTROL_ID_PATTERN

        corpus_tokens = bm25s.tokenize(
            [c.text for c in chunks],
            stopwords="en",
            token_pattern=_CONTROL_ID_PATTERN,
            show_progress=False,
        )
        bm25 = bm25s.BM25(k1=1.5, b=0.75)
        bm25.index(corpus_tokens, show_progress=False)
        self._bm25 = bm25
        return chunks

    # ------------------------------------------------------------------ #
    # Retrieval: dense via pgvector, BM25 in-process, RRF in-process
    # ------------------------------------------------------------------ #

    def retrieve_dense(
        self, query_vec: Any, k: int = 50, tenant: Tenant | None = None
    ) -> list[Retrieved]:
        """Dense top-k via pgvector cosine distance, pre-filtered by ACL."""
        import numpy as np

        vec = np.asarray(query_vec, dtype=np.float32)
        where, acl_params = acl_sql(tenant)
        params: list[Any] = [vec, *acl_params, vec, k]
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                params,
            ).fetchall()
        out: list[Retrieved] = []
        for r in rows:
            chunk = self._chunks_by_id.get(r[0])
            if chunk is None:
                continue
            out.append(Retrieved(chunk=chunk, score=float(r[1]), rank=len(out) + 1, source="dense"))
        return out

    def retrieve_bm25(
        self, query: str, k: int = 50, tenant: Tenant | None = None
    ) -> list[Retrieved]:
        """BM25 top-k from the in-process index, allow-masked by ACL."""
        assert self._bm25 is not None, "call load_corpus() first"
        import bm25s

        from rag_control_cite.acl import can_access
        from rag_control_cite.hybrid import _CONTROL_ID_PATTERN

        query_tokens = bm25s.tokenize(
            query,
            stopwords="en",
            token_pattern=_CONTROL_ID_PATTERN,
            show_progress=False,
        )
        # Over-fetch so the mask still leaves k entitled candidates.
        fetch = min(len(self._chunks), max(k * 5, k))
        results, scores = self._bm25.retrieve(query_tokens, k=fetch, show_progress=False)
        out: list[Retrieved] = []
        for ix, score in zip(results[0], scores[0], strict=True):
            chunk = self._chunks[int(ix)]
            if tenant is not None and not can_access(chunk.access_tags, tenant):
                continue
            out.append(Retrieved(chunk=chunk, score=float(score), rank=len(out) + 1, source="bm25"))
            if len(out) == k:
                break
        return out

    def retrieve_hybrid(
        self,
        query: str,
        query_vec: Any,
        k: int = 10,
        tenant: Tenant | None = None,
        k_leg: int | None = None,
    ) -> list[Retrieved]:
        """Hybrid retrieval: dense (pgvector) + BM25 (in-process), RRF fusion.

        Both legs are pre-filtered by the tenant's entitlement, so a tenant
        with a small entitled set is ranked within that set rather than
        filtered out after top-k. ``enforce_acl`` still runs after fusion
        as defence in depth.
        """
        from rag_control_cite.hybrid import rrf_fuse

        leg_k = k_leg or max(k * 5, 50)
        dense = self.retrieve_dense(query_vec, k=leg_k, tenant=tenant)
        bm25 = self.retrieve_bm25(query, k=leg_k, tenant=tenant)
        return rrf_fuse([dense, bm25], k_rrf=K_RRF, k=k)

    def close(self) -> None:
        self._pool.close()
