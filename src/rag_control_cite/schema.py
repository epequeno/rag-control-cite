"""Schemas for the pgvector index.

The chunk table stores provenance metadata (source_uri, last_modified,
author, doc_type, access_tags, chunk_index, char offsets) alongside
the embedding vector. ACL enforcement filters on access_tags before
returning results.
"""

from __future__ import annotations

__all__ = ["SCHEMA_SQL"]

SCHEMA_SQL = """
-- Extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks with provenance metadata + embedding
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    source_uri    TEXT NOT NULL,
    text          TEXT NOT NULL,
    start_char    INTEGER NOT NULL,
    end_char      INTEGER NOT NULL,
    chunk_index   INTEGER NOT NULL,
    doc_type      TEXT NOT NULL,
    access_tags   TEXT[] NOT NULL DEFAULT '{}',
    last_modified TIMESTAMPTZ NOT NULL,
    author        TEXT,
    title         TEXT,
    control_id    TEXT,
    framework     TEXT,
    embedding     vector(3072)  -- text-embedding-3-large dimension
);

-- Exact cosine scan — no ANN index. text-embedding-3-large is 3072-dim,
-- past ivfflat's 2000-dim cap (hnsw needs halfvec, approximate). At the
-- current corpus scale exact scan is sub-millisecond anyway; add a
-- halfvec hnsw index only when the corpus approaches ~1M chunks.
-- ACL filter index
CREATE INDEX IF NOT EXISTS chunks_access_tags_idx
    ON chunks USING gin (access_tags);

-- Source lookup
CREATE INDEX IF NOT EXISTS chunks_source_uri_idx
    ON chunks (source_uri);
"""
