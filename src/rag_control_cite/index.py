"""In-memory index — chunks + L2-normalized vectors.

Same layout as rag-eval-project's ``Index``: build once, cache to disk,
reuse across query-time axes. For 2,115 chunks, in-memory numpy cosine
is instant; pgvector is the production target (schema ready in
``schema.py``, docker-compose ready in ``infra/docker/``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_control_cite.models import Chunk

__all__ = ["Index", "build_or_load_index", "index_fingerprint"]

INDEX_FORMAT_VERSION = "1"


@dataclass(slots=True)
class Index:
    """Chunks plus their embeddings. Build once, reuse across configs."""

    chunks: list[Chunk]
    vectors: np.ndarray
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {self.vectors.shape}")
        if self.vectors.shape[0] != len(self.chunks):
            raise ValueError(f"{self.vectors.shape[0]} vectors for {len(self.chunks)} chunks")

    def save(self, path: Path | str) -> None:
        """Write to a directory: chunks.json, vectors.npy, meta.json."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "chunks.json").write_text(json.dumps([_chunk_to_dict(c) for c in self.chunks]))
        np.save(path / "vectors.npy", self.vectors)
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "format_version": INDEX_FORMAT_VERSION,
                    "fingerprint": self.fingerprint,
                    "n_chunks": len(self.chunks),
                }
            )
        )

    @classmethod
    def load(cls, path: Path | str) -> Index:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        version = meta.get("format_version")
        if version != INDEX_FORMAT_VERSION:
            raise ValueError(f"index format {version!r} != expected {INDEX_FORMAT_VERSION!r}")
        chunks = [_chunk_from_dict(d) for d in json.loads((path / "chunks.json").read_text())]
        vectors = np.load(path / "vectors.npy")
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"corrupt index at {path}: {vectors.shape[0]} vectors, {len(chunks)} chunks"
            )
        return cls(chunks=chunks, vectors=vectors, fingerprint=meta["fingerprint"])

    def retrieve(self, query_vec: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Top-k by cosine similarity (dot product on L2-normalized vectors).

        Returns list of (chunk_index, score) sorted by score descending.
        """
        scores = self.vectors @ query_vec  # (N,) — both L2-normalized
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #


def _chunk_to_dict(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "source_uri": c.source_uri,
        "text": c.text,
        "start_char": c.start_char,
        "end_char": c.end_char,
        "chunk_index": c.chunk_index,
        "doc_type": c.doc_type,
        "access_tags": sorted(c.access_tags),
        "last_modified": c.last_modified.isoformat(),
        "author": c.author,
        "title": c.title,
        "control_id": c.control_id,
        "framework": c.framework,
    }


def _chunk_from_dict(d: dict) -> Chunk:
    from datetime import datetime

    return Chunk(
        chunk_id=d["chunk_id"],
        source_uri=d["source_uri"],
        text=d["text"],
        start_char=d["start_char"],
        end_char=d["end_char"],
        chunk_index=d["chunk_index"],
        doc_type=d["doc_type"],
        access_tags=frozenset(d["access_tags"]),
        last_modified=datetime.fromisoformat(d["last_modified"]),
        author=d.get("author"),
        title=d.get("title"),
        control_id=d.get("control_id"),
        framework=d.get("framework"),
    )


# --------------------------------------------------------------------------- #
# Fingerprint + build
# --------------------------------------------------------------------------- #


def _chunks_hash(chunks: list[Chunk]) -> str:
    """Order-independent hash of chunk contents and access tags.

    Tags are part of the hash: a tagging change (e.g. adding
    ``baseline:none``) must invalidate a cached index, or the ACL audit
    would run against stale tags.
    """
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda c: c.chunk_id):
        h.update(c.chunk_id.encode())
        h.update(b"\x00")
        h.update(c.text.encode())
        h.update(b"\x00")
        h.update("\x01".join(sorted(c.access_tags)).encode())
        h.update(b"\x00\x00")
    return h.hexdigest()


def index_fingerprint(chunks: list[Chunk], embedder_id: str) -> str:
    """Identity of an index build: chunks + embedding model."""
    payload = ":".join([_chunks_hash(chunks), embedder_id, INDEX_FORMAT_VERSION])
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def build_or_load_index(
    chunks: list[Chunk],
    embedder: object,
    cache_dir: Path | str = ".index_cache",
) -> Index:
    """Embed all chunks (or load from cache if fingerprint matches).

    Args:
        chunks: the chunks to embed.
        embedder: an object with ``embed(texts) -> ndarray`` and ``.id`` attr.
        cache_dir: where to cache the index.

    Returns:
        Index with L2-normalized vectors.
    """
    fp = index_fingerprint(chunks, embedder.id)  # type: ignore[attr-defined]
    path = Path(cache_dir) / fp
    if (path / "meta.json").exists():
        cached = Index.load(path)
        if cached.fingerprint == fp:
            return cached

    texts = [c.text for c in chunks]
    vectors = embedder.embed(texts)  # type: ignore[attr-defined]
    index = Index(chunks=chunks, vectors=vectors, fingerprint=fp)
    index.save(path)
    return index
