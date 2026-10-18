"""OpenRouter embedder — one key for embeddings + generation.

Uses the ``openai`` SDK pointed at OpenRouter's OpenAI-compatible
``/api/v1/embeddings`` endpoint. Returns L2-normalized float32 rows
so retrieval is a plain dot product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag_control_cite.config import OPENROUTER_BASE_URL, settings

__all__ = ["OpenRouterEmbedder"]


@dataclass
class OpenRouterEmbedder:
    """Batched OpenRouter embeddings, L2-normalized, float32."""

    model: str = ""
    batch_size: int = 256
    api_key: str = ""

    def __post_init__(self) -> None:
        from openai import OpenAI

        if not self.model:
            self.model = settings.embedding_model
        if not self.api_key:
            self.api_key = settings.openrouter_api_key
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for embeddings")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        self._client = OpenAI(api_key=self.api_key, base_url=OPENROUTER_BASE_URL)

    @property
    def id(self) -> str:
        return f"openrouter:{self.model}"

    @property
    def dim(self) -> int:
        """Embedding dimension — derived from one probe."""
        probe = self.embed(["dimension probe"])
        return probe.shape[1]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed in order, L2-normalized, float32."""
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            if len(response.data) != len(batch):
                raise ValueError(
                    f"embedder returned {len(response.data)} rows for {len(batch)} inputs"
                )
            rows.extend(item.embedding for item in response.data)
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)
