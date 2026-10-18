"""Settings for every stage of the pipeline.

Loaded from environment variables (or `.env`). The one decision worth
over-investing in up front is the embedding model — switching means
re-indexing everything.

All inference (embeddings + generation) routes through OpenRouter — one
key, one provider. Reranking runs locally via sentence-transformers.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class Settings(BaseSettings):
    """Project-wide configuration."""

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}

    # --- OpenRouter (embeddings + generation, one key) ---
    openrouter_api_key: str = ""
    embedding_model: str = "openai/text-embedding-3-large"
    generator_model: str = "openai/gpt-4o-mini"
    # The judge must not be the generator: a model grading its own output
    # scores its own family favourably. Different model, different family.
    judge_model: str = "openai/gpt-4o"

    # --- reranking (local, no key) ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- vector store (pgvector; local compose maps host 5433 → 5432) ---
    database_url: str = "postgresql://rag:rag@localhost:5433/rag"

    # --- auth (static demo bearer tokens; empty = auth disabled, local only) ---
    # RCC_TENANT_TOKENS: JSON {"tenant_id": "token"}; RCC_OPERATOR_TOKEN: str.
    # Read directly from the environment (see tenants.load_tenant_tokens).

    # --- observability ---
    otel_exporter_otlp_endpoint: str = ""


settings = Settings()
