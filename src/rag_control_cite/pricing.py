"""One price table, keyed by model id — measured cost, not constants.

Every cost figure in the repo resolves through ``price_for``. An unknown
model id is a hard error: a silent default price is how a stale constant
ends up as a headline number.

Prices are USD per 1M tokens, taken from the provider's published list
price. Each row records its source and the date it was read.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PRICES", "ModelPrice", "UnknownModelError", "cost_usd", "price_for"]


class UnknownModelError(ValueError):
    """Raised when a model id has no price entry."""


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """List price for one model, with provenance."""

    model: str
    prompt_per_m: float  # USD per 1M input tokens
    completion_per_m: float  # USD per 1M output tokens
    source: str
    as_of: str  # ISO date the price was read


# Read 2026-10-16 from the providers' published pricing pages:
#   OpenAI list prices: https://developers.openai.com/api/docs/pricing
#   OpenRouter embedding page: https://openrouter.ai/openai/text-embedding-3-large
PRICES: dict[str, ModelPrice] = {
    "openai/text-embedding-3-large": ModelPrice(
        model="openai/text-embedding-3-large",
        prompt_per_m=0.13,
        completion_per_m=0.0,  # embeddings have no completion
        source="openrouter:openai/text-embedding-3-large",
        as_of="2026-10-16",
    ),
    "openai/gpt-4o-mini": ModelPrice(
        model="openai/gpt-4o-mini",
        prompt_per_m=0.15,
        completion_per_m=0.60,
        source="openai:api/docs/pricing",
        as_of="2026-10-16",
    ),
    "openai/gpt-4o": ModelPrice(
        model="openai/gpt-4o",
        prompt_per_m=2.50,
        completion_per_m=10.00,
        source="openai:api/docs/pricing",
        as_of="2026-10-16",
    ),
}


def price_for(model: str) -> ModelPrice:
    """Return the list price for ``model`` or raise ``UnknownModelError``."""
    try:
        return PRICES[model]
    except KeyError:
        known = ", ".join(sorted(PRICES))
        raise UnknownModelError(f"no price for model {model!r}; known models: {known}") from None


def cost_usd(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """USD cost of one call at the model's list price.

    Raises ``UnknownModelError`` for an unpriced model — cost is never
    guessed.
    """
    price = price_for(model)
    return (
        prompt_tokens * price.prompt_per_m + completion_tokens * price.completion_per_m
    ) / 1_000_000
