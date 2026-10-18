"""Pricing: one table, provenance, hard failure on an unknown model."""

from __future__ import annotations

import pytest

from rag_control_cite.pricing import PRICES, UnknownModelError, cost_usd, price_for


def test_unknown_model_fails() -> None:
    with pytest.raises(UnknownModelError):
        price_for("openai/does-not-exist")
    with pytest.raises(UnknownModelError):
        cost_usd(1000, 1000, "openai/does-not-exist")


def test_arithmetic() -> None:
    # gpt-4o-mini: $0.15/1M in, $0.60/1M out
    assert cost_usd(1_000_000, 0, "openai/gpt-4o-mini") == pytest.approx(0.15)
    assert cost_usd(0, 1_000_000, "openai/gpt-4o-mini") == pytest.approx(0.60)
    # embeddings: input only
    assert cost_usd(1_000_000, 0, "openai/text-embedding-3-large") == pytest.approx(0.13)
    assert cost_usd(0, 1_000_000, "openai/text-embedding-3-large") == 0.0


def test_every_price_has_provenance() -> None:
    for model, price in PRICES.items():
        assert price.model == model
        assert price.source
        assert price.as_of
        assert price.prompt_per_m >= 0
        assert price.completion_per_m >= 0
