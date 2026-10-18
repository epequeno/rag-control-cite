"""Groundedness judge — did the answer use the retrieved context?

Uses OpenRouter to judge whether the answer is grounded in the retrieved
context (RAGAS-style faithfulness). Returns a 0-1 score plus this call's
token usage.

The verdict parser tolerates markdown and punctuation (``**Yes**`` and
"Yes, because…" both score 1.0) — ``startswith("yes")`` mis-scored both.

The judge model is configured separately from the generator and defaults
to a different model family, so a model never grades its own output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_control_cite.config import OPENROUTER_BASE_URL, settings
from rag_control_cite.models import QueryResult, Usage

__all__ = ["Judge", "Judgement", "judge_groundedness"]

_JUDGE_PROMPT = """\
You are evaluating whether an answer is grounded in the provided context.

Context:
{context}

Question: {query}

Answer: {answer}

Is the answer fully grounded in the context above? Answer with a single
word: "yes" or "no", then explain briefly.
"""

_VERDICT_RE = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Judgement:
    """A judge verdict with the call's token usage."""

    score: float
    usage: Usage

    @property
    def grounded(self) -> bool:
        return self.score >= 1.0


def parse_verdict(text: str) -> float:
    """Parse a yes/no verdict robustly: markdown, punctuation, casing."""
    match = _VERDICT_RE.match(text.strip())
    if match is None:
        return 0.0
    return 1.0 if match.group(1).lower() == "yes" else 0.0


class Judge:
    """OpenRouter-based groundedness judge."""

    def __init__(self, model: str = "", api_key: str = "") -> None:
        from openai import OpenAI

        if not model:
            model = settings.judge_model
        if not api_key:
            api_key = settings.openrouter_api_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for the judge")
        if model == settings.generator_model:
            raise ValueError(
                f"judge model must differ from the generator model (both are {model!r})"
            )
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    def judge_groundedness(self, result: QueryResult, max_chunks: int = 5) -> Judgement:
        """Judge groundedness; return the verdict and this call's usage."""
        context = "\n".join(r.chunk.text for r in result.retrieved[:max_chunks])
        prompt = _JUDGE_PROMPT.format(
            context=context,
            query=result.query,
            answer=result.answer,
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
        )
        verdict = response.choices[0].message.content or ""
        return Judgement(
            score=parse_verdict(verdict),
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                model=self.model,
            ),
        )


def judge_groundedness(result: QueryResult) -> Judgement:
    """Judge whether an answer is grounded in its retrieved context."""
    return Judge().judge_groundedness(result)
