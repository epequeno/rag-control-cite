"""Generation + citations via OpenRouter.

The answer cites the exact spans it used. Citations are selected by the
control IDs parsed from the answer and matched against the retrieved
chunks (see ``citations.py``); the cited span is returned whole, never a
truncated prefix.

Usage is returned per call on the ``QueryResult`` — never accumulated on
a long-lived client (a cumulative counter is the bug that inflated
``/observability`` by 163x).
"""

from __future__ import annotations

from rag_control_cite.citations import select_citations
from rag_control_cite.config import OPENROUTER_BASE_URL, settings
from rag_control_cite.models import QueryResult, Retrieved, Usage

__all__ = ["Generator", "generate"]

_GENERATOR_PROMPT = """\
You are a compliance analyst answering questions about NIST security controls.
Use only the provided context. Cite each claim by control ID.

Context:
{context}

Question: {query}

Answer (cite control IDs from the context):
"""


def _assemble_context(retrieved: list[Retrieved], max_chunks: int = 5) -> str:
    """Render retrieved chunks into a context string for the generator."""
    lines: list[str] = []
    for r in retrieved[:max_chunks]:
        lines.append(f"[{r.chunk.title}] ({r.chunk.source_uri})")
        lines.append(r.chunk.text)
        lines.append("")
    return "\n".join(lines)


class Generator:
    """OpenRouter chat completion generator."""

    def __init__(self, model: str = "", api_key: str = "") -> None:
        from openai import OpenAI

        if not model:
            model = settings.generator_model
        if not api_key:
            api_key = settings.openrouter_api_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for generation")
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    def generate(
        self,
        query: str,
        retrieved: list[Retrieved],
        tenant_id: str = "",
        max_chunks: int = 5,
    ) -> QueryResult:
        """Generate an answer with citations from retrieved chunks.

        ``QueryResult.usage`` carries this call's tokens — read it, do not
        read a counter off the client.
        """
        context = _assemble_context(retrieved, max_chunks)
        prompt = _GENERATOR_PROMPT.format(context=context, query=query)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        answer = response.choices[0].message.content or ""

        citations = select_citations(retrieved, answer)
        return QueryResult(
            query=query,
            answer=answer,
            citations=citations,
            retrieved=retrieved,
            tenant_id=tenant_id,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                model=self.model,
            ),
        )


def generate(
    query: str,
    retrieved: list[Retrieved],
    tenant_id: str = "",
) -> QueryResult:
    """Generate an answer with citations from retrieved chunks."""
    return Generator().generate(query, retrieved, tenant_id)
