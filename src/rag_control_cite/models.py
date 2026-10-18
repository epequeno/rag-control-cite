"""Data types crossing every stage boundary.

Chunk carries provenance metadata (source_uri, last_modified, author,
doc_type, access_tags, chunk_index, char offsets) so that ACL enforcement
and citation can both reach back to the exact span.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Document:
    """A parsed source document — one OSCAL control or one PDF section.

    The chunker operates on ``text``; every other field is provenance
    that flows through to each chunk.
    """

    doc_id: str
    source_uri: str
    text: str
    title: str
    doc_type: str
    access_tags: frozenset[str]
    last_modified: datetime
    author: str | None = None
    control_id: str | None = None
    family: str | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous character span of one source document.

    ``start_char`` / ``end_char`` index into the source text, which makes
    citation exact and gold-containment auditable.
    """

    chunk_id: str
    source_uri: str
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    doc_type: str
    access_tags: frozenset[str]
    last_modified: datetime
    author: str | None = None
    title: str | None = None
    control_id: str | None = None
    framework: str | None = None

    @property
    def char_span(self) -> tuple[int, int]:
        return (self.start_char, self.end_char)


@dataclass(frozen=True, slots=True)
class Retrieved:
    """A retrieved chunk with its score and rank.

    ``source`` names the leg that produced the hit ("dense", "bm25",
    "fused") so a hybrid loss can be attributed to a leg.
    """

    chunk: Chunk
    score: float
    rank: int
    source: str | None = None


@dataclass(frozen=True, slots=True)
class Citation:
    """A citation pointing at an exact span of a source document."""

    chunk_id: str
    source_uri: str
    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage for a single model call — per call, never cumulative.

    A cumulative counter on a long-lived client object is the bug that
    made ``/observability`` report 163x the true prompt tokens; usage
    must travel with the call that produced it.
    """

    prompt_tokens: int
    completion_tokens: int
    model: str = ""


@dataclass(slots=True)
class QueryResult:
    """End-to-end query result: answer, citations, provenance, traces."""

    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[Retrieved] = field(default_factory=list)
    tenant_id: str = ""
    trace_id: str = ""
    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class Tenant:
    """A tenant with permissioned access dimensions.

    Synthetic — the access dimensions (framework, control family, baseline
    tier, audience) are real, but the tenant assignments are invented.

    A dimension set of ``None`` is an explicit wildcard: that dimension
    imposes no restriction (chunks missing the dimension tag pass, and
    every present tag value passes). A finite set is fail-closed — the
    chunk must carry at least one tag from the set, or it is denied.
    """

    tenant_id: str
    frameworks: frozenset[str] | None
    control_families: frozenset[str] | None  # None = all
    baseline_tiers: frozenset[str] | None
    audiences: frozenset[str] | None
