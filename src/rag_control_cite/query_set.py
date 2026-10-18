"""Exact-term query set — generated from OSCAL control IDs.

The codes-and-IDs case dense search misses: AC-2, AU-6(1), SC-8. Each
query asks about a control by ID; the gold is the chunk(s) with that
``control_id`` (multi-chunk controls included), not a substring match on
the first 100 characters.

Control IDs are validated against the shapes actually present in each
framework; anything else is dropped (a malformed ``3. {{ insert`` header
is not a control).

Queries are stratified across framework and family so the set covers the
catalogue rather than the first 50 controls, and split into a dev half
(tokenizer/weights are chosen on dev) and a test half (the headline).
``by_family`` queries move to a separate metric — their recall@10 ceiling
is ~0.16 because a family tag matches every control in the family.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rag_control_cite.models import Chunk

__all__ = ["Query", "generate_exact_term_queries", "split_queries"]

# Control-ID shapes present in the corpus, keyed by framework tag.
ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "sp-800-53": re.compile(r"^[a-z]{2}-\d+(?:\.\d+)?(?:\(\d+\))?$"),
    "sp-800-171": re.compile(r"^\d{2}\.\d{2}\.\d{2}$"),
    "csf": re.compile(r"^[a-z]{2}\.[a-z]{2}-\d{2}$"),
}


@dataclass(frozen=True, slots=True)
class Query:
    """One eval query with gold chunk IDs and its dev/test split."""

    query: str
    gold_chunk_ids: frozenset[str]
    control_id: str
    category: str  # "by_id", "by_name", "by_family"
    framework: str = ""
    family: str = ""
    split: str = "test"  # "dev" or "test"


def validate_control_id(control_id: str, framework: str) -> bool:
    """Whether ``control_id`` matches its framework's ID shape."""
    pattern = ID_PATTERNS.get(framework)
    return pattern is not None and pattern.match(control_id) is not None


def _split_for(control_id: str) -> str:
    """Deterministic dev/test split, stable across runs."""
    digest = hashlib.sha1(control_id.encode()).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


def _tag_value(tags: frozenset[str], prefix: str) -> str:
    return next((t.split(":", 1)[1] for t in tags if t.startswith(prefix)), "")


@dataclass(frozen=True, slots=True)
class _Control:
    control_id: str
    title: str
    framework: str
    family: str


def _control_index(chunks: list[Chunk]) -> dict[str, _Control]:
    controls: dict[str, _Control] = {}
    for c in chunks:
        cid = (c.control_id or "").lower()
        if not cid or cid in controls:
            continue
        controls[cid] = _Control(
            control_id=cid,
            title=c.title or cid,
            framework=c.framework or _tag_value(c.access_tags, "framework:"),
            family=_tag_value(c.access_tags, "family:"),
        )
    return controls


def _gold_for(chunks: list[Chunk], control_id: str) -> frozenset[str]:
    """All chunk IDs whose ``control_id`` matches — exact, no substring."""
    return frozenset(c.chunk_id for c in chunks if (c.control_id or "").lower() == control_id)


def generate_exact_term_queries(
    chunks: list[Chunk],
    per_group: int = 8,
) -> list[Query]:
    """Generate stratified exact-term queries from the corpus control IDs.

    Args:
        chunks: the corpus chunks.
        per_group: max by_id controls sampled per (framework, family).

    Returns:
        Queries with exact gold, split into dev/test.
    """
    controls = _control_index(chunks)

    # Stratify: group by (framework, family), sample deterministically.
    groups: dict[tuple[str, str], list[_Control]] = {}
    for control in controls.values():
        if not validate_control_id(control.control_id, control.framework):
            continue
        if not _gold_for(chunks, control.control_id):
            continue
        groups.setdefault((control.framework, control.family), []).append(control)

    sampled: list[_Control] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda c: c.control_id)
        sampled.extend(members[:per_group])

    queries: list[Query] = []
    for control in sampled:
        gold = _gold_for(chunks, control.control_id)
        split = _split_for(control.control_id)
        queries.append(
            Query(
                query=f"What does control {control.control_id.upper()} require?",
                gold_chunk_ids=gold,
                control_id=control.control_id,
                category="by_id",
                framework=control.framework,
                family=control.family,
                split=split,
            )
        )
        queries.append(
            Query(
                query=f"What is {control.title}?",
                gold_chunk_ids=gold,
                control_id=control.control_id,
                category="by_name",
                framework=control.framework,
                family=control.family,
                split=split,
            )
        )

    # by_family — one query per (framework, family); gold is the family.
    for key in sorted(groups):
        framework, family = key
        members = groups[key]
        gold = frozenset(
            c.chunk_id
            for c in chunks
            if _tag_value(c.access_tags, "family:") == family
            and _tag_value(c.access_tags, "framework:") == framework
        )
        if not gold:
            continue
        queries.append(
            Query(
                query=f"What are the {members[0].title} requirements?",
                gold_chunk_ids=gold,
                control_id=family,
                category="by_family",
                framework=framework,
                family=family,
                split=_split_for(f"{framework}:{family}"),
            )
        )

    return queries


def split_queries(queries: list[Query], split: str) -> list[Query]:
    """Filter queries to one split ("dev" or "test")."""
    return [q for q in queries if q.split == split]
