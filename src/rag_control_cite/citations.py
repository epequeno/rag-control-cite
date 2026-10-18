"""Citation selection — by control ID parsed from the answer, not by title.

The generator prompt asks the model to cite control IDs. The citation
builder must therefore parse control IDs out of the answer, and match
them against the retrieved chunks' header IDs — a chunk whose prose
title happens to appear in the answer is not a citation.

A cited ID that matches no retrieved chunk is a hallucinated citation;
``citation_metrics`` counts it rather than hiding it.

Offsets: ``start_char``/``end_char`` index the *extracted control text*
(the synthesized ``ID: Title`` body the chunker ran over), not the OSCAL
file at ``source_uri``. The README states this.
"""

from __future__ import annotations

import re

from rag_control_cite.models import Citation, Retrieved

__all__ = [
    "chunk_control_id",
    "citation_metrics",
    "parse_cited_control_ids",
    "select_citations",
]

# Control-ID shapes present in the corpus:
#   sp-800-53: ac-2, ac-2.1, au-6(1)
#   sp-800-171: 03.01.01
#   csf: gv.oc-01
_CONTROL_ID_RE = re.compile(
    r"(?ix)"
    r"\b(?:"
    r"[a-z]{2}-\d+(?:\.\d+)?(?:\(\d+\))?"
    r"|\d{2}\.\d{2}\.\d{2}"
    r"|[a-z]{2}\.[a-z]{2}-\d{2}"
    r")"
    r"(?!\w)"  # not a trailing \b — ")" is a non-word char, so \b would truncate "(1)"
)


def parse_cited_control_ids(answer: str) -> list[str]:
    """Return cited control IDs, lower-cased, de-duplicated, in order."""
    seen: dict[str, None] = {}
    for match in _CONTROL_ID_RE.findall(answer):
        seen.setdefault(match.lower(), None)
    return list(seen)


def chunk_control_id(chunk) -> str:
    """The chunk's control ID — its metadata, else its header line."""
    if getattr(chunk, "control_id", None):
        return chunk.control_id.lower()
    first_line = chunk.text.split("\n", 1)[0]
    if ": " in first_line:
        return first_line.split(": ", 1)[0].strip().lower()
    return first_line.strip().lower()


def select_citations(retrieved: list[Retrieved], answer: str) -> list[Citation]:
    """Citations for retrieved chunks whose control ID the answer cites.

    Returns the cited chunk span (full chunk text, not a truncated
    prefix); a cited ID with no matching retrieved chunk yields nothing.
    """
    cited = set(parse_cited_control_ids(answer))
    citations: list[Citation] = []
    for r in retrieved:
        if chunk_control_id(r.chunk) in cited:
            citations.append(
                Citation(
                    chunk_id=r.chunk.chunk_id,
                    source_uri=r.chunk.source_uri,
                    start_char=r.chunk.start_char,
                    end_char=r.chunk.end_char,
                    text=r.chunk.text,
                )
            )
    return citations


def citation_metrics(retrieved: list[Retrieved], answer: str) -> dict[str, float | int]:
    """Citation accuracy: hallucinated IDs and answered-with-a-citation rate.

    ``retrieved`` is already the entitled, served set.
    """
    cited = set(parse_cited_control_ids(answer))
    retrieved_ids = {chunk_control_id(r.chunk) for r in retrieved}
    hallucinated = cited - retrieved_ids
    return {
        "n_cited_ids": len(cited),
        "n_hallucinated_ids": len(hallucinated),
        "n_cited_ids_present": len(cited & retrieved_ids),
        "cites_entitled_chunk": 1 if (cited & retrieved_ids) else 0,
    }
