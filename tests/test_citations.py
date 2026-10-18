"""Citations: parse control IDs from the answer, return the cited span."""

from __future__ import annotations

from datetime import UTC, datetime

from rag_control_cite.citations import (
    citation_metrics,
    parse_cited_control_ids,
    select_citations,
)
from rag_control_cite.models import Chunk, Retrieved


def _r(cid: str) -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=f"c-{cid}",
            source_uri="oscal://cat",
            text=f"{cid.upper()}: Title for {cid}\nbody text",
            start_char=10,
            end_char=40,
            chunk_index=0,
            doc_type="oscal-json",
            access_tags=frozenset({"framework:sp-800-53"}),
            last_modified=datetime(2026, 1, 1, tzinfo=UTC),
            title=f"Title for {cid}",
            control_id=cid,
        ),
        score=1.0,
        rank=1,
    )


def test_parse_formats() -> None:
    answer = "See AC-2, AU-6(1), 03.01.01, GV.OC-01, and ac-2.3. Also AC-2 again."
    assert parse_cited_control_ids(answer) == ["ac-2", "au-6(1)", "03.01.01", "gv.oc-01", "ac-2.3"]


def test_select_returns_cited_span_not_prefix() -> None:
    retrieved = [_r("ac-2"), _r("au-6")]
    citations = select_citations(retrieved, "AC-2 is relevant; AU-9 is not.")
    assert len(citations) == 1
    cite = citations[0]
    assert cite.chunk_id == "c-ac-2"
    # The span is the chunk's full span, not text[:200]
    assert (cite.start_char, cite.end_char) == (10, 40)
    assert cite.text == retrieved[0].chunk.text


def test_no_citation_answer() -> None:
    assert select_citations([_r("ac-2")], "I do not know.") == []


def test_hallucinated_id_metric() -> None:
    retrieved = [_r("ac-2")]
    metrics = citation_metrics(retrieved, "AC-2 and Section 9 apply.")
    assert metrics["n_cited_ids_present"] == 1
    assert metrics["n_hallucinated_ids"] == 0  # "Section 9" is not an ID shape

    metrics2 = citation_metrics(retrieved, "AC-2 and SC-8 apply.")
    assert metrics2["n_hallucinated_ids"] == 1  # SC-8 not retrieved
    assert metrics2["cites_entitled_chunk"] == 1


def test_citation_of_non_retrieved_control_yields_nothing() -> None:
    assert select_citations([_r("ac-2")], "SC-8 applies.") == []
