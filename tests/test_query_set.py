"""Query set: exact-header gold, junk rejection, stratification, dev/test split."""

from __future__ import annotations

from datetime import UTC, datetime

from rag_control_cite.models import Chunk
from rag_control_cite.query_set import (
    generate_exact_term_queries,
    split_queries,
    validate_control_id,
)


def _chunk(
    cid: str | None, framework: str, family: str, title: str = "Title", suffix: str = ""
) -> Chunk:
    tag_cid = cid or "none"
    return Chunk(
        chunk_id=f"c-{tag_cid}{suffix}",
        source_uri=f"oscal://{framework}",
        text=f"{(cid or '').upper()}: {title}{suffix}",
        start_char=0,
        end_char=10,
        chunk_index=0,
        doc_type="oscal-json",
        access_tags=frozenset(
            {f"framework:{framework}", f"family:{family}", "baseline:low", "audience:security"}
        ),
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        title=title,
        control_id=cid,
        framework=framework,
    )


def test_id_validation() -> None:
    assert validate_control_id("ac-2", "sp-800-53")
    assert validate_control_id("ac-2.3", "sp-800-53")
    assert not validate_control_id("3", "csf")
    assert validate_control_id("03.01.01", "sp-800-171")
    assert validate_control_id("gv.oc-01", "csf")
    assert not validate_control_id("ac-2", "csf")


def test_exact_gold_includes_multi_chunk_control() -> None:
    chunks = [
        _chunk("ac-2", "sp-800-53", "ac", suffix=""),
        _chunk("ac-2", "sp-800-53", "ac", suffix="-2"),
        # Decoy: same ID text, different control_id — must not be gold.
        _chunk("ac-9", "sp-800-53", "ac", title="mentions AC-2", suffix=""),
    ]
    queries = generate_exact_term_queries(chunks)
    by_id = [q for q in queries if q.category == "by_id" and q.control_id == "ac-2"]
    assert len(by_id) == 1
    assert by_id[0].gold_chunk_ids == frozenset({"c-ac-2", "c-ac-2-2"})
    assert "c-ac-9" not in by_id[0].gold_chunk_ids


def test_junk_control_rejected() -> None:
    chunks = [
        _chunk("ac-2", "sp-800-53", "ac"),
        _chunk("3", "csf", "gv", title="{{ insert"),
        _chunk(None, "csf", "gv"),
    ]
    queries = generate_exact_term_queries(chunks)
    control_ids = {q.control_id for q in queries if q.category == "by_id"}
    assert control_ids == {"ac-2"}


def test_stratification_per_group() -> None:
    chunks = []
    for i in range(5):
        chunks.append(_chunk(f"ac-{i + 1}", "sp-800-53", "ac"))
    for i in range(5):
        chunks.append(_chunk(f"au-{i + 1}", "sp-800-53", "au"))
    for i in range(5):
        chunks.append(_chunk(f"{i + 1:02d}.01.01", "sp-800-171", "03"))
    queries = generate_exact_term_queries(chunks, per_group=2)
    by_id = [q for q in queries if q.category == "by_id"]
    # Three (framework, family) groups, at most 2 controls each.
    assert len(by_id) == 6
    assert len({q.control_id for q in by_id}) == 6
    assert {q.framework for q in by_id} == {"sp-800-53", "sp-800-171"}


def test_dev_test_split_disjoint_and_total() -> None:
    chunks = [_chunk(f"ac-{i}", "sp-800-53", "ac") for i in range(1, 21)]
    queries = generate_exact_term_queries(chunks)
    dev = split_queries(queries, "dev")
    test = split_queries(queries, "test")
    assert {q.control_id for q in dev}.isdisjoint({q.control_id for q in test})
    assert len(dev) + len(test) == len(queries)


def test_by_family_is_separate_category() -> None:
    chunks = [_chunk("ac-1", "sp-800-53", "ac"), _chunk("ac-2", "sp-800-53", "ac")]
    queries = generate_exact_term_queries(chunks)
    by_family = [q for q in queries if q.category == "by_family"]
    assert len(by_family) == 1
    assert by_family[0].gold_chunk_ids == frozenset({"c-ac-1", "c-ac-2"})
