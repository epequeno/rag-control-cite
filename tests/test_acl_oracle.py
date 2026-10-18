"""ACL: fail-closed policy, independent oracle, planted leaks, mutation check."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_control_cite import acl
from rag_control_cite.acl import (
    ControlDims,
    EntitlementOracle,
    can_access,
    enforce_acl,
    planted_leak_chunks,
    run_acl_audit,
)
from rag_control_cite.models import Chunk, Retrieved, Tenant
from rag_control_cite.tenants import TENANTS


def _chunk(cid: str, tags: frozenset[str], framework: str = "sp-800-53") -> Chunk:
    return Chunk(
        chunk_id=f"c-{cid}",
        source_uri=f"test://{cid}",
        text=f"{cid.upper()}: title",
        start_char=0,
        end_char=10,
        chunk_index=0,
        doc_type="oscal-json",
        access_tags=tags,
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        title="title",
        control_id=cid,
        framework=framework,
    )


def test_fail_closed_missing_dimension_denied() -> None:
    tenant = Tenant(
        "t", frozenset({"sp-800-53"}), None, frozenset({"low"}), frozenset({"security"})
    )
    # Missing baseline for a restricted baseline dimension → denied
    assert can_access(frozenset({"framework:sp-800-53", "audience:security"}), tenant) is False
    # Zero tags → denied
    assert can_access(frozenset(), tenant) is False
    # Complete tags → allowed
    assert (
        can_access(frozenset({"framework:sp-800-53", "baseline:low", "audience:security"}), tenant)
        is True
    )


def test_wildcard_dimension_allows_missing() -> None:
    wildcard = Tenant("w", None, None, None, None)
    assert can_access(frozenset(), wildcard) is True


def test_planted_leaks_rejected() -> None:
    tenant = TENANTS["eng-low"]
    planted = planted_leak_chunks(tenant)
    assert len(planted) == 5
    raw = [Retrieved(chunk=c, score=1.0, rank=i + 1) for i, (c, _) in enumerate(planted)]
    served = enforce_acl(raw, tenant)
    assert served == []  # every planted leak is rejected


def test_mutation_always_true_reports_leaks() -> None:
    """Patching can_access to always-True must make the audit report leaks."""
    tenant = TENANTS["eng-low"]
    planted = planted_leak_chunks(tenant)
    retrievers = {
        "leg": lambda q, k: [
            Retrieved(chunk=c, score=1.0, rank=i + 1) for i, (c, _) in enumerate(planted)
        ]
    }
    oracle = EntitlementOracle({})

    clean = run_acl_audit(retrievers, ["q"], {"eng-low": tenant}, oracle, planted=planted)
    assert clean.n_leaked == 0
    assert clean.n_prevented > 0

    original = acl.can_access
    acl.can_access = lambda *a, **k: True  # type: ignore[assignment]
    try:
        broken = run_acl_audit(retrievers, ["q"], {"eng-low": tenant}, oracle, planted=planted)
    finally:
        acl.can_access = original  # type: ignore[assignment]
    assert broken.n_leaked > 0


def test_reference_implementation_matches() -> None:
    """Property test: can_access equals a set-theoretic reference."""
    rng = random.Random(7)
    values = {
        "framework": ["sp-800-53", "sp-800-171", "csf"],
        "family": ["ac", "au", "sc", "zz"],
        "baseline": ["low", "moderate", "high", "none"],
        "audience": ["security", "privacy"],
    }

    def reference(tags: frozenset[str], tenant: Tenant) -> bool:
        dims: dict[str, set[str]] = {k: set() for k in values}
        for t in tags:
            prefix, value = t.split(":", 1)
            dims[prefix].add(value)
        for prefix, allowed in (
            ("framework", tenant.frameworks),
            ("family", tenant.control_families),
            ("baseline", tenant.baseline_tiers),
            ("audience", tenant.audiences),
        ):
            if allowed is not None and not (dims[prefix] & set(allowed)):
                return False
        return True

    for _ in range(500):
        tags = frozenset(f"{d}:{rng.choice(values[d])}" for d in values if rng.random() < 0.7)
        tenant = Tenant(
            "t",
            None if rng.random() < 0.2 else frozenset(rng.sample(values["framework"], 1)),
            None if rng.random() < 0.2 else frozenset(rng.sample(values["family"], 2)),
            None if rng.random() < 0.2 else frozenset(rng.sample(values["baseline"], 1)),
            None if rng.random() < 0.2 else frozenset(rng.sample(values["audience"], 1)),
        )
        assert can_access(tags, tenant) == reference(tags, tenant)


@pytest.mark.skipif(not Path("data/corpus/oscal").is_dir(), reason="OSCAL corpus not fetched")
def test_oracle_agrees_with_implementation_on_corpus() -> None:
    """After the fail-closed change, oracle and implementation agree on every chunk x tenant."""
    from rag_control_cite.ingestion import ingest

    chunks, _ = ingest()
    oracle = EntitlementOracle.from_corpus()
    assert oracle.n_controls > 0
    checked = 0
    for tenant in TENANTS.values():
        for chunk in chunks:
            assert can_access(chunk.access_tags, tenant) == oracle.can_access(chunk, tenant), (
                f"disagreement on {chunk.control_id} for {tenant.tenant_id}"
            )
            checked += 1
    assert checked == len(chunks) * len(TENANTS)


def test_planted_control_dims_deny_every_tenant() -> None:
    dims = ControlDims(
        framework="sp-800-171",
        family="zz",
        baselines=frozenset({"high"}),
        audiences=frozenset({"privacy"}),
    )
    for tenant in TENANTS.values():
        for chunk, chunk_dims in planted_leak_chunks(tenant):
            assert chunk_dims == dims
            assert dims.allows(tenant) is False  # true dims deny
            assert can_access(chunk.access_tags, tenant) is False  # claimed tags deny
