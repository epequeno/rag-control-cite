"""Stage 4 — ACL enforcement, with an oracle that can fail.

Every chunk carries ``access_tags`` (``frozenset[str]``), namespaced by
dimension: ``framework:sp-800-53``, ``family:ac``, ``baseline:low``,
``audience:security``. Enforcement is **fail-closed**:

  * a dimension set of ``None`` on the tenant is an explicit wildcard —
    that dimension imposes no restriction;
  * a finite set requires the chunk to carry at least one tag from the
    set. A chunk missing tags for a restricted dimension — including a
    chunk with zero tags — is denied.

``can_access`` implements the policy. The audit must not: an oracle that
reads the same tags it is checking passes even when ``can_access`` is
patched to always return ``True`` (that was the original Stage 4 finding).
``EntitlementOracle`` derives each control's dimensions from the raw
OSCAL catalogs and baseline profiles — never from ``access_tags`` — so a
passing audit means something.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rag_control_cite.models import Chunk, Retrieved, Tenant

__all__ = [
    "ACLAuditReport",
    "ControlDims",
    "EntitlementOracle",
    "can_access",
    "enforce_acl",
    "run_acl_audit",
]


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


def _tags_for_prefix(tags: frozenset[str], prefix: str) -> set[str]:
    """Extract values from tags with a given prefix (e.g. 'framework:')."""
    return {t[len(prefix) :] for t in tags if t.startswith(prefix)}


def _dimension_allows(chunk_values: set[str], allowed: frozenset[str] | None) -> bool:
    """Fail-closed dimension check.

    ``None`` = wildcard (always allowed). A finite set requires a
    non-empty intersection — a chunk with no value for the dimension is
    denied.
    """
    if allowed is None:
        return True
    return bool(chunk_values & allowed)


def can_access(chunk_tags: frozenset[str], tenant: Tenant) -> bool:
    """Fail-closed check: may a tenant see a chunk with these access tags?

    Every restricted dimension must have a matching tag. A missing
    dimension tag (or a chunk with zero tags) is denied unless the tenant
    holds that dimension as a wildcard (``None``).
    """
    return (
        _dimension_allows(_tags_for_prefix(chunk_tags, "framework:"), tenant.frameworks)
        and _dimension_allows(_tags_for_prefix(chunk_tags, "family:"), tenant.control_families)
        and _dimension_allows(_tags_for_prefix(chunk_tags, "baseline:"), tenant.baseline_tiers)
        and _dimension_allows(_tags_for_prefix(chunk_tags, "audience:"), tenant.audiences)
    )


def enforce_acl(retrieved: list[Retrieved], tenant: Tenant) -> list[Retrieved]:
    """Filter retrieved chunks by tenant ACL and re-rank (1-based, in order)."""
    allowed = [r for r in retrieved if can_access(r.chunk.access_tags, tenant)]
    return [
        Retrieved(chunk=item.chunk, score=item.score, rank=rank, source=item.source)
        for rank, item in enumerate(allowed, start=1)
    ]


# --------------------------------------------------------------------------- #
# Independent oracle — entitlement from raw OSCAL, never from access_tags
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ControlDims:
    """A control's true access dimensions, derived from source, not tags."""

    framework: str
    family: str
    baselines: frozenset[str]
    audiences: frozenset[str]

    def allows(self, tenant: Tenant) -> bool:
        """Apply the same fail-closed rule, using oracle-derived dims."""
        return (
            _dimension_allows({self.framework}, tenant.frameworks)
            and _dimension_allows({self.family}, tenant.control_families)
            and _dimension_allows(set(self.baselines), tenant.baseline_tiers)
            and _dimension_allows(set(self.audiences), tenant.audiences)
        )


def _control_id_of(chunk: Chunk) -> str:
    """The chunk's control ID — its metadata, else its ``ID: Title`` header."""
    if chunk.control_id:
        return chunk.control_id.lower()
    first_line = chunk.text.split("\n", 1)[0]
    if ": " in first_line:
        return first_line.split(": ", 1)[0].strip().lower()
    return first_line.strip().lower()


class EntitlementOracle:
    """Independent ACL oracle: entitlement computed from raw OSCAL sources.

    ``controls`` maps a lower-cased control ID to its true dimensions.
    A chunk whose header control ID is unknown is denied (fail-closed).
    """

    def __init__(self, controls: dict[str, ControlDims]) -> None:
        self._controls = dict(controls)

    def register(self, control_id: str, dims: ControlDims) -> None:
        """Register a control's dimensions (used to plant synthetic leaks)."""
        self._controls[control_id.lower()] = dims

    @property
    def n_controls(self) -> int:
        return len(self._controls)

    def dims_for_chunk(self, chunk: Chunk) -> ControlDims | None:
        return self._controls.get(_control_id_of(chunk))

    def can_access(self, chunk: Chunk, tenant: Tenant) -> bool:
        """Oracle verdict: may the tenant see this chunk?"""
        dims = self.dims_for_chunk(chunk)
        if dims is None:
            return False  # unknown control — fail closed
        return dims.allows(tenant)

    @classmethod
    def from_corpus(cls, corpus_dir: Path | str = "data/corpus/oscal") -> EntitlementOracle:
        """Build the oracle by parsing the raw OSCAL catalogs + profiles."""
        from rag_control_cite.ingestion import (
            _BASELINE_FILES,
            _CATALOGS,
            _baseline_tiers_for_control,
            _load_baseline_ids,
        )

        corpus = Path(corpus_dir).expanduser()
        baseline_dir = corpus / "nist.gov" / "SP800-53" / "rev5" / "json"
        baseline_maps: dict[str, set[str]] = {}
        for tier, fname in _BASELINE_FILES.items():
            path = baseline_dir / fname
            if path.exists():
                baseline_maps[tier] = _load_baseline_ids(path)

        controls: dict[str, ControlDims] = {}

        def register(control: dict, framework: str, family: str) -> None:
            cid = control["id"].lower()
            tiers = _baseline_tiers_for_control(cid, baseline_maps)
            audiences = {"security"}
            if "privacy" in tiers:
                audiences.add("privacy")
            controls[cid] = ControlDims(
                framework=framework,
                family=family,
                baselines=frozenset(tiers),
                audiences=frozenset(audiences),
            )

        for framework, rel_path in _CATALOGS:
            cat_path = corpus / rel_path
            if not cat_path.exists():
                continue
            data = json.loads(cat_path.read_text())
            catalog = data.get("catalog", data)
            for group in catalog.get("groups", []):
                family = group.get("id", "").lower()
                for control in group.get("controls", []):
                    register(control, framework, family)
                    for sub in control.get("controls", []):
                        register(sub, framework, family)

        return cls(controls)


# --------------------------------------------------------------------------- #
# Audit over real retrievals — chunk units, planted leaks, mutation-checked
# --------------------------------------------------------------------------- #

RetrieveFn = Callable[[str, int], list[Retrieved]]


@dataclass
class ACLAuditReport:
    """Stage 4 measurement — chunk-unit leak check over real retrievals."""

    n_checked: int = 0
    n_prevented: int = 0
    n_leaked: int = 0
    combos: list[dict[str, object]] = field(default_factory=list)

    @property
    def prevented_rate(self) -> float:
        """Fraction of unauthorized candidates that enforcement removed."""
        unauthorized = self.n_prevented + self.n_leaked
        if unauthorized == 0:
            return 1.0
        return self.n_prevented / unauthorized

    def as_dict(self) -> dict[str, object]:
        return {
            "n_checked": self.n_checked,
            "n_prevented": self.n_prevented,
            "n_leaked": self.n_leaked,
            "prevented_rate": round(self.prevented_rate, 4),
            "n_combos": len(self.combos),
        }


def planted_leak_chunks(tenant: Tenant) -> list[tuple[Chunk, ControlDims]]:
    """Synthetic chunks whose tags AND true dims are unauthorized for ``tenant``.

    Five kinds — wrong framework, baseline, audience, family, and a
    zero-tag chunk — each of which enforcement must reject. A run that
    leaks any of these proves the audit itself can fail.
    """
    import hashlib
    from datetime import UTC, datetime

    def make(name: str, tags: frozenset[str]) -> Chunk:
        cid = f"planted-{tenant.tenant_id}-{name}"
        return Chunk(
            chunk_id=hashlib.sha256(cid.encode()).hexdigest()[:16],
            source_uri="planted://leak",
            text=f"PLANTED-{name.upper()}: planted leak control",
            start_char=0,
            end_char=40,
            chunk_index=0,
            doc_type="planted",
            access_tags=tags,
            last_modified=datetime(2026, 1, 1, tzinfo=UTC),
            title=f"PLANTED-{name}",
        )

    # True dims: framework sp-800-171, family zz, baseline high, audience privacy.
    # Unauthorized for every real tenant (which use sp-800-53 and low/mod/high).
    dims = ControlDims(
        framework="sp-800-171",
        family="zz",
        baselines=frozenset({"high"}),
        audiences=frozenset({"privacy"}),
    )
    planted: list[tuple[Chunk, ControlDims]] = [
        (
            make("framework", frozenset({"framework:sp-800-171", "audience:security"})),
            dims,
        ),
        (
            make("baseline", frozenset({"framework:sp-800-53", "baseline:high"})),
            dims,
        ),
        (
            make("audience", frozenset({"framework:sp-800-53", "audience:privacy"})),
            dims,
        ),
        (
            make("family", frozenset({"framework:sp-800-53", "family:zz"})),
            dims,
        ),
        (make("zerotag", frozenset()), dims),
    ]
    return planted


def run_acl_audit(
    retrievers: dict[str, RetrieveFn],
    queries: list[str],
    tenants: dict[str, Tenant],
    oracle: EntitlementOracle,
    planted: list[tuple[Chunk, ControlDims]] | None = None,
    k: int = 10,
) -> ACLAuditReport:
    """Measure leakage in chunk units over real retrievals.

    For every (leg, query, tenant): retrieve without ACL, inject the
    planted-leak controls, run ``enforce_acl``, then check each served
    chunk against the oracle. ``n_leaked`` counts served chunks the oracle
    denies; ``n_prevented`` counts unauthorized candidates dropped.
    """
    report = ACLAuditReport()
    planted = planted or []
    for chunk, dims in planted:
        oracle.register(_control_id_of(chunk), dims)

    for leg_name, retrieve in retrievers.items():
        for tenant_id, tenant in tenants.items():
            for query in queries:
                raw = list(retrieve(query, k))
                injected = [
                    Retrieved(chunk=c, score=0.0, rank=0, source="planted") for c, _ in planted
                ]
                served = enforce_acl(raw + injected, tenant)

                served_ids = {r.chunk.chunk_id for r in served}
                n_checked = len(served)
                n_leaked = sum(1 for r in served if not oracle.can_access(r.chunk, tenant))

                unauthorized = [r for r in raw + injected if not oracle.can_access(r.chunk, tenant)]
                n_prevented = sum(1 for r in unauthorized if r.chunk.chunk_id not in served_ids)

                report.n_checked += n_checked
                report.n_prevented += n_prevented
                report.n_leaked += n_leaked
                report.combos.append(
                    {
                        "leg": leg_name,
                        "tenant": tenant_id,
                        "query": query,
                        "n_checked": n_checked,
                        "n_prevented": n_prevented,
                        "n_leaked": n_leaked,
                    }
                )
    return report
