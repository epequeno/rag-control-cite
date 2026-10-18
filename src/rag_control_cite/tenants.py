"""Synthetic tenants + demo credentials — invented access-control dimensions.

The frameworks, control families, baseline tiers, and audiences are
real OSCAL dimensions. The tenant assignments are synthetic. This is
the seam: nothing here reads as a hidden claim.

Audience values match what chunks carry: ``security`` (all OSCAL
controls) and ``privacy`` (privacy-baseline controls). The tenant's
audience field gates which audience-tagged chunks they may see.

Fail-closed policy: a dimension set of ``None`` is an explicit wildcard;
a finite set requires the chunk to carry a matching tag, or the chunk is
denied. A chunk in no baseline carries ``baseline:none``, which matches
no tenant's finite baseline set — non-baselined controls are visible only
to a tenant granted the baseline wildcard.

Credentials are static demo bearer tokens (one per tenant) — NOT an
identity-provider integration. ``RCC_TENANT_TOKENS`` is a JSON object
mapping ``tenant_id`` → token; when it is empty, authentication is
disabled for local runs (and the README says so).
"""

from __future__ import annotations

import json
import os

from rag_control_cite.models import Tenant

# Synthetic tenants exercising the four access dimensions:
#   framework, control family, baseline tier, audience
TENANTS: dict[str, Tenant] = {
    "eng-low": Tenant(
        tenant_id="eng-low",
        frameworks=frozenset({"sp-800-53"}),
        control_families=None,  # all families
        baseline_tiers=frozenset({"low"}),
        audiences=frozenset({"security"}),
    ),
    "sec-moderate": Tenant(
        tenant_id="sec-moderate",
        frameworks=frozenset({"sp-800-53", "sp-800-171"}),
        control_families=None,
        baseline_tiers=frozenset({"moderate"}),
        audiences=frozenset({"security"}),
    ),
    "priv-high": Tenant(
        tenant_id="priv-high",
        frameworks=frozenset({"sp-800-53"}),
        control_families=frozenset({"ac", "au", "sc"}),
        baseline_tiers=frozenset({"high"}),
        audiences=frozenset({"privacy"}),
    ),
    "sec-high": Tenant(
        tenant_id="sec-high",
        frameworks=frozenset({"sp-800-53"}),
        control_families=None,
        baseline_tiers=frozenset({"high"}),
        audiences=frozenset({"security"}),
    ),
}

# Evaluation tenant: unrestricted across every dimension. Used by the
# offline retrieval/eval paths, which measure ranking quality, not access.
EVAL_TENANT = Tenant(
    tenant_id="eval",
    frameworks=None,
    control_families=None,
    baseline_tiers=None,
    audiences=None,
)

__all__ = ["EVAL_TENANT", "TENANTS", "load_operator_token", "load_tenant_tokens"]


def load_tenant_tokens() -> dict[str, str]:
    """Return ``{token: tenant_id}`` from ``RCC_TENANT_TOKENS``.

    The env var holds a JSON object mapping ``tenant_id`` → token. An
    empty or unset value disables authentication (local runs only).
    """
    raw = os.environ.get("RCC_TENANT_TOKENS", "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("RCC_TENANT_TOKENS must be a JSON object of tenant_id -> token")
    return {str(token): str(tenant) for tenant, token in data.items()}


def load_operator_token() -> str:
    """Return the operator token gating ``/observability`` ("" = open)."""
    return os.environ.get("RCC_OPERATOR_TOKEN", "").strip()
