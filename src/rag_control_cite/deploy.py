"""Stage 7 — deploy + teardown: container, CDK, `just down`.

Container (docker compose), CDK for cloud infra. ``just down`` tears
everything down for cents. Ends in: cold start, p95 under load, $/1k,
teardown verified.
"""

from __future__ import annotations

__all__ = ["deploy", "teardown"]


def deploy() -> str:
    """Deploy the service.

    Raises:
        NotImplementedError: Stage 7 not implemented.
    """
    raise NotImplementedError("Stage 7 — deploy not implemented")


def teardown() -> None:
    """Tear down everything — cents, not data.

    Raises:
        NotImplementedError: Stage 7 not implemented.
    """
    raise NotImplementedError("Stage 7 — teardown not implemented")
