"""Stage 6 — observability: one OTEL span per stage.

Every stage — embedding, retrieval, rerank, ACL, generation, judge —
emits a span through a real ``TracerProvider`` (``configure_otel``) and
records into a bounded, lock-protected ``TraceRecorder``:
  - % queries fully traced (outcome-aware: a 403 needs no generate stage)
  - per-stage latency (p50, p95, mean)
  - per-stage cost from the one price table (``pricing.py``)

The recorder is the source of truth for the served numbers. Tokens come
from per-call usage on each result, never from a cumulative counter — the
counter inflated the reported prompt tokens by 163x.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SpanContext",
    "TraceRecorder",
    "configure_otel",
    "stage_span",
]


def _required_stages(outcome: int) -> frozenset[str]:
    """Stages a fully-traced query must have, given its outcome.

    200 answers retrieve → acl → generate. A 403 is rejected at ingress
    (no stages). A 404 ran retrieval + acl and found nothing to generate.
    """
    if outcome == 200:
        return frozenset({"retrieve", "acl", "generate"})
    if outcome == 404:
        return frozenset({"retrieve", "acl"})
    return frozenset()


@dataclass
class SpanContext:
    """Returned by ``stage_span`` — lets the caller add token counts."""

    stage_name: str
    start_time: float
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    def add_tokens(self, prompt: int, completion: int, model: str = "") -> None:
        """Record this stage's API token usage (per call, not cumulative)."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        if model:
            self.model = model


@dataclass
class StageRecord:
    """One stage execution within one query trace."""

    stage_name: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


@dataclass
class QueryTrace:
    """All stages for one query execution, plus its HTTP outcome."""

    query_id: str
    stages: list[StageRecord] = field(default_factory=list)
    outcome: int = 0

    @property
    def stage_names(self) -> set[str]:
        return {s.stage_name for s in self.stages}

    @property
    def fully_traced(self) -> bool:
        required = (
            _required_stages(self.outcome) if self.outcome else frozenset({"retrieve", "generate"})
        )
        return required <= self.stage_names

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.stages)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.stages)

    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.stages)


class TraceRecorder:
    """Bounded, thread-safe accumulator of per-stage timings and costs.

    The store is a ``deque(maxlen=...)``: memory is bounded after any
    number of requests. All mutation and reads hold a lock, so concurrent
    requests neither lose nor double-count.
    """

    def __init__(self, maxlen: int = 1000) -> None:
        self._maxlen = maxlen
        self._traces: deque[QueryTrace] = deque(maxlen=maxlen)
        self._by_id: dict[str, QueryTrace] = {}
        self._lock = threading.Lock()
        self._n_total = 0
        self._n_fully_traced = 0

    @contextmanager
    def trace(self, query_id: str) -> Iterator[QueryTrace]:
        """Start a query trace; evict the oldest once the bound is hit."""
        qt = QueryTrace(query_id=query_id)
        with self._lock:
            if len(self._traces) == self._maxlen:
                evicted = self._traces[0]
                self._by_id.pop(evicted.query_id, None)
            self._traces.append(qt)
            self._by_id[query_id] = qt
        try:
            yield qt
        finally:
            with self._lock:
                self._n_total += 1
                if qt.fully_traced:
                    self._n_fully_traced += 1

    def set_outcome(self, query_id: str, outcome: int) -> None:
        """Record the HTTP outcome so 'fully traced' can depend on it."""
        with self._lock:
            qt = self._by_id.get(query_id)
            if qt is not None:
                qt.outcome = outcome

    def record_stage(
        self,
        query_id: str,
        stage_name: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "",
    ) -> None:
        """Record a completed stage."""
        with self._lock:
            qt = self._by_id.get(query_id)
            if qt is None:
                # No trace() was opened — still register it so later stages
                # for the same query land on one trace.
                qt = QueryTrace(query_id=query_id)
                if len(self._traces) == self._maxlen:
                    evicted = self._traces[0]
                    self._by_id.pop(evicted.query_id, None)
                self._traces.append(qt)
                self._by_id[query_id] = qt
            qt.stages.append(
                StageRecord(
                    stage_name=stage_name,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model=model,
                )
            )

    @property
    def traces(self) -> list[QueryTrace]:
        with self._lock:
            return list(self._traces)

    @property
    def n_queries(self) -> int:
        """Total queries recorded (lifetime), not the retained window."""
        with self._lock:
            return self._n_total

    @property
    def n_retained(self) -> int:
        with self._lock:
            return len(self._traces)

    @property
    def n_fully_traced(self) -> int:
        with self._lock:
            return self._n_fully_traced

    @property
    def pct_fully_traced(self) -> float:
        with self._lock:
            if self._n_total == 0:
                return 0.0
            return self._n_fully_traced / self._n_total

    def stage_latencies(self, stage_name: str) -> list[float]:
        return [s.latency_ms for t in self.traces for s in t.stages if s.stage_name == stage_name]

    def stage_tokens(self, stage_name: str) -> tuple[int, int]:
        records = [s for t in self.traces for s in t.stages if s.stage_name == stage_name]
        return (
            sum(s.prompt_tokens for s in records),
            sum(s.completion_tokens for s in records),
        )

    def stage_cost(self, stage_name: str) -> float:
        """Sum this stage's cost from the one price table."""
        from rag_control_cite.pricing import UnknownModelError, cost_usd

        total = 0.0
        for t in self.traces:
            for s in t.stages:
                if s.stage_name != stage_name or not s.model:
                    continue
                try:
                    total += cost_usd(s.prompt_tokens, s.completion_tokens, s.model)
                except UnknownModelError:
                    continue  # unpriced model — no guessed cost
        return total

    @property
    def stage_names(self) -> set[str]:
        return {s.stage_name for t in self.traces for s in t.stages}

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-stage latency and cost breakdown."""
        from rag_control_cite.measure import _percentile

        result: dict[str, dict[str, float]] = {}
        for stage in sorted(self.stage_names):
            lats = self.stage_latencies(stage)
            p, c = self.stage_tokens(stage)
            result[stage] = {
                "n": len(lats),
                "p50_ms": round(_percentile(lats, 50), 1) if lats else 0.0,
                "p95_ms": round(_percentile(lats, 95), 1) if lats else 0.0,
                "mean_ms": round(sum(lats) / len(lats), 1) if lats else 0.0,
                "prompt_tokens": p,
                "completion_tokens": c,
                "est_cost": round(self.stage_cost(stage), 6),
            }
        return result

    def as_dict(self) -> dict[str, Any]:
        """Full observability summary for RESULTS.md."""
        return {
            "n_queries": self.n_queries,
            "n_retained": self.n_retained,
            "n_fully_traced": self.n_fully_traced,
            "pct_fully_traced": round(self.pct_fully_traced, 4),
            "stages": self.summary(),
        }


# --------------------------------------------------------------------------- #
# OTEL — a real tracer provider, wired by configure_otel
# --------------------------------------------------------------------------- #

_tracer: Any = None


def configure_otel(exporter: Any = None, endpoint: str = "") -> Any:
    """Configure a real TracerProvider and return its tracer.

    Args:
        exporter: a span exporter (e.g. ``InMemorySpanExporter`` in tests,
            or an OTLP exporter in production). None = no exporter; spans
            are created and dropped (still real spans, just unsampled).
        endpoint: OTLP endpoint; if set and no exporter is given, an OTLP
            span exporter is built lazily (requires the otlp exporter
            dependency).

    Returns:
        The tracer, also stored as the module-level tracer used by
        ``stage_span``.
    """
    global _tracer
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    provider = TracerProvider()
    if exporter is None and endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(endpoint=endpoint)
    if exporter is not None:
        processor_cls = (
            SimpleSpanProcessor
            if exporter.__class__.__name__ == "InMemorySpanExporter"
            else BatchSpanProcessor
        )
        provider.add_span_processor(processor_cls(exporter))
    _tracer = provider.get_tracer("rag-control-cite")
    return _tracer


def get_tracer() -> Any:
    """Return the configured tracer, or None (spans are best-effort)."""
    return _tracer


@contextmanager
def stage_span(
    stage_name: str,
    recorder: TraceRecorder | None = None,
    query_id: str = "",
) -> Iterator[SpanContext]:
    """Time a pipeline stage, emit an OTEL span, and record metrics."""
    t0 = time.perf_counter()
    ctx = SpanContext(stage_name=stage_name, start_time=t0)

    span_cm = _tracer.start_as_current_span(stage_name) if _tracer is not None else None
    if span_cm is not None:
        span_cm.__enter__()

    try:
        yield ctx
    finally:
        elapsed = (time.perf_counter() - t0) * 1000
        ctx.latency_ms = elapsed

        if span_cm is not None:
            try:
                span_cm.__exit__(None, None, None)
            except (AttributeError, TypeError, RuntimeError) as exc:
                # OTEL span close is best-effort: never fail a request for it.
                logging.getLogger(__name__).debug("otel span close failed: %s", exc)

        if recorder is not None:
            recorder.record_stage(
                query_id=query_id,
                stage_name=stage_name,
                latency_ms=elapsed,
                prompt_tokens=ctx.prompt_tokens,
                completion_tokens=ctx.completion_tokens,
                model=ctx.model,
            )
