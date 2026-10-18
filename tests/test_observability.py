"""Observability: exact per-request accounting, bounded store, thread safety."""

from __future__ import annotations

import threading

import pytest

from rag_control_cite.observability import TraceRecorder, stage_span


def test_exact_per_request_accounting() -> None:
    """100 requests with fixed usage report exactly 100 x usage."""
    recorder = TraceRecorder(maxlen=10_000)
    for i in range(100):
        qid = f"q{i}"
        with recorder.trace(qid), stage_span("generate", recorder, qid) as span:
            span.add_tokens(100, 200, "openai/gpt-4o-mini")
    prompt, completion = recorder.stage_tokens("generate")
    assert prompt == 100 * 100
    assert completion == 100 * 200
    assert recorder.n_queries == 100
    # Cost comes from the price table, not a constant: 100*100 prompt @0.15/M
    # + 100*200 completion @0.60/M.
    expected = (10_000 * 0.15 + 20_000 * 0.60) / 1_000_000
    assert recorder.stage_cost("generate") == pytest.approx(expected)


def test_bounded_memory() -> None:
    recorder = TraceRecorder(maxlen=100)
    for i in range(10_000):
        with recorder.trace(f"q{i}"), stage_span("retrieve", recorder, f"q{i}"):
            pass
    assert recorder.n_retained == 100  # bounded
    assert recorder.n_queries == 10_000  # lifetime count


def test_thread_safety_no_loss_or_double_count() -> None:
    recorder = TraceRecorder(maxlen=10_000)

    def worker(base: int) -> None:
        for i in range(200):
            qid = f"{base}-{i}"
            with recorder.trace(qid), stage_span("generate", recorder, qid) as span:
                span.add_tokens(10, 20, "openai/gpt-4o-mini")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert recorder.n_queries == 8 * 200
    prompt, completion = recorder.stage_tokens("generate")
    assert prompt == 8 * 200 * 10
    assert completion == 8 * 200 * 20


def test_outcome_aware_fully_traced() -> None:
    recorder = TraceRecorder()

    with recorder.trace("q1") as t:
        with (
            stage_span("retrieve", recorder, "q1"),
            stage_span("acl", recorder, "q1"),
            stage_span("generate", recorder, "q1"),
        ):
            pass
        t.outcome = 200

    with recorder.trace("q2") as t:
        with stage_span("retrieve", recorder, "q2"), stage_span("acl", recorder, "q2"):
            pass
        t.outcome = 200  # missing generate → not fully traced

    with recorder.trace("q3") as t:
        t.outcome = 403  # rejected at ingress → no stages needed

    with recorder.trace("q4") as t:
        with stage_span("retrieve", recorder, "q4"), stage_span("acl", recorder, "q4"):
            pass
        t.outcome = 404

    with recorder.trace("q5") as t:
        with stage_span("retrieve", recorder, "q5"):
            pass
        t.outcome = 404  # missing acl → not fully traced

    assert recorder.n_queries == 5
    assert recorder.n_fully_traced == 3  # q1, q3, q4


def test_otel_spans_are_created() -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import rag_control_cite.observability as obs

    exporter = InMemorySpanExporter()
    obs.configure_otel(exporter)

    recorder = TraceRecorder()
    with stage_span("retrieve", recorder, "q"):
        pass
    with stage_span("acl", recorder, "q"):
        pass
    with stage_span("generate", recorder, "q"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    assert {s.name for s in spans} == {"retrieve", "acl", "generate"}

    # Reset the module tracer so other tests are unaffected.
    obs._tracer = None
