"""OpenTelemetry tracing to the cluster's existing collector (OTLP/HTTP, OTEL_EXPORTER_OTLP_ENDPOINT, e.g.
http://otel-collector.observability:4318 → Jaeger, like the web app). One span per flow step (flow.<flow>.<step>) and
per crew run (crew.<name>). No endpoint → spans are created but not exported. CrewAI's own tracing and telemetry stay
off (config.py)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

_tracer = trace.get_tracer("seo-agent")


def setup_tracing(endpoint: str, service_name: str, version: str = "dev") -> bool:
    if not endpoint:
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name, "service.version": version}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")))
    trace.set_tracer_provider(provider)
    return True


@contextmanager
def span(name: str, attrs: dict[str, Any] | None = None) -> Iterator[Any]:
    with trace.get_tracer("seo-agent").start_as_current_span(name, attributes={k: str(v) for k, v in (attrs or {}).items()}) as s:
        try:
            yield s
        except Exception as e:
            s.record_exception(e)
            s.set_status(trace.Status(trace.StatusCode.ERROR, type(e).__name__))
            raise


def emit_span(name: str, *, start_ns: int, attrs: dict[str, Any] | None = None, error: str | None = None) -> None:
    """A finished span from `start_ns` to now, as a child of the current span (e.g. the flow step's). Used for LLM
    calls (one `llm.call` span each): a streamed call lives across generator yields, where a context-managed current
    span (span() above) would be attached in one context and detached in another, so the span is emitted once the call
    is booked. Numbers stay numeric (Jaeger range queries); None values are dropped."""
    clean = {k: (v if isinstance(v, (bool, int, float, str)) else str(v)) for k, v in (attrs or {}).items() if v is not None}
    s = trace.get_tracer("seo-agent").start_span(name, start_time=start_ns, attributes=clean)
    if error:
        s.set_status(trace.Status(trace.StatusCode.ERROR, error))
    s.end()


def current_ids() -> tuple[str, str] | None:
    """(trace_id, span_id) of the current span as hex, for log lines (Loki → Jaeger); None when there is no span."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


def annotate(attrs: dict[str, Any], *, error: str | None = None) -> None:
    """Set attributes (and an error status) on the current span, e.g. a flow run's result once it is known."""
    s = trace.get_current_span()
    for k, v in attrs.items():
        if v is not None:
            s.set_attribute(k, v if isinstance(v, (bool, int, float, str)) else str(v))
    if error:
        s.set_status(trace.Status(trace.StatusCode.ERROR, error))


def inject(headers: dict[str, str]) -> dict[str, str]:
    """W3C traceparent of the current span into outgoing headers, so the web app's spans join the agent's trace."""
    from opentelemetry import propagate

    propagate.inject(headers)
    return headers
