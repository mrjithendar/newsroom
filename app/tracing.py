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
