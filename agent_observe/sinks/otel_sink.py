"""
OTLP sink for agent-observe.

Exports traces, spans, and events to an OpenTelemetry-compatible collector
(Jaeger, Honeycomb, Datadog, Grafana Tempo, etc.).
"""

from __future__ import annotations

import atexit
import logging
from datetime import datetime, timezone
from typing import Any

from agent_observe.sinks.base import Sink

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_otel_imported = False
_tracer = None
_tracer_provider = None


def _import_otel() -> bool:
    """Lazily import OpenTelemetry dependencies."""
    global _otel_imported
    if _otel_imported:
        return True

    try:
        global trace, SpanKind, StatusCode, Resource, TracerProvider
        global BatchSpanProcessor, OTLPSpanExporter

        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace import SpanKind, StatusCode

        _otel_imported = True
        return True
    except ImportError as e:
        logger.error(
            f"OpenTelemetry dependencies not installed: {e}. "
            "Install with: pip install 'agent-observe[otlp]'"
        )
        return False


def _ms_to_ns(ms: int | None) -> int | None:
    """Convert milliseconds to nanoseconds."""
    if ms is None:
        return None
    return ms * 1_000_000


def _ms_to_datetime(ms: int | None) -> datetime | None:
    """Convert milliseconds timestamp to datetime."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class OTLPSink(Sink):
    """
    OpenTelemetry Protocol (OTLP) sink.

    Exports agent-observe traces to any OTLP-compatible backend:
    - Jaeger
    - Honeycomb
    - Datadog
    - Grafana Tempo
    - AWS X-Ray (via OTEL collector)
    - And many more...

    Usage:
        # Set environment variable
        export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

        # Or configure in code
        observe.install(sink_type="otlp")
    """

    def __init__(
        self,
        endpoint: str,
        service_name: str = "agent-observe",
        async_writes: bool = True,
        headers: dict[str, str] | None = None,
        insecure: bool = True,
    ):
        """
        Initialize OTLP sink.

        Args:
            endpoint: OTLP gRPC endpoint URL (e.g., "http://localhost:4317").
            service_name: Service name for traces.
            async_writes: If True, writes are queued and flushed in background.
            headers: Optional headers for authentication (e.g., API keys).
            insecure: If True, use insecure connection (no TLS).
        """
        super().__init__(async_writes=async_writes)
        self.endpoint = endpoint
        self.service_name = service_name
        self.headers = headers or {}
        self.insecure = insecure
        self._tracer = None
        self._provider = None

    def _do_initialize(self) -> None:
        """Initialize OpenTelemetry tracer and exporter."""
        if not _import_otel():
            logger.warning("OTLP sink disabled due to missing dependencies")
            return

        from agent_observe import __version__

        # Create resource with service info
        resource = Resource.create(
            {
                "service.name": self.service_name,
                "service.version": __version__,
                "telemetry.sdk.name": "agent-observe",
            }
        )

        # Create tracer provider
        self._provider = TracerProvider(resource=resource)

        # Create OTLP exporter
        exporter = OTLPSpanExporter(
            endpoint=self.endpoint,
            headers=self.headers if self.headers else None,
            insecure=self.insecure,
        )

        # Add batch processor for efficient export
        processor = BatchSpanProcessor(exporter)
        self._provider.add_span_processor(processor)

        # Set as global tracer provider
        trace.set_tracer_provider(self._provider)

        # Get tracer
        self._tracer = trace.get_tracer("agent-observe", __version__)

        # Register cleanup
        atexit.register(self._shutdown_tracer)

        logger.info(f"OTLP sink initialized, exporting to {self.endpoint}")

    def _shutdown_tracer(self) -> None:
        """Shutdown the tracer provider."""
        if self._provider:
            try:
                self._provider.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down OTLP tracer: {e}")

    def _do_close(self) -> None:
        """Close the sink and shutdown tracer."""
        self._shutdown_tracer()

    def _map_span_kind(self, kind: str) -> Any:
        """Map agent-observe span kind to OTel SpanKind."""
        if not _otel_imported:
            return None

        mapping = {
            "tool": SpanKind.CLIENT,
            "model": SpanKind.CLIENT,
            "root": SpanKind.SERVER,
            "internal": SpanKind.INTERNAL,
        }
        return mapping.get(kind, SpanKind.INTERNAL)

    def _map_status(self, status: str) -> Any:
        """Map agent-observe status to OTel StatusCode."""
        if not _otel_imported:
            return None

        mapping = {
            "ok": StatusCode.OK,
            "error": StatusCode.ERROR,
            "blocked": StatusCode.ERROR,
        }
        return mapping.get(status, StatusCode.UNSET)

    def _do_write_runs(self, runs: list[dict[str, Any]]) -> None:
        """Write runs as root spans to OTLP."""
        if not self._tracer:
            return

        for run in runs:
            try:
                self._export_run(run)
            except Exception as e:
                logger.error(f"Error exporting run to OTLP: {e}")

    def _export_run(self, run: dict[str, Any]) -> None:
        """Export a single run as a trace with root span."""
        if not self._tracer:
            return

        # Create root span for the run
        with self._tracer.start_as_current_span(
            name=f"run:{run.get('name', 'unknown')}",
            kind=SpanKind.SERVER,
        ) as span:
            # Set span attributes
            span.set_attribute("agent_observe.run_id", run.get("run_id", ""))
            span.set_attribute("agent_observe.trace_id", run.get("trace_id", ""))
            span.set_attribute("agent_observe.name", run.get("name", ""))
            span.set_attribute("agent_observe.project", run.get("project", ""))
            span.set_attribute("agent_observe.env", run.get("env", ""))
            span.set_attribute("agent_observe.agent_version", run.get("agent_version", ""))

            # Metrics
            span.set_attribute("agent_observe.tool_calls", run.get("tool_calls", 0))
            span.set_attribute("agent_observe.model_calls", run.get("model_calls", 0))
            span.set_attribute("agent_observe.policy_violations", run.get("policy_violations", 0))
            span.set_attribute("agent_observe.risk_score", run.get("risk_score", 0))
            span.set_attribute("agent_observe.latency_ms", run.get("latency_ms", 0))

            # Eval tags
            eval_tags = run.get("eval_tags", [])
            if eval_tags:
                span.set_attribute("agent_observe.eval_tags", eval_tags)

            # Task (as JSON string if dict)
            task = run.get("task")
            if task:
                import json
                span.set_attribute("agent_observe.task", json.dumps(task, default=str))

            # Set status
            status = run.get("status", "ok")
            span.set_status(self._map_status(status))

    def _do_write_spans(self, spans: list[dict[str, Any]]) -> None:
        """Write spans to OTLP."""
        if not self._tracer:
            return

        for span_data in spans:
            try:
                self._export_span(span_data)
            except Exception as e:
                logger.error(f"Error exporting span to OTLP: {e}")

    def _export_span(self, span_data: dict[str, Any]) -> None:
        """Export a single span."""
        if not self._tracer:
            return

        kind = span_data.get("kind", "internal")
        name = span_data.get("name", "unknown")

        with self._tracer.start_as_current_span(
            name=f"{kind}:{name}",
            kind=self._map_span_kind(kind),
        ) as span:
            # Core attributes
            span.set_attribute("agent_observe.span_id", span_data.get("span_id", ""))
            span.set_attribute("agent_observe.run_id", span_data.get("run_id", ""))
            span.set_attribute("agent_observe.kind", kind)
            span.set_attribute("agent_observe.name", name)

            # Parent span (for reference, OTel manages its own hierarchy)
            parent_id = span_data.get("parent_span_id")
            if parent_id:
                span.set_attribute("agent_observe.parent_span_id", parent_id)

            # Duration
            ts_start = span_data.get("ts_start")
            ts_end = span_data.get("ts_end")
            if ts_start and ts_end:
                span.set_attribute("agent_observe.duration_ms", ts_end - ts_start)

            # Custom attributes from span
            attrs = span_data.get("attrs", {})
            for key, value in attrs.items():
                # Flatten nested values
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"agent_observe.{key}", value)
                elif isinstance(value, list):
                    span.set_attribute(f"agent_observe.{key}", str(value))

            # Error message
            error_msg = span_data.get("error_message")
            if error_msg:
                span.set_attribute("agent_observe.error_message", error_msg)

            # Set status
            status = span_data.get("status", "ok")
            span.set_status(self._map_status(status))

    def _do_write_events(self, events: list[dict[str, Any]]) -> None:
        """Write events to OTLP as span events."""
        if not self._tracer:
            return

        # Events are exported as a single span containing all events
        # This is because OTel events must be attached to spans
        for event in events:
            try:
                self._export_event(event)
            except Exception as e:
                logger.error(f"Error exporting event to OTLP: {e}")

    def _export_event(self, event: dict[str, Any]) -> None:
        """Export a single event as a span with event attached."""
        if not self._tracer:
            return

        event_type = event.get("type", "unknown")

        with self._tracer.start_as_current_span(
            name=f"event:{event_type}",
            kind=SpanKind.INTERNAL,
        ) as span:
            # Add event to span
            payload = event.get("payload", {})

            # Flatten payload for attributes
            event_attrs = {
                "event.id": event.get("event_id", ""),
                "event.run_id": event.get("run_id", ""),
                "event.type": event_type,
            }

            for key, value in payload.items():
                if isinstance(value, (str, int, float, bool)):
                    event_attrs[f"event.payload.{key}"] = value

            span.add_event(event_type, attributes=event_attrs)

            # Also set as span attributes for searchability
            span.set_attribute("agent_observe.event_id", event.get("event_id", ""))
            span.set_attribute("agent_observe.run_id", event.get("run_id", ""))
            span.set_attribute("agent_observe.event_type", event_type)
