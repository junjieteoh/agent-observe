"""
Context management for agent-observe.

Uses contextvars for async-safe context propagation of runs and spans.
Provides the foundation for tracking nested tool calls and model invocations.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Context variables for async-safe run/span tracking
_current_run: ContextVar[RunContext | None] = ContextVar("current_run", default=None)
_current_span: ContextVar[SpanContext | None] = ContextVar("current_span", default=None)


class SpanKind(Enum):
    """Types of spans."""

    ROOT = "root"
    TOOL = "tool"
    MODEL = "model"
    INTERNAL = "internal"


class SpanStatus(Enum):
    """Span completion status."""

    OK = "ok"
    ERROR = "error"
    BLOCKED = "blocked"


class RunStatus(Enum):
    """Run completion status."""

    OK = "ok"
    ERROR = "error"
    BLOCKED = "blocked"


def generate_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def generate_trace_id() -> str:
    """Generate a 32-character hex trace ID (OpenTelemetry compatible)."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a 16-character hex span ID (OpenTelemetry compatible)."""
    return uuid.uuid4().hex[:16]


def now_ms() -> int:
    """Return current timestamp in milliseconds since epoch."""
    return int(time.time() * 1000)


@dataclass
class SpanContext:
    """
    Context for a single span (tool call, model call, etc.).

    Spans form a tree structure via parent_span_id.
    """

    span_id: str
    run_id: str
    parent_span_id: str | None
    kind: SpanKind
    name: str
    ts_start: int
    ts_end: int | None = None
    status: SpanStatus = SpanStatus.OK
    attrs: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    # Internal: context token for cleanup
    _token: Token[SpanContext] | None = field(default=None, repr=False)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attrs[key] = value

    def set_status(self, status: SpanStatus, error_message: str | None = None) -> None:
        """Set span status."""
        self.status = status
        if error_message:
            self.error_message = error_message

    def end(self) -> None:
        """End the span and record end timestamp."""
        self.ts_end = now_ms()

    @property
    def duration_ms(self) -> int | None:
        """Get span duration in milliseconds."""
        if self.ts_end is None:
            return None
        return self.ts_end - self.ts_start

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for storage."""
        return {
            "span_id": self.span_id,
            "run_id": self.run_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "name": self.name,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "status": self.status.value,
            "attrs": self.attrs,
            "error_message": self.error_message,
        }


@dataclass
class RunContext:
    """
    Context for an entire agent run.

    Tracks all spans, events, and metrics for a single run.
    """

    run_id: str
    trace_id: str
    name: str
    ts_start: int
    task: dict[str, Any] | None = None
    agent_version: str = ""
    project: str = ""
    env: str = ""
    ts_end: int | None = None
    status: RunStatus = RunStatus.OK

    # Metrics (accumulated during run)
    tool_calls: int = 0
    model_calls: int = 0
    policy_violations: int = 0
    retry_count: int = 0

    # Internal tracking
    _token: Token[RunContext] | None = field(default=None, repr=False)
    _spans: list[SpanContext] = field(default_factory=list, repr=False)
    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _tool_call_hashes: list[str] = field(default_factory=list, repr=False)
    _observe: Any | None = field(default=None, repr=False)  # Reference to Observe instance

    def add_span(self, span: SpanContext) -> None:
        """Record a span in this run."""
        self._spans.append(span)
        if span.kind == SpanKind.TOOL:
            self.tool_calls += 1
        elif span.kind == SpanKind.MODEL:
            self.model_calls += 1

    def add_event(self, event: dict[str, Any]) -> None:
        """Record an event in this run."""
        self._events.append(event)

    def record_tool_call_hash(self, tool_hash: str) -> None:
        """Record tool call hash for loop detection."""
        self._tool_call_hashes.append(tool_hash)

    def record_policy_violation(self) -> None:
        """Increment policy violation count."""
        self.policy_violations += 1

    def record_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1

    def end(self, status: RunStatus | None = None) -> None:
        """End the run and record end timestamp."""
        self.ts_end = now_ms()
        if status:
            self.status = status

    @property
    def duration_ms(self) -> int | None:
        """Get run duration in milliseconds."""
        if self.ts_end is None:
            return None
        return self.ts_end - self.ts_start

    @property
    def spans(self) -> list[SpanContext]:
        """Get all spans in this run."""
        return self._spans.copy()

    @property
    def events(self) -> list[dict[str, Any]]:
        """Get all events in this run."""
        return self._events.copy()

    @property
    def tool_call_hashes(self) -> list[str]:
        """Get all tool call hashes for loop detection."""
        return self._tool_call_hashes.copy()

    def to_dict(self) -> dict[str, Any]:
        """Convert run to dictionary for storage."""
        return {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "task": self.task,
            "agent_version": self.agent_version,
            "project": self.project,
            "env": self.env,
            "status": self.status.value,
            "tool_calls": self.tool_calls,
            "model_calls": self.model_calls,
            "policy_violations": self.policy_violations,
            "retry_count": self.retry_count,
        }


def get_current_run() -> RunContext | None:
    """Get the current run context, if any."""
    return _current_run.get()


def get_current_span() -> SpanContext | None:
    """Get the current span context, if any."""
    return _current_span.get()


def set_current_run(run: RunContext | None) -> Token[RunContext | None] | None:
    """Set the current run context. Returns token for restoration."""
    return _current_run.set(run)


def set_current_span(span: SpanContext | None) -> Token[SpanContext | None] | None:
    """Set the current span context. Returns token for restoration."""
    return _current_span.set(span)


def reset_current_run(token: Token[RunContext | None]) -> None:
    """Reset current run context to previous value."""
    _current_run.reset(token)


def reset_current_span(token: Token[SpanContext | None]) -> None:
    """Reset current span context to previous value."""
    _current_span.reset(token)


def create_span(
    name: str,
    kind: SpanKind,
    attrs: dict[str, Any] | None = None,
) -> SpanContext:
    """
    Create a new span in the current run context.

    Args:
        name: Span name.
        kind: Type of span (tool, model, internal).
        attrs: Optional initial attributes.

    Returns:
        New SpanContext.

    Raises:
        RuntimeError: If no run context is active.
    """
    run = get_current_run()
    if run is None:
        raise RuntimeError("Cannot create span outside of a run context")

    parent = get_current_span()
    parent_span_id = parent.span_id if parent else None

    span = SpanContext(
        span_id=generate_span_id(),
        run_id=run.run_id,
        parent_span_id=parent_span_id,
        kind=kind,
        name=name,
        ts_start=now_ms(),
        attrs=attrs or {},
    )

    return span


class SpanContextManager:
    """Context manager for spans with automatic context propagation."""

    def __init__(
        self,
        name: str,
        kind: SpanKind,
        attrs: dict[str, Any] | None = None,
    ):
        self.name = name
        self.kind = kind
        self.attrs = attrs
        self.span: SpanContext | None = None
        self._token: Token[SpanContext | None] | None = None

    def __enter__(self) -> SpanContext:
        self.span = create_span(self.name, self.kind, self.attrs)
        self._token = set_current_span(self.span)
        return self.span

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self.span is not None:
            self.span.end()

            if exc_type is not None:
                self.span.set_status(SpanStatus.ERROR, str(exc_val))

            # Record span in run
            run = get_current_run()
            if run is not None:
                run.add_span(self.span)

        # Restore previous span context
        if self._token is not None:
            reset_current_span(self._token)


def span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attrs: dict[str, Any] | None = None,
) -> SpanContextManager:
    """
    Create a span context manager.

    Usage:
        with span("my_operation", SpanKind.INTERNAL) as s:
            s.set_attribute("key", "value")
            # do work
    """
    return SpanContextManager(name, kind, attrs)
