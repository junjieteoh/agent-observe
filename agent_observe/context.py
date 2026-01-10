"""
Context management for agent-observe.

Uses contextvars for async-safe context propagation of runs and spans.
Provides the foundation for tracking nested tool calls and model invocations.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent_observe.config import CaptureMode
    from agent_observe.policy import PolicyEngine

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

    # v0.1.7: Attribution and context
    user_id: str | None = None
    session_id: str | None = None
    prompt_version: str | None = None
    model_config: dict[str, Any] | None = None
    experiment_id: str | None = None

    # v0.1.7: Run-level input/output (the Wide Event content)
    input: Any | None = field(default=None, repr=False)
    output: Any | None = field(default=None, repr=False)
    _input_set_explicitly: bool = field(default=False, repr=False)
    _output_set_explicitly: bool = field(default=False, repr=False)

    # v0.1.7: Auto-calculated prompt hash
    prompt_hash: str | None = None

    # Metrics (accumulated during run)
    tool_calls: int = 0
    model_calls: int = 0
    policy_violations: int = 0
    retry_count: int = 0

    # Custom metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal tracking
    _token: Token[RunContext] | None = field(default=None, repr=False)
    _spans: list[SpanContext] = field(default_factory=list, repr=False)
    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _tool_call_hashes: list[str] = field(default_factory=list, repr=False)
    _observe: Any | None = field(default=None, repr=False)  # Reference to Observe instance

    # Per-run config overrides (None = use global config)
    _mode_override: CaptureMode | None = field(default=None, repr=False)
    _policy_engine_override: PolicyEngine | None = field(default=None, repr=False)
    _fail_on_violation_override: bool | None = field(default=None, repr=False)
    _latency_budget_ms_override: int | None = field(default=None, repr=False)

    # Span memory management
    _max_spans_in_memory: int = field(default=10000, repr=False)
    _flushed_span_count: int = field(default=0, repr=False)

    def get_capture_mode(self) -> CaptureMode:
        """Get effective capture mode (per-run override or global config)."""
        if self._mode_override is not None:
            return self._mode_override
        if self._observe is not None:
            return self._observe.config.mode
        # Fallback import to avoid circular dependency at runtime
        from agent_observe.config import CaptureMode as CM

        return CM.METADATA_ONLY

    def get_policy_engine(self) -> PolicyEngine | None:
        """Get effective policy engine (per-run override or global)."""
        if self._policy_engine_override is not None:
            return self._policy_engine_override
        if self._observe is not None:
            return self._observe.policy_engine
        return None

    def get_fail_on_violation(self) -> bool:
        """Get effective fail_on_violation setting."""
        if self._fail_on_violation_override is not None:
            return self._fail_on_violation_override
        if self._observe is not None:
            return self._observe.config.fail_on_violation
        return False

    def get_latency_budget_ms(self) -> int:
        """Get effective latency budget in milliseconds."""
        if self._latency_budget_ms_override is not None:
            return self._latency_budget_ms_override
        if self._observe is not None:
            return self._observe.config.latency_budget_ms
        return 20000

    def set_input(self, input_data: Any) -> None:
        """
        Set the original user request/input for this run.

        Args:
            input_data: The input to the agent (user message, request, etc.)
        """
        self.input = input_data
        self._input_set_explicitly = True

    def set_output(self, output_data: Any) -> None:
        """
        Set the final agent output for this run.

        Args:
            output_data: The output from the agent (response, result, etc.)
        """
        self.output = output_data
        self._output_set_explicitly = True

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add custom metadata to this run.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata[key] = value

    def _infer_input_output(self) -> None:
        """
        Auto-infer input/output from spans if not explicitly set.

        Called at run end to ensure we always have input/output for traces.
        """
        if not self._spans:
            return

        # Infer input from first span if not explicitly set
        if not self._input_set_explicitly and self.input is None:
            first_span = self._spans[0]
            # Try to extract input from span attrs
            if "input" in first_span.attrs:
                self.input = first_span.attrs.get("input")
                logger.debug(
                    f"Run '{self.name}' input auto-inferred from span '{first_span.name}'"
                )
            elif "args" in first_span.attrs:
                self.input = first_span.attrs.get("args")
                logger.debug(
                    f"Run '{self.name}' input auto-inferred from span '{first_span.name}' args"
                )

        # Infer output from last successful span if not explicitly set
        if not self._output_set_explicitly and self.output is None:
            for span in reversed(self._spans):
                if span.status == SpanStatus.OK:
                    if "output" in span.attrs:
                        self.output = span.attrs.get("output")
                        logger.debug(
                            f"Run '{self.name}' output auto-inferred from span '{span.name}'"
                        )
                        break
                    elif "result" in span.attrs:
                        self.output = span.attrs.get("result")
                        logger.debug(
                            f"Run '{self.name}' output auto-inferred from span '{span.name}' result"
                        )
                        break

    def _infer_prompt_hash(self) -> None:
        """
        Auto-calculate prompt hash from first model call's system prompt.

        Called at run end if prompt_version not explicitly set.
        """
        if self.prompt_hash is not None or self.prompt_version is not None:
            return  # Already set

        for span in self._spans:
            if span.kind == SpanKind.MODEL:
                # Look for system prompt in LLM context
                llm_context = span.attrs.get("llm_context")
                if llm_context and isinstance(llm_context, dict):
                    system_prompt = llm_context.get("system_prompt")
                    if system_prompt:
                        from agent_observe.hashing import hash_json
                        self.prompt_hash = hash_json(system_prompt)[:16]
                        logger.debug(
                            f"Run '{self.name}' prompt_hash auto-calculated: {self.prompt_hash}"
                        )
                        return

                # Try extracting from input messages
                input_data = span.attrs.get("input")
                if input_data:
                    try:
                        import json
                        if isinstance(input_data, str):
                            input_data = json.loads(input_data)
                        if isinstance(input_data, dict):
                            # Check kwargs.messages or args
                            messages = None
                            if "kwargs" in input_data:
                                messages = input_data["kwargs"].get("messages")
                            if not messages and "args" in input_data and input_data["args"]:
                                first_arg = input_data["args"][0]
                                if isinstance(first_arg, list):
                                    messages = first_arg

                            if messages:
                                # Find system message
                                for msg in messages:
                                    if isinstance(msg, dict) and msg.get("role") == "system":
                                        from agent_observe.hashing import hash_json
                                        self.prompt_hash = hash_json(msg.get("content", ""))[:16]
                                        logger.debug(
                                            f"Run '{self.name}' prompt_hash auto-calculated from messages"
                                        )
                                        return
                    except (json.JSONDecodeError, TypeError, KeyError):
                        pass

    def add_span(self, span: SpanContext) -> None:
        """Record a span in this run."""
        self._spans.append(span)
        if span.kind == SpanKind.TOOL:
            self.tool_calls += 1
        elif span.kind == SpanKind.MODEL:
            self.model_calls += 1

        # Flush if over memory limit
        if len(self._spans) >= self._max_spans_in_memory:
            self._flush_spans()

    def _flush_spans(self) -> None:
        """Flush accumulated spans to sink and clear memory."""
        if self._observe is None or not self._spans:
            return

        spans_to_flush = len(self._spans)
        try:
            for span in self._spans:
                self._observe.sink.write_span(span.to_dict())
            self._flushed_span_count += spans_to_flush
            logger.debug(
                f"Flushed {spans_to_flush} spans to sink "
                f"(total flushed: {self._flushed_span_count})"
            )
        except Exception as e:
            logger.warning(f"Failed to flush spans: {e}")
        finally:
            # IMPORTANT: Clear spans even on failure to prevent unbounded memory growth
            # Spans are best-effort; losing them is better than OOM
            self._spans.clear()

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
        import json

        # Serialize input/output to JSON strings for storage
        input_json = None
        input_text = None
        if self.input is not None:
            try:
                if isinstance(self.input, str):
                    input_json = json.dumps(self.input)
                    input_text = self.input
                else:
                    input_json = json.dumps(self.input, default=str)
                    # Extract text for FTS
                    input_text = self._extract_text(self.input)
            except (TypeError, ValueError):
                input_json = str(self.input)
                input_text = str(self.input)

        output_json = None
        output_text = None
        if self.output is not None:
            try:
                if isinstance(self.output, str):
                    output_json = json.dumps(self.output)
                    output_text = self.output
                else:
                    output_json = json.dumps(self.output, default=str)
                    output_text = self._extract_text(self.output)
            except (TypeError, ValueError):
                output_json = str(self.output)
                output_text = str(self.output)

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
            # v0.1.7: Attribution
            "user_id": self.user_id,
            "session_id": self.session_id,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "model_config": self.model_config,
            "experiment_id": self.experiment_id,
            # v0.1.7: Content
            "input_json": input_json,
            "input_text": input_text,
            "output_json": output_json,
            "output_text": output_text,
            # Metrics
            "tool_calls": self.tool_calls,
            "model_calls": self.model_calls,
            "policy_violations": self.policy_violations,
            "retry_count": self.retry_count,
            # Custom metadata
            "metadata": self.metadata if self.metadata else None,
        }

    @staticmethod
    def _extract_text(data: Any) -> str:
        """Extract searchable text from data for FTS."""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            # Extract common text fields
            parts = []
            for key in ("content", "text", "message", "response"):
                if key in data:
                    parts.append(str(data[key]))
            if parts:
                return " ".join(parts)
            # Fallback to JSON
            import json
            return json.dumps(data, default=str)
        if isinstance(data, list):
            # Handle message lists
            parts = []
            for item in data:
                if isinstance(item, dict) and "content" in item:
                    parts.append(str(item["content"]))
            if parts:
                return " ".join(parts)
        return str(data)


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
