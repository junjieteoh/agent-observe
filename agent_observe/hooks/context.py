"""
Hook context objects.

These dataclasses provide all the information hooks need to make decisions.
They wrap the existing RunContext and SpanContext with hook-specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_observe.context import RunContext, SpanContext
    from agent_observe.observe import Observe


@dataclass
class ToolContext:
    """
    Context provided to tool hooks (before_tool, after_tool, on_tool_error).

    Contains all information about the tool call being made.
    """

    # References to core contexts
    run: RunContext
    span: SpanContext
    observe: Observe

    # Tool information
    tool_name: str
    tool_kind: str
    tool_version: str

    # Arguments (can be modified in before_tool hooks)
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)

    # Timing
    timestamp_ms: int = 0

    def get_arg(self, name: str, default: Any = None) -> Any:
        """Get a keyword argument by name."""
        return self.kwargs.get(name, default)

    def set_arg(self, name: str, value: Any) -> None:
        """Set a keyword argument (for before hooks)."""
        self.kwargs[name] = value


@dataclass
class ModelContext:
    """
    Context provided to model hooks (before_model, after_model, on_model_error).

    Contains all information about the LLM call being made.
    """

    # References to core contexts
    run: RunContext
    span: SpanContext
    observe: Observe

    # Model information
    provider: str
    model: str

    # Extracted for convenience (also available in kwargs)
    messages: list[dict[str, Any]] | None = None
    system_prompt: str | None = None

    # Arguments (can be modified in before_model hooks)
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)

    # Timing
    timestamp_ms: int = 0

    def get_message_count(self) -> int:
        """Get number of messages in the conversation."""
        if self.messages:
            return len(self.messages)
        return 0

    def get_last_user_message(self) -> str | None:
        """Get the content of the last user message."""
        if not self.messages:
            return None
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return msg.get("content")
        return None


@dataclass
class RunStartContext:
    """
    Context provided to on_run_start hook.

    Contains information about the run that's starting.
    """

    # References
    run: RunContext
    observe: Observe

    # Timing
    timestamp_ms: int = 0


@dataclass
class RunEndContext:
    """
    Context provided to on_run_end hook.

    Contains information about the completed run including summary stats.
    """

    # References
    run: RunContext
    observe: Observe

    # Status
    status: str = "ok"  # "ok" or "error"
    error: Exception | None = None

    # Timing
    duration_ms: int = 0
    timestamp_ms: int = 0

    # Summary stats
    tool_calls: int = 0
    model_calls: int = 0
    policy_violations: int = 0

    # All spans from the run (for analysis)
    spans: list[SpanContext] = field(default_factory=list)


@dataclass
class SpanStartContext:
    """
    Context provided to on_span_start hook.
    """

    run: RunContext
    span: SpanContext
    observe: Observe
    timestamp_ms: int = 0


@dataclass
class SpanEndContext:
    """
    Context provided to on_span_end hook.

    Called just before the span is written to the sink.
    """

    run: RunContext
    span: SpanContext
    observe: Observe

    # Timing
    duration_ms: int = 0
    timestamp_ms: int = 0

    # Status
    status: str = "ok"  # "ok", "error", "blocked"
    error: Exception | None = None


@dataclass
class ErrorContext:
    """
    Context provided to error hooks (on_tool_error, on_model_error).
    """

    # References
    run: RunContext
    span: SpanContext
    observe: Observe

    # Error information
    error: Exception
    error_type: str = ""
    error_message: str = ""

    # Timing
    timestamp_ms: int = 0

    def __post_init__(self) -> None:
        """Extract error details."""
        if not self.error_type:
            self.error_type = type(self.error).__name__
        if not self.error_message:
            self.error_message = str(self.error)
