"""
agent-observe: Framework-agnostic observability, audit, and eval for AI agent applications.

Usage:
    from agent_observe import observe, tool, model_call

    observe.install()

    @tool(name="my_tool", kind="generic")
    def my_tool(arg: str) -> str:
        return f"result: {arg}"

    with observe.run("my-agent"):
        result = my_tool("hello")
"""

from agent_observe.context import RunContext, SpanContext, SpanKind, SpanStatus
from agent_observe.decorators import model_call, tool
from agent_observe.hooks import (
    CircuitBreakerConfig,
    HookAction,
    HookRegistry,
    HookResult,
    ModelContext,
    RunEndContext,
    RunStartContext,
    ToolContext,
)
from agent_observe.observe import observe
from agent_observe.pii import PIIAction, PIIConfig, PIIHandler
from agent_observe.policy import PolicyViolationError

__all__ = [
    # Core API
    "observe",
    "tool",
    "model_call",
    # Context types
    "RunContext",
    "SpanContext",
    "SpanKind",
    "SpanStatus",
    # Hook types
    "HookAction",
    "HookResult",
    "HookRegistry",
    "ToolContext",
    "ModelContext",
    "RunStartContext",
    "RunEndContext",
    "CircuitBreakerConfig",
    # PII types
    "PIIConfig",
    "PIIHandler",
    "PIIAction",
    # Exceptions
    "PolicyViolationError",
]

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("agent-observe")
except Exception:
    __version__ = "0.1.2"  # Fallback for development
