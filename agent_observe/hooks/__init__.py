"""
Hooks module for agent-observe.

Provides lifecycle hooks for tool calls, model calls, and run lifecycle events.
"""

from agent_observe.hooks.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
    HookCircuitBreaker,
)
from agent_observe.hooks.context import (
    ErrorContext,
    ModelContext,
    RunEndContext,
    RunStartContext,
    SpanEndContext,
    SpanStartContext,
    ToolContext,
)
from agent_observe.hooks.registry import HookRegistry
from agent_observe.hooks.result import HookAction, HookResult
from agent_observe.hooks.testing import (
    HookCallRecord,
    RecordingHookRegistry,
    assert_hook_blocks,
    assert_hook_modifies,
    assert_hook_proceeds,
    assert_hook_skips,
    mock_error_context,
    mock_model_context,
    mock_run_end_context,
    mock_run_start_context,
    mock_tool_context,
)

__all__ = [
    # Registry
    "HookRegistry",
    # Result
    "HookAction",
    "HookResult",
    # Circuit Breaker
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "CircuitState",
    "HookCircuitBreaker",
    # Contexts
    "ToolContext",
    "ModelContext",
    "RunStartContext",
    "RunEndContext",
    "SpanStartContext",
    "SpanEndContext",
    "ErrorContext",
    # Testing utilities
    "mock_tool_context",
    "mock_model_context",
    "mock_run_start_context",
    "mock_run_end_context",
    "mock_error_context",
    "RecordingHookRegistry",
    "HookCallRecord",
    "assert_hook_blocks",
    "assert_hook_proceeds",
    "assert_hook_skips",
    "assert_hook_modifies",
]
