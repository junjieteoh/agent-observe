"""
Testing utilities for agent-observe hooks.

Provides mock contexts and helpers for testing hooks in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from agent_observe.hooks.context import (
    ErrorContext,
    ModelContext,
    RunEndContext,
    RunStartContext,
    SpanEndContext,
    SpanStartContext,
    ToolContext,
)
from agent_observe.hooks.result import HookAction, HookResult


def mock_tool_context(
    *,
    tool_name: str = "test_tool",
    tool_kind: str = "generic",
    tool_version: str = "1",
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    run_id: str = "test-run-id",
    span_id: str = "test-span-id",
    **extra: Any,
) -> ToolContext:
    """
    Create a mock ToolContext for testing before_tool, after_tool, on_tool_error hooks.

    Args:
        tool_name: Name of the tool.
        tool_kind: Kind of tool (e.g., "db", "http", "generic").
        tool_version: Version of the tool.
        args: Positional arguments to the tool.
        kwargs: Keyword arguments to the tool.
        run_id: Mock run ID.
        span_id: Mock span ID.
        **extra: Additional attributes to set.

    Returns:
        A ToolContext with mocked run, span, and observe objects.

    Example:
        ctx = mock_tool_context(tool_name="query_db", kwargs={"sql": "SELECT *"})
        result = my_before_tool_hook(ctx)
        assert result is None  # proceed
    """
    run = MagicMock()
    run.run_id = run_id
    run.tool_calls = 0
    run.model_calls = 0
    run.policy_violations = 0

    span = MagicMock()
    span.span_id = span_id
    span.run_id = run_id

    observe = MagicMock()
    observe._hooks = None

    ctx = ToolContext(
        run=run,
        span=span,
        observe=observe,
        tool_name=tool_name,
        tool_kind=tool_kind,
        tool_version=tool_version,
        args=args or [],
        kwargs=kwargs or {},
        timestamp_ms=0,
    )

    # Apply any extra attributes
    for key, value in extra.items():
        setattr(ctx, key, value)

    return ctx


def mock_model_context(
    *,
    provider: str = "openai",
    model: str = "gpt-4",
    messages: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    run_id: str = "test-run-id",
    span_id: str = "test-span-id",
    **extra: Any,
) -> ModelContext:
    """
    Create a mock ModelContext for testing before_model, after_model, on_model_error hooks.

    Args:
        provider: Model provider (e.g., "openai", "anthropic").
        model: Model name.
        messages: Message list for the LLM call.
        system_prompt: System prompt if present.
        args: Positional arguments.
        kwargs: Keyword arguments.
        run_id: Mock run ID.
        span_id: Mock span ID.
        **extra: Additional attributes to set.

    Returns:
        A ModelContext with mocked run, span, and observe objects.

    Example:
        ctx = mock_model_context(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        result = my_before_model_hook(ctx)
    """
    run = MagicMock()
    run.run_id = run_id
    run.tool_calls = 0
    run.model_calls = 0
    run.policy_violations = 0

    span = MagicMock()
    span.span_id = span_id
    span.run_id = run_id

    observe = MagicMock()
    observe._hooks = None

    ctx = ModelContext(
        run=run,
        span=span,
        observe=observe,
        provider=provider,
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        args=args or [],
        kwargs=kwargs or {},
        timestamp_ms=0,
    )

    for key, value in extra.items():
        setattr(ctx, key, value)

    return ctx


def mock_run_start_context(
    *,
    name: str = "test-run",
    run_id: str = "test-run-id",
    **extra: Any,
) -> RunStartContext:
    """
    Create a mock RunStartContext for testing on_run_start hooks.

    Args:
        name: Run name.
        run_id: Mock run ID.
        **extra: Additional attributes.

    Returns:
        A RunStartContext with mocked run and observe objects.
    """
    run = MagicMock()
    run.run_id = run_id
    run.name = name

    observe = MagicMock()

    ctx = RunStartContext(
        run=run,
        observe=observe,
        timestamp_ms=0,
    )

    for key, value in extra.items():
        setattr(ctx, key, value)

    return ctx


def mock_run_end_context(
    *,
    name: str = "test-run",
    run_id: str = "test-run-id",
    status: str = "ok",
    error: Exception | None = None,
    duration_ms: int = 100,
    tool_calls: int = 0,
    model_calls: int = 0,
    policy_violations: int = 0,
    **extra: Any,
) -> RunEndContext:
    """
    Create a mock RunEndContext for testing on_run_end hooks.

    Args:
        name: Run name.
        run_id: Mock run ID.
        status: Run status ("ok" or "error").
        error: Optional exception if run errored.
        duration_ms: Run duration in milliseconds.
        tool_calls: Number of tool calls.
        model_calls: Number of model calls.
        policy_violations: Number of policy violations.
        **extra: Additional attributes.

    Returns:
        A RunEndContext with mocked run and observe objects.
    """
    run = MagicMock()
    run.run_id = run_id
    run.name = name
    run.tool_calls = tool_calls
    run.model_calls = model_calls
    run.policy_violations = policy_violations

    observe = MagicMock()

    ctx = RunEndContext(
        run=run,
        observe=observe,
        status=status,
        error=error,
        duration_ms=duration_ms,
        timestamp_ms=0,
        tool_calls=tool_calls,
        model_calls=model_calls,
        policy_violations=policy_violations,
        spans=[],
    )

    for key, value in extra.items():
        setattr(ctx, key, value)

    return ctx


def mock_error_context(
    *,
    error: Exception | None = None,
    run_id: str = "test-run-id",
    span_id: str = "test-span-id",
    **extra: Any,
) -> ErrorContext:
    """
    Create a mock ErrorContext for testing on_tool_error, on_model_error hooks.

    Args:
        error: The exception that occurred.
        run_id: Mock run ID.
        span_id: Mock span ID.
        **extra: Additional attributes.

    Returns:
        An ErrorContext with mocked run, span, and observe objects.
    """
    run = MagicMock()
    run.run_id = run_id

    span = MagicMock()
    span.span_id = span_id
    span.run_id = run_id

    observe = MagicMock()

    ctx = ErrorContext(
        run=run,
        span=span,
        observe=observe,
        error=error or ValueError("Test error"),
        timestamp_ms=0,
    )

    for key, value in extra.items():
        setattr(ctx, key, value)

    return ctx


@dataclass
class HookCallRecord:
    """Record of a hook call for testing."""

    phase: str
    context: Any
    args: tuple[Any, ...] = field(default_factory=tuple)
    result: Any = None


class RecordingHookRegistry:
    """
    A hook registry that records all hook calls for testing.

    Usage:
        registry = RecordingHookRegistry()

        @registry.before_tool
        def my_hook(ctx):
            return HookResult.proceed()

        # Simulate tool call
        registry.run_before_hooks("before_tool", mock_tool_context(), [], {})

        # Check recorded calls
        assert len(registry.calls) == 1
        assert registry.calls[0].phase == "before_tool"
    """

    def __init__(self) -> None:
        self.calls: list[HookCallRecord] = []
        self._hooks: dict[str, list[Any]] = {}

    def before_tool(self, func: Any) -> Any:
        """Register a before_tool hook."""
        self._hooks.setdefault("before_tool", []).append(func)
        return func

    def after_tool(self, func: Any) -> Any:
        """Register an after_tool hook."""
        self._hooks.setdefault("after_tool", []).append(func)
        return func

    def on_tool_error(self, func: Any) -> Any:
        """Register an on_tool_error hook."""
        self._hooks.setdefault("on_tool_error", []).append(func)
        return func

    def before_model(self, func: Any) -> Any:
        """Register a before_model hook."""
        self._hooks.setdefault("before_model", []).append(func)
        return func

    def after_model(self, func: Any) -> Any:
        """Register an after_model hook."""
        self._hooks.setdefault("after_model", []).append(func)
        return func

    def on_model_error(self, func: Any) -> Any:
        """Register an on_model_error hook."""
        self._hooks.setdefault("on_model_error", []).append(func)
        return func

    def on_run_start(self, func: Any) -> Any:
        """Register an on_run_start hook."""
        self._hooks.setdefault("on_run_start", []).append(func)
        return func

    def on_run_end(self, func: Any) -> Any:
        """Register an on_run_end hook."""
        self._hooks.setdefault("on_run_end", []).append(func)
        return func

    def run_before_hooks(
        self,
        phase: str,
        ctx: Any,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any], HookResult | None]:
        """Run before hooks and record calls."""
        for hook in self._hooks.get(phase, []):
            result = hook(ctx)
            self.calls.append(HookCallRecord(phase=phase, context=ctx, result=result))

            if result is not None and result.action in (HookAction.BLOCK, HookAction.SKIP):
                return args, kwargs, result
            if result is not None and result.action == HookAction.MODIFY:
                if result.args is not None:
                    args = result.args
                if result.kwargs is not None:
                    kwargs = {**kwargs, **result.kwargs}

        return args, kwargs, None

    def run_after_hooks(self, phase: str, ctx: Any, result: Any) -> Any:
        """Run after hooks and record calls."""
        for hook in self._hooks.get(phase, []):
            new_result = hook(ctx, result)
            self.calls.append(HookCallRecord(phase=phase, context=ctx, args=(result,), result=new_result))
            if new_result is not None:
                result = new_result
        return result

    def run_lifecycle_hooks(self, phase: str, ctx: Any) -> None:
        """Run lifecycle hooks and record calls."""
        for hook in self._hooks.get(phase, []):
            hook(ctx)
            self.calls.append(HookCallRecord(phase=phase, context=ctx))

    def clear_calls(self) -> None:
        """Clear recorded calls."""
        self.calls.clear()


def assert_hook_blocks(hook: Any, ctx: Any, reason_contains: str | None = None) -> None:
    """
    Assert that a hook returns a BLOCK result.

    Args:
        hook: The hook function to test.
        ctx: The context to pass to the hook.
        reason_contains: Optional substring that should be in the block reason.

    Raises:
        AssertionError: If the hook doesn't block or reason doesn't match.

    Example:
        def my_security_hook(ctx):
            if "DROP" in ctx.kwargs.get("sql", ""):
                return HookResult.block("DROP statements not allowed")
            return None

        ctx = mock_tool_context(kwargs={"sql": "DROP TABLE users"})
        assert_hook_blocks(my_security_hook, ctx, "DROP")
    """
    result = hook(ctx)
    assert result is not None, f"Hook returned None, expected BLOCK result"
    assert result.action == HookAction.BLOCK, f"Hook returned {result.action}, expected BLOCK"
    if reason_contains is not None:
        assert reason_contains in (result.reason or ""), (
            f"Block reason '{result.reason}' doesn't contain '{reason_contains}'"
        )


def assert_hook_proceeds(hook: Any, ctx: Any) -> None:
    """
    Assert that a hook returns None or PROCEED (allowing execution to continue).

    Args:
        hook: The hook function to test.
        ctx: The context to pass to the hook.

    Raises:
        AssertionError: If the hook doesn't proceed.

    Example:
        ctx = mock_tool_context(kwargs={"sql": "SELECT * FROM users"})
        assert_hook_proceeds(my_security_hook, ctx)
    """
    result = hook(ctx)
    if result is not None:
        assert result.action == HookAction.PROCEED, (
            f"Hook returned {result.action}, expected PROCEED or None"
        )


def assert_hook_skips(hook: Any, ctx: Any, expected_result: Any = None) -> Any:
    """
    Assert that a hook returns a SKIP result.

    Args:
        hook: The hook function to test.
        ctx: The context to pass to the hook.
        expected_result: Optional expected skip result value.

    Returns:
        The skip result value.

    Raises:
        AssertionError: If the hook doesn't skip or result doesn't match.
    """
    result = hook(ctx)
    assert result is not None, f"Hook returned None, expected SKIP result"
    assert result.action == HookAction.SKIP, f"Hook returned {result.action}, expected SKIP"
    if expected_result is not None:
        assert result.result == expected_result, (
            f"Skip result {result.result} doesn't match expected {expected_result}"
        )
    return result.result


def assert_hook_modifies(
    hook: Any,
    ctx: Any,
    expected_args: list[Any] | None = None,
    expected_kwargs: dict[str, Any] | None = None,
) -> HookResult:
    """
    Assert that a hook returns a MODIFY result.

    Args:
        hook: The hook function to test.
        ctx: The context to pass to the hook.
        expected_args: Optional expected modified args.
        expected_kwargs: Optional expected modified kwargs.

    Returns:
        The HookResult.

    Raises:
        AssertionError: If the hook doesn't modify or values don't match.
    """
    result = hook(ctx)
    assert result is not None, f"Hook returned None, expected MODIFY result"
    assert result.action == HookAction.MODIFY, f"Hook returned {result.action}, expected MODIFY"
    if expected_args is not None:
        assert result.args == expected_args, (
            f"Modified args {result.args} don't match expected {expected_args}"
        )
    if expected_kwargs is not None:
        assert result.kwargs == expected_kwargs, (
            f"Modified kwargs {result.kwargs} don't match expected {expected_kwargs}"
        )
    return result
