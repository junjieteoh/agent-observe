"""
Decorators for agent-observe.

Provides @tool and @model_call decorators for instrumenting
tool functions and model invocations. Supports both sync and async functions.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, TypeVar, overload

from agent_observe.context import (
    SpanContext,
    SpanKind,
    SpanStatus,
    generate_span_id,
    get_current_run,
    get_current_span,
    now_ms,
    reset_current_span,
    set_current_span,
)
from agent_observe.hashing import hash_json

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _get_parent_span_id() -> str | None:
    """Get the current parent span ID for nesting."""
    current_span = get_current_span()
    return current_span.span_id if current_span else None


class ToolDecorator:
    """
    Decorator for instrumenting tool functions.

    Supports both sync and async functions:

        @tool(name="query_db", kind="db", version="1")
        def query_database(sql: str) -> dict:
            ...

        @tool(name="fetch_data", kind="http")
        async def fetch_data(url: str) -> dict:
            ...

        # Or with defaults
        @tool
        def my_tool(arg: str) -> str:
            ...
    """

    def __init__(
        self,
        name: str | None = None,
        kind: str = "generic",
        version: str = "1",
    ):
        """
        Initialize tool decorator.

        Args:
            name: Tool name (default: function name).
            kind: Tool kind (e.g., "db", "http", "file", "generic").
            version: Tool version for replay cache.
        """
        self.name = name
        self.kind = kind
        self.version = version

    def __call__(self, fn: F) -> F:
        """Wrap the function with instrumentation."""
        tool_name = self.name or fn.__name__
        tool_kind = self.kind
        tool_version = self.version

        if asyncio.iscoroutinefunction(fn):
            # Async function
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self._execute_tool_async(
                    fn, tool_name, tool_kind, tool_version, args, kwargs
                )

            return async_wrapper  # type: ignore
        else:
            # Sync function
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self._execute_tool(
                    fn, tool_name, tool_kind, tool_version, args, kwargs
                )

            return wrapper  # type: ignore

    def _create_span(
        self,
        tool_name: str,
        tool_kind: str,
        tool_version: str,
        args_hash: str,
        run_id: str,
    ) -> SpanContext:
        """Create a span for the tool call."""
        return SpanContext(
            span_id=generate_span_id(),
            run_id=run_id,
            parent_span_id=_get_parent_span_id(),  # Fixed: now properly gets parent
            kind=SpanKind.TOOL,
            name=tool_name,
            ts_start=now_ms(),
            attrs={
                "tool.kind": tool_kind,
                "tool.version": tool_version,
                "args_hash": args_hash,
            },
        )

    def _check_policies(
        self,
        run: Any,
        observe: Any,
        tool_name: str,
    ) -> None:
        """Check tool policies and raise if violations block execution."""
        policy_engine = observe.policy_engine

        # Check tool allowed
        violation = policy_engine.check_tool_allowed(tool_name)
        if violation:
            run.record_policy_violation()
            observe._emit_event_internal(run, "policy.violation", violation.to_dict())
            if observe.config.fail_on_violation:
                from agent_observe.policy import PolicyViolationError

                raise PolicyViolationError(
                    violation.message, violation.rule, violation.details
                )

        # Check tool call limit
        limit_violation = policy_engine.check_tool_call_limit(run.tool_calls)
        if limit_violation:
            run.record_policy_violation()
            observe._emit_event_internal(run, "policy.violation", limit_violation.to_dict())
            if observe.config.fail_on_violation:
                from agent_observe.policy import PolicyViolationError

                raise PolicyViolationError(
                    limit_violation.message, limit_violation.rule, limit_violation.details
                )

    def _execute_tool(
        self,
        fn: Callable[..., Any],
        tool_name: str,
        tool_kind: str,
        tool_version: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute sync tool with full instrumentation."""
        run = get_current_run()

        # If no run context or no observe instance attached, just execute
        if run is None or run._observe is None:
            return fn(*args, **kwargs)

        observe = run._observe

        # Check policies
        self._check_policies(run, observe, tool_name)

        # Hash args for loop detection and replay
        args_for_hash = {"args": args, "kwargs": kwargs}
        args_hash = hash_json(args_for_hash)
        tool_call_hash = f"{tool_name}:{args_hash}"
        run.record_tool_call_hash(tool_call_hash)

        # Create span with proper parent
        span = self._create_span(tool_name, tool_kind, tool_version, args_hash, run.run_id)

        # Set as current span
        token = set_current_span(span)

        try:
            # Try replay cache first
            replay_cache = observe.replay_cache

            def execute() -> Any:
                return fn(*args, **kwargs)

            result, was_cached = replay_cache.execute_with_cache(
                tool_name, args_for_hash, execute, tool_version
            )

            span.set_attribute("replay.hit", was_cached)

            # Hash result
            result_hash = hash_json(result)
            span.set_attribute("result_hash", result_hash)

            span.set_status(SpanStatus.OK)
            return result

        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            run.record_retry()  # Record as potential retry
            raise

        finally:
            span.end()
            run.add_span(span)
            if token is not None:
                reset_current_span(token)

    async def _execute_tool_async(
        self,
        fn: Callable[..., Any],
        tool_name: str,
        tool_kind: str,
        tool_version: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute async tool with full instrumentation."""
        run = get_current_run()

        # If no run context or no observe instance attached, just execute
        if run is None or run._observe is None:
            return await fn(*args, **kwargs)

        observe = run._observe

        # Check policies
        self._check_policies(run, observe, tool_name)

        # Hash args for loop detection and replay
        args_for_hash = {"args": args, "kwargs": kwargs}
        args_hash = hash_json(args_for_hash)
        tool_call_hash = f"{tool_name}:{args_hash}"
        run.record_tool_call_hash(tool_call_hash)

        # Create span with proper parent
        span = self._create_span(tool_name, tool_kind, tool_version, args_hash, run.run_id)

        # Set as current span
        token = set_current_span(span)

        try:
            # Note: Replay cache doesn't support async yet, execute directly
            result = await fn(*args, **kwargs)

            span.set_attribute("replay.hit", False)

            # Hash result
            result_hash = hash_json(result)
            span.set_attribute("result_hash", result_hash)

            span.set_status(SpanStatus.OK)
            return result

        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            run.record_retry()  # Record as potential retry
            raise

        finally:
            span.end()
            run.add_span(span)
            if token is not None:
                reset_current_span(token)


class ModelCallDecorator:
    """
    Decorator for instrumenting model/LLM calls.

    Supports both sync and async functions:

        @model_call(provider="openai", model="gpt-4")
        def call_openai(prompt: str) -> str:
            ...

        @model_call(provider="anthropic", model="claude-3")
        async def call_claude(prompt: str) -> str:
            ...
    """

    def __init__(
        self,
        provider: str = "unknown",
        model: str = "unknown",
    ):
        """
        Initialize model call decorator.

        Args:
            provider: Model provider (e.g., "openai", "anthropic").
            model: Model name (e.g., "gpt-4", "claude-3").
        """
        self.provider = provider
        self.model = model

    def __call__(self, fn: F) -> F:
        """Wrap the function with instrumentation."""
        provider = self.provider
        model = self.model

        if asyncio.iscoroutinefunction(fn):
            # Async function
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self._execute_model_call_async(fn, provider, model, args, kwargs)

            return async_wrapper  # type: ignore
        else:
            # Sync function
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self._execute_model_call(fn, provider, model, args, kwargs)

            return wrapper  # type: ignore

    def _create_span(
        self,
        provider: str,
        model: str,
        run_id: str,
    ) -> SpanContext:
        """Create a span for the model call."""
        return SpanContext(
            span_id=generate_span_id(),
            run_id=run_id,
            parent_span_id=_get_parent_span_id(),  # Fixed: now properly gets parent
            kind=SpanKind.MODEL,
            name=f"{provider}.{model}",
            ts_start=now_ms(),
            attrs={
                "model.provider": provider,
                "model.name": model,
            },
        )

    def _check_policies(self, run: Any, observe: Any) -> None:
        """Check model call policies."""
        policy_engine = observe.policy_engine
        limit_violation = policy_engine.check_model_call_limit(run.model_calls)
        if limit_violation:
            run.record_policy_violation()
            observe._emit_event_internal(run, "policy.violation", limit_violation.to_dict())
            if observe.config.fail_on_violation:
                from agent_observe.policy import PolicyViolationError

                raise PolicyViolationError(
                    limit_violation.message, limit_violation.rule, limit_violation.details
                )

    def _execute_model_call(
        self,
        fn: Callable[..., Any],
        provider: str,
        model: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute sync model call with instrumentation."""
        run = get_current_run()

        # If no run context or no observe instance attached, just execute
        if run is None or run._observe is None:
            return fn(*args, **kwargs)

        observe = run._observe

        # Check policies
        self._check_policies(run, observe)

        # Create span with proper parent
        span = self._create_span(provider, model, run.run_id)

        # Set as current span
        token = set_current_span(span)

        try:
            # Hash input
            input_for_hash = {"args": args, "kwargs": kwargs}
            input_hash = hash_json(input_for_hash)
            span.set_attribute("input_hash", input_hash)

            result = fn(*args, **kwargs)

            # Hash output
            output_hash = hash_json(result)
            span.set_attribute("output_hash", output_hash)

            # Extract token usage if available in result
            self._extract_token_usage(span, result)

            span.set_status(SpanStatus.OK)
            return result

        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            run.record_retry()
            raise

        finally:
            span.end()
            run.add_span(span)
            if token is not None:
                reset_current_span(token)

    async def _execute_model_call_async(
        self,
        fn: Callable[..., Any],
        provider: str,
        model: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute async model call with instrumentation."""
        run = get_current_run()

        # If no run context or no observe instance attached, just execute
        if run is None or run._observe is None:
            return await fn(*args, **kwargs)

        observe = run._observe

        # Check policies
        self._check_policies(run, observe)

        # Create span with proper parent
        span = self._create_span(provider, model, run.run_id)

        # Set as current span
        token = set_current_span(span)

        try:
            # Hash input
            input_for_hash = {"args": args, "kwargs": kwargs}
            input_hash = hash_json(input_for_hash)
            span.set_attribute("input_hash", input_hash)

            result = await fn(*args, **kwargs)

            # Hash output
            output_hash = hash_json(result)
            span.set_attribute("output_hash", output_hash)

            # Extract token usage if available in result
            self._extract_token_usage(span, result)

            span.set_status(SpanStatus.OK)
            return result

        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            run.record_retry()
            raise

        finally:
            span.end()
            run.add_span(span)
            if token is not None:
                reset_current_span(token)

    def _extract_token_usage(self, span: SpanContext, result: Any) -> None:
        """Extract token usage from LLM response if available."""
        # Handle dict responses
        if isinstance(result, dict):
            usage = result.get("usage", {})
            if usage:
                if "prompt_tokens" in usage:
                    span.set_attribute("tokens.prompt", usage["prompt_tokens"])
                if "completion_tokens" in usage:
                    span.set_attribute("tokens.completion", usage["completion_tokens"])
                if "total_tokens" in usage:
                    span.set_attribute("tokens.total", usage["total_tokens"])

        # Handle OpenAI-style response objects
        elif hasattr(result, "usage") and result.usage is not None:
            usage = result.usage
            if hasattr(usage, "prompt_tokens"):
                span.set_attribute("tokens.prompt", usage.prompt_tokens)
            if hasattr(usage, "completion_tokens"):
                span.set_attribute("tokens.completion", usage.completion_tokens)
            if hasattr(usage, "total_tokens"):
                span.set_attribute("tokens.total", usage.total_tokens)


# Factory functions for clean API


@overload
def tool(fn: F) -> F: ...


@overload
def tool(
    fn: None = None,
    *,
    name: str | None = None,
    kind: str = "generic",
    version: str = "1",
) -> Callable[[F], F]: ...


def tool(
    fn: F | None = None,
    *,
    name: str | None = None,
    kind: str = "generic",
    version: str = "1",
) -> F | Callable[[F], F]:
    """
    Decorator for instrumenting tool functions.

    Can be used with or without arguments. Supports both sync and async:

        @tool
        def my_tool(arg: str) -> str:
            ...

        @tool(name="query_db", kind="db")
        async def query_database(sql: str) -> dict:
            ...
    """
    decorator = ToolDecorator(name=name, kind=kind, version=version)

    if fn is not None:
        # Called without arguments: @tool
        return decorator(fn)
    else:
        # Called with arguments: @tool(name=...)
        return decorator


def model_call(
    provider: str = "unknown",
    model: str = "unknown",
) -> Callable[[F], F]:
    """
    Decorator for instrumenting model/LLM calls.

    Supports both sync and async functions:

        @model_call(provider="openai", model="gpt-4")
        def call_openai(prompt: str) -> str:
            ...

        @model_call(provider="anthropic", model="claude-3")
        async def call_claude(messages: list) -> str:
            ...
    """
    return ModelCallDecorator(provider=provider, model=model)
