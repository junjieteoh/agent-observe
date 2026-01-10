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

from agent_observe.config import CaptureMode
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
from agent_observe.hooks.context import ErrorContext, ModelContext, ToolContext
from agent_observe.hooks.result import HookAction

logger = logging.getLogger(__name__)

# Maximum size for content storage in evidence_only mode
MAX_EVIDENCE_BYTES = 64 * 1024  # 64KB


def _serialize_for_storage(value: Any, max_bytes: int | None = None) -> str | None:
    """
    Serialize a value for storage in span attributes.

    Args:
        value: Value to serialize.
        max_bytes: Maximum size in bytes (None for unlimited).

    Returns:
        JSON string or None if serialization fails or exceeds limit.
    """
    import json

    try:
        serialized = json.dumps(value, default=str)
        if max_bytes is not None and len(serialized.encode("utf-8")) > max_bytes:
            return None  # Too large, skip storage
        return serialized
    except (TypeError, ValueError):
        return None


def _is_stream(result: Any) -> bool:
    """Check if result is a sync stream/generator (not a regular collection)."""
    import types

    # Check for sync generator type
    if isinstance(result, types.GeneratorType):
        return True

    # Check for common streaming response types (OpenAI, Anthropic) - but not async variants
    type_name = type(result).__name__.lower()
    if ("stream" in type_name or "iterator" in type_name) and "async" not in type_name:
        return True

    # Check if it's iterable but NOT a string/dict/list/tuple
    return (
        hasattr(result, "__iter__")
        and hasattr(result, "__next__")
        and not isinstance(result, (str, bytes, dict, list, tuple))
    )


def _is_async_stream(result: Any) -> bool:
    """Check if result is an async stream/generator."""
    import types

    # Check for async generator type
    if isinstance(result, types.AsyncGeneratorType):
        return True

    # Check for common async streaming response types
    type_name = type(result).__name__.lower()
    if ("stream" in type_name or "iterator" in type_name) and "async" in type_name:
        return True

    # Check if it has async iteration protocol
    return hasattr(result, "__aiter__") and hasattr(result, "__anext__")


def _extract_chunk_content(chunk: Any) -> str:
    """Extract text content from a streaming chunk."""
    # OpenAI format
    if hasattr(chunk, "choices") and chunk.choices:
        delta = getattr(chunk.choices[0], "delta", None)
        if delta and hasattr(delta, "content"):
            return delta.content or ""

    # Anthropic format
    if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
        return chunk.delta.text or ""

    # Dict format (generic)
    if isinstance(chunk, dict):
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0].get("delta", {})
            return delta.get("content", "")
        if "delta" in chunk:
            return chunk["delta"].get("text", "")

    # Fallback: convert to string
    return str(chunk) if chunk else ""


def _extract_llm_context(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Extract full LLM context from function arguments.

    This captures the complete context sent to the LLM for the Wide Event:
    - System prompt
    - Message history
    - Model configuration
    - Tools/functions
    """
    context: dict[str, Any] = {}

    # Try to find messages in kwargs or args
    messages = None
    if "messages" in kwargs:
        messages = kwargs["messages"]
    elif args and isinstance(args[0], list):
        # First positional arg might be messages
        messages = args[0]

    if messages and isinstance(messages, list):
        context["messages"] = messages

        # Extract system prompt from messages
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                context["system_prompt"] = msg.get("content")
                break

    # Extract model configuration
    for key in ("model", "temperature", "max_tokens", "top_p", "presence_penalty",
                "frequency_penalty", "stop", "stream"):
        if key in kwargs:
            context[key] = kwargs[key]

    # Extract tools/functions
    if "tools" in kwargs:
        context["tools"] = kwargs["tools"]
    if "functions" in kwargs:
        context["functions"] = kwargs["functions"]
    if "tool_choice" in kwargs:
        context["tool_choice"] = kwargs["tool_choice"]
    if "function_call" in kwargs:
        context["function_call"] = kwargs["function_call"]

    # Extract response format
    if "response_format" in kwargs:
        context["response_format"] = kwargs["response_format"]

    return context


def _format_error_context(
    error: Exception,
    capture_mode: CaptureMode,
    input_data: Any = None,
) -> dict[str, Any]:
    """
    Format error with context based on capture mode.

    Args:
        error: The exception that occurred.
        capture_mode: Current capture mode.
        input_data: Input that caused the error (for full mode).

    Returns:
        Structured error information.
    """
    import traceback

    error_info: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }

    # Add traceback for evidence_only and full modes
    if capture_mode in (CaptureMode.EVIDENCE_ONLY, CaptureMode.FULL):
        tb = traceback.format_exc()
        if capture_mode == CaptureMode.EVIDENCE_ONLY:
            # Truncate traceback in evidence mode
            max_tb_length = 4096
            if len(tb) > max_tb_length:
                tb = tb[:max_tb_length] + "\n... [truncated]"
        error_info["traceback"] = tb

    # Add input context in full mode
    if capture_mode == CaptureMode.FULL and input_data is not None:
        input_serialized = _serialize_for_storage(input_data)
        if input_serialized:
            error_info["input"] = input_serialized

    return error_info


class SyncStreamWrapper:
    """
    Wrapper for synchronous LLM streams that records metrics while yielding chunks.

    Captures:
    - Time to first token (TTFT)
    - Time to last token
    - Chunk count
    - Accumulated output (in full/evidence_only mode)
    """

    def __init__(
        self,
        stream: Any,
        span: SpanContext,
        capture_mode: CaptureMode,
        run: Any,
    ):
        self._stream = stream
        self._span = span
        self._capture_mode = capture_mode
        self._run = run
        self._chunks: list[Any] = []
        self._content_parts: list[str] = []
        self._first_token_recorded = False

    def __iter__(self) -> SyncStreamWrapper:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)

            # Record time to first token
            if not self._first_token_recorded:
                self._span.set_attribute("ts_first_token", now_ms())
                self._first_token_recorded = True

            # Accumulate chunks
            self._chunks.append(chunk)
            content = _extract_chunk_content(chunk)
            if content:
                self._content_parts.append(content)

            return chunk

        except StopIteration:
            # Stream exhausted - finalize metrics
            self._finalize()
            raise
        except Exception as e:
            # Stream errored - finalize with error status
            self._finalize_with_error(e)
            raise

    def _finalize_with_error(self, error: Exception) -> None:
        """Record final metrics when stream errors."""
        self._span.set_attribute("ts_last_token", now_ms())
        self._span.set_attribute("chunk_count", len(self._chunks))
        self._span.set_attribute("streaming", True)
        self._span.set_status(SpanStatus.ERROR, str(error))
        self._span.end()
        self._run.add_span(self._span)

    def _finalize(self) -> None:
        """Record final metrics when stream is exhausted."""
        self._span.set_attribute("ts_last_token", now_ms())
        self._span.set_attribute("chunk_count", len(self._chunks))
        self._span.set_attribute("streaming", True)

        # Reconstruct and hash output
        full_output = "".join(self._content_parts)
        output_hash = hash_json(full_output)
        self._span.set_attribute("output_hash", output_hash)

        # Store content based on capture mode
        if self._capture_mode == CaptureMode.FULL or (
            self._capture_mode == CaptureMode.EVIDENCE_ONLY
            and len(full_output.encode("utf-8")) <= MAX_EVIDENCE_BYTES
        ):
            self._span.set_attribute("output", full_output)

        self._span.set_status(SpanStatus.OK)
        self._span.end()
        self._run.add_span(self._span)


class AsyncStreamWrapper:
    """
    Wrapper for asynchronous LLM streams that records metrics while yielding chunks.
    """

    def __init__(
        self,
        stream: Any,
        span: SpanContext,
        capture_mode: CaptureMode,
        run: Any,
    ):
        self._stream = stream
        self._span = span
        self._capture_mode = capture_mode
        self._run = run
        self._chunks: list[Any] = []
        self._content_parts: list[str] = []
        self._first_token_recorded = False

    def __aiter__(self) -> AsyncStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()

            # Record time to first token
            if not self._first_token_recorded:
                self._span.set_attribute("ts_first_token", now_ms())
                self._first_token_recorded = True

            # Accumulate chunks
            self._chunks.append(chunk)
            content = _extract_chunk_content(chunk)
            if content:
                self._content_parts.append(content)

            return chunk

        except StopAsyncIteration:
            # Stream exhausted - finalize metrics
            self._finalize()
            raise
        except Exception as e:
            # Stream errored - finalize with error status
            self._finalize_with_error(e)
            raise

    def _finalize_with_error(self, error: Exception) -> None:
        """Record final metrics when stream errors."""
        self._span.set_attribute("ts_last_token", now_ms())
        self._span.set_attribute("chunk_count", len(self._chunks))
        self._span.set_attribute("streaming", True)
        self._span.set_status(SpanStatus.ERROR, str(error))
        self._span.end()
        self._run.add_span(self._span)

    def _finalize(self) -> None:
        """Record final metrics when stream is exhausted."""
        self._span.set_attribute("ts_last_token", now_ms())
        self._span.set_attribute("chunk_count", len(self._chunks))
        self._span.set_attribute("streaming", True)

        # Reconstruct and hash output
        full_output = "".join(self._content_parts)
        output_hash = hash_json(full_output)
        self._span.set_attribute("output_hash", output_hash)

        # Store content based on capture mode
        if self._capture_mode == CaptureMode.FULL or (
            self._capture_mode == CaptureMode.EVIDENCE_ONLY
            and len(full_output.encode("utf-8")) <= MAX_EVIDENCE_BYTES
        ):
            self._span.set_attribute("output", full_output)

        self._span.set_status(SpanStatus.OK)
        self._span.end()
        self._run.add_span(self._span)


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
        policy_engine = run.get_policy_engine()
        if policy_engine is None:
            logger.debug("No policy engine configured, skipping tool policy check")
            return

        fail_on_violation = run.get_fail_on_violation()

        # Check tool allowed
        violation = policy_engine.check_tool_allowed(tool_name)
        if violation:
            run.record_policy_violation()
            observe._emit_event_internal(run, "policy.violation", violation.to_dict())
            if fail_on_violation:
                from agent_observe.policy import PolicyViolationError

                raise PolicyViolationError(
                    violation.message, violation.rule, violation.details
                )

        # Check tool call limit
        limit_violation = policy_engine.check_tool_call_limit(run.tool_calls)
        if limit_violation:
            run.record_policy_violation()
            observe._emit_event_internal(run, "policy.violation", limit_violation.to_dict())
            if fail_on_violation:
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

        # Convert args to list for hook modification
        current_args = list(args)
        current_kwargs = dict(kwargs)

        try:
            # Run before_tool hooks
            hooks = observe._hooks
            if hooks is not None:
                tool_ctx = ToolContext(
                    run=run,
                    span=span,
                    observe=observe,
                    tool_name=tool_name,
                    tool_kind=tool_kind,
                    tool_version=tool_version,
                    args=current_args,
                    kwargs=current_kwargs,
                    timestamp_ms=now_ms(),
                )
                current_args, current_kwargs, hook_result = hooks.run_before_hooks(
                    "before_tool", tool_ctx, current_args, current_kwargs
                )

                if hook_result is not None:
                    if hook_result.action == HookAction.BLOCK:
                        from agent_observe.policy import PolicyViolationError
                        span.set_status(SpanStatus.BLOCKED, hook_result.reason or "Blocked by hook")
                        span.set_attribute("hook.blocked_by", hook_result.hook_name)
                        raise PolicyViolationError(
                            hook_result.reason or "Blocked by hook",
                            rule=f"hook:{hook_result.hook_name}",
                        )
                    elif hook_result.action == HookAction.SKIP:
                        span.set_attribute("hook.skipped_by", hook_result.hook_name)
                        span.set_status(SpanStatus.OK)
                        return hook_result.result

            # Try replay cache first
            replay_cache = observe.replay_cache

            def execute() -> Any:
                return fn(*current_args, **current_kwargs)

            result, was_cached = replay_cache.execute_with_cache(
                tool_name, {"args": current_args, "kwargs": current_kwargs}, execute, tool_version
            )

            span.set_attribute("replay.hit", was_cached)

            # Run after_tool hooks
            if hooks is not None:
                tool_ctx = ToolContext(
                    run=run,
                    span=span,
                    observe=observe,
                    tool_name=tool_name,
                    tool_kind=tool_kind,
                    tool_version=tool_version,
                    args=current_args,
                    kwargs=current_kwargs,
                    timestamp_ms=now_ms(),
                )
                result = hooks.run_after_hooks("after_tool", tool_ctx, result)

            # Hash result
            result_hash = hash_json(result)
            span.set_attribute("result_hash", result_hash)

            # Store actual content based on capture mode
            capture_mode = run.get_capture_mode()
            if capture_mode == CaptureMode.FULL:
                # Full mode: store everything
                args_serialized = _serialize_for_storage(args_for_hash)
                result_serialized = _serialize_for_storage(result)
                if args_serialized:
                    span.set_attribute("args", args_serialized)
                if result_serialized:
                    span.set_attribute("result", result_serialized)
            elif capture_mode == CaptureMode.EVIDENCE_ONLY:
                # Evidence mode: store with size limit
                args_serialized = _serialize_for_storage(args_for_hash, MAX_EVIDENCE_BYTES)
                result_serialized = _serialize_for_storage(result, MAX_EVIDENCE_BYTES)
                if args_serialized:
                    span.set_attribute("args", args_serialized)
                if result_serialized:
                    span.set_attribute("result", result_serialized)

            span.set_status(SpanStatus.OK)
            return result

        except Exception as e:
            # Run on_tool_error hooks
            if hooks is not None:
                error_ctx = ErrorContext(
                    run=run,
                    span=span,
                    observe=observe,
                    error=e,
                    timestamp_ms=now_ms(),
                )
                hooks.run_error_hooks("on_tool_error", error_ctx, e)

            # Enhanced error context
            error_capture_mode = run.get_capture_mode()
            error_context = _format_error_context(e, error_capture_mode, args_for_hash)
            span.set_attribute("error_context", error_context)
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

        # Convert args to list for hook modification
        current_args = list(args)
        current_kwargs = dict(kwargs)

        try:
            # Run before_tool hooks
            hooks = observe._hooks
            if hooks is not None:
                tool_ctx = ToolContext(
                    run=run,
                    span=span,
                    observe=observe,
                    tool_name=tool_name,
                    tool_kind=tool_kind,
                    tool_version=tool_version,
                    args=current_args,
                    kwargs=current_kwargs,
                    timestamp_ms=now_ms(),
                )
                current_args, current_kwargs, hook_result = await hooks.run_before_hooks_async(
                    "before_tool", tool_ctx, current_args, current_kwargs
                )

                if hook_result is not None:
                    if hook_result.action == HookAction.BLOCK:
                        from agent_observe.policy import PolicyViolationError
                        span.set_status(SpanStatus.BLOCKED, hook_result.reason or "Blocked by hook")
                        span.set_attribute("hook.blocked_by", hook_result.hook_name)
                        raise PolicyViolationError(
                            hook_result.reason or "Blocked by hook",
                            rule=f"hook:{hook_result.hook_name}",
                        )
                    elif hook_result.action == HookAction.SKIP:
                        span.set_attribute("hook.skipped_by", hook_result.hook_name)
                        span.set_status(SpanStatus.OK)
                        return hook_result.result

            # Note: Replay cache doesn't support async yet, execute directly
            result = await fn(*current_args, **current_kwargs)

            span.set_attribute("replay.hit", False)

            # Run after_tool hooks
            if hooks is not None:
                tool_ctx = ToolContext(
                    run=run,
                    span=span,
                    observe=observe,
                    tool_name=tool_name,
                    tool_kind=tool_kind,
                    tool_version=tool_version,
                    args=current_args,
                    kwargs=current_kwargs,
                    timestamp_ms=now_ms(),
                )
                result = await hooks.run_after_hooks_async("after_tool", tool_ctx, result)

            # Hash result
            result_hash = hash_json(result)
            span.set_attribute("result_hash", result_hash)

            # Store actual content based on capture mode
            capture_mode = run.get_capture_mode()
            if capture_mode == CaptureMode.FULL:
                # Full mode: store everything
                args_serialized = _serialize_for_storage(args_for_hash)
                result_serialized = _serialize_for_storage(result)
                if args_serialized:
                    span.set_attribute("args", args_serialized)
                if result_serialized:
                    span.set_attribute("result", result_serialized)
            elif capture_mode == CaptureMode.EVIDENCE_ONLY:
                # Evidence mode: store with size limit
                args_serialized = _serialize_for_storage(args_for_hash, MAX_EVIDENCE_BYTES)
                result_serialized = _serialize_for_storage(result, MAX_EVIDENCE_BYTES)
                if args_serialized:
                    span.set_attribute("args", args_serialized)
                if result_serialized:
                    span.set_attribute("result", result_serialized)

            span.set_status(SpanStatus.OK)
            return result

        except Exception as e:
            # Run on_tool_error hooks
            if hooks is not None:
                error_ctx = ErrorContext(
                    run=run,
                    span=span,
                    observe=observe,
                    error=e,
                    timestamp_ms=now_ms(),
                )
                await hooks.run_error_hooks_async("on_tool_error", error_ctx, e)

            # Enhanced error context
            error_capture_mode = run.get_capture_mode()
            error_context = _format_error_context(e, error_capture_mode, args_for_hash)
            span.set_attribute("error_context", error_context)
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
        policy_engine = run.get_policy_engine()
        if policy_engine is None:
            logger.debug("No policy engine configured, skipping model policy check")
            return

        fail_on_violation = run.get_fail_on_violation()

        limit_violation = policy_engine.check_model_call_limit(run.model_calls)
        if limit_violation:
            run.record_policy_violation()
            observe._emit_event_internal(run, "policy.violation", limit_violation.to_dict())
            if fail_on_violation:
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
        capture_mode = run.get_capture_mode()

        # Check policies
        self._check_policies(run, observe)

        # Create span with proper parent
        span = self._create_span(provider, model, run.run_id)

        # Set as current span
        token = set_current_span(span)

        # Convert args to list for hook modification
        current_args = list(args)
        current_kwargs = dict(kwargs)

        # Extract LLM context for hooks
        llm_context = _extract_llm_context(args, kwargs)
        messages = llm_context.get("messages") if llm_context else None
        system_prompt = llm_context.get("system_prompt") if llm_context else None

        # Hash input
        input_for_hash = {"args": args, "kwargs": kwargs}
        input_hash = hash_json(input_for_hash)
        span.set_attribute("input_hash", input_hash)

        # v0.1.7: Store full LLM context (Wide Event)
        if llm_context and capture_mode in (CaptureMode.FULL, CaptureMode.EVIDENCE_ONLY):
            llm_context_serialized = _serialize_for_storage(
                llm_context,
                MAX_EVIDENCE_BYTES if capture_mode == CaptureMode.EVIDENCE_ONLY else None
            )
            if llm_context_serialized:
                span.set_attribute("llm_context", llm_context_serialized)

        # Store input based on capture mode
        if capture_mode == CaptureMode.FULL:
            input_serialized = _serialize_for_storage(input_for_hash)
            if input_serialized:
                span.set_attribute("input", input_serialized)
        elif capture_mode == CaptureMode.EVIDENCE_ONLY:
            input_serialized = _serialize_for_storage(input_for_hash, MAX_EVIDENCE_BYTES)
            if input_serialized:
                span.set_attribute("input", input_serialized)

        is_stream_result = False
        try:
            # Run before_model hooks
            hooks = observe._hooks
            if hooks is not None:
                model_ctx = ModelContext(
                    run=run,
                    span=span,
                    observe=observe,
                    provider=provider,
                    model=model,
                    messages=messages,
                    system_prompt=system_prompt,
                    args=current_args,
                    kwargs=current_kwargs,
                    timestamp_ms=now_ms(),
                )
                current_args, current_kwargs, hook_result = hooks.run_before_hooks(
                    "before_model", model_ctx, current_args, current_kwargs
                )

                if hook_result is not None:
                    if hook_result.action == HookAction.BLOCK:
                        from agent_observe.policy import PolicyViolationError
                        span.set_status(SpanStatus.BLOCKED, hook_result.reason or "Blocked by hook")
                        span.set_attribute("hook.blocked_by", hook_result.hook_name)
                        raise PolicyViolationError(
                            hook_result.reason or "Blocked by hook",
                            rule=f"hook:{hook_result.hook_name}",
                        )
                    elif hook_result.action == HookAction.SKIP:
                        span.set_attribute("hook.skipped_by", hook_result.hook_name)
                        span.set_status(SpanStatus.OK)
                        return hook_result.result

            result = fn(*current_args, **current_kwargs)

            # Check if result is a stream
            if _is_stream(result):
                is_stream_result = True
                # For streams, reset the current span context but DON'T end the span
                # The stream wrapper will handle finalization when exhausted
                # Note: after_model hooks are not called for streaming responses
                if token is not None:
                    reset_current_span(token)
                return SyncStreamWrapper(result, span, capture_mode, run)

            # Run after_model hooks (non-streaming only)
            if hooks is not None:
                model_ctx = ModelContext(
                    run=run,
                    span=span,
                    observe=observe,
                    provider=provider,
                    model=model,
                    messages=messages,
                    system_prompt=system_prompt,
                    args=current_args,
                    kwargs=current_kwargs,
                    timestamp_ms=now_ms(),
                )
                result = hooks.run_after_hooks("after_model", model_ctx, result)

            # Non-streaming: hash and store output
            output_hash = hash_json(result)
            span.set_attribute("output_hash", output_hash)
            span.set_attribute("streaming", False)

            # Store output based on capture mode
            if capture_mode == CaptureMode.FULL:
                output_serialized = _serialize_for_storage(result)
                if output_serialized:
                    span.set_attribute("output", output_serialized)
            elif capture_mode == CaptureMode.EVIDENCE_ONLY:
                output_serialized = _serialize_for_storage(result, MAX_EVIDENCE_BYTES)
                if output_serialized:
                    span.set_attribute("output", output_serialized)

            # Extract token usage if available in result
            self._extract_token_usage(span, result)

            span.set_status(SpanStatus.OK)
            return result

        except Exception as e:
            # Run on_model_error hooks
            if hooks is not None:
                error_ctx = ErrorContext(
                    run=run,
                    span=span,
                    observe=observe,
                    error=e,
                    timestamp_ms=now_ms(),
                )
                hooks.run_error_hooks("on_model_error", error_ctx, e)

            # Enhanced error context
            error_context = _format_error_context(e, capture_mode, input_for_hash)
            span.set_attribute("error_context", error_context)
            span.set_status(SpanStatus.ERROR, str(e))
            run.record_retry()
            raise

        finally:
            # Only finalize span if we didn't return a stream wrapper
            # (stream wrapper handles its own finalization)
            if not is_stream_result:
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
        capture_mode = run.get_capture_mode()

        # Check policies
        self._check_policies(run, observe)

        # Create span with proper parent
        span = self._create_span(provider, model, run.run_id)

        # Set as current span
        token = set_current_span(span)

        # Convert args to list for hook modification
        current_args = list(args)
        current_kwargs = dict(kwargs)

        # Extract LLM context for hooks
        llm_context = _extract_llm_context(args, kwargs)
        messages = llm_context.get("messages") if llm_context else None
        system_prompt = llm_context.get("system_prompt") if llm_context else None

        # Hash input
        input_for_hash = {"args": args, "kwargs": kwargs}
        input_hash = hash_json(input_for_hash)
        span.set_attribute("input_hash", input_hash)

        # v0.1.7: Store full LLM context (Wide Event)
        if llm_context and capture_mode in (CaptureMode.FULL, CaptureMode.EVIDENCE_ONLY):
            llm_context_serialized = _serialize_for_storage(
                llm_context,
                MAX_EVIDENCE_BYTES if capture_mode == CaptureMode.EVIDENCE_ONLY else None
            )
            if llm_context_serialized:
                span.set_attribute("llm_context", llm_context_serialized)

        # Store input based on capture mode
        if capture_mode == CaptureMode.FULL:
            input_serialized = _serialize_for_storage(input_for_hash)
            if input_serialized:
                span.set_attribute("input", input_serialized)
        elif capture_mode == CaptureMode.EVIDENCE_ONLY:
            input_serialized = _serialize_for_storage(input_for_hash, MAX_EVIDENCE_BYTES)
            if input_serialized:
                span.set_attribute("input", input_serialized)

        is_stream_result = False
        try:
            # Run before_model hooks
            hooks = observe._hooks
            if hooks is not None:
                model_ctx = ModelContext(
                    run=run,
                    span=span,
                    observe=observe,
                    provider=provider,
                    model=model,
                    messages=messages,
                    system_prompt=system_prompt,
                    args=current_args,
                    kwargs=current_kwargs,
                    timestamp_ms=now_ms(),
                )
                current_args, current_kwargs, hook_result = await hooks.run_before_hooks_async(
                    "before_model", model_ctx, current_args, current_kwargs
                )

                if hook_result is not None:
                    if hook_result.action == HookAction.BLOCK:
                        from agent_observe.policy import PolicyViolationError
                        span.set_status(SpanStatus.BLOCKED, hook_result.reason or "Blocked by hook")
                        span.set_attribute("hook.blocked_by", hook_result.hook_name)
                        raise PolicyViolationError(
                            hook_result.reason or "Blocked by hook",
                            rule=f"hook:{hook_result.hook_name}",
                        )
                    elif hook_result.action == HookAction.SKIP:
                        span.set_attribute("hook.skipped_by", hook_result.hook_name)
                        span.set_status(SpanStatus.OK)
                        return hook_result.result

            result = await fn(*current_args, **current_kwargs)

            # Check if result is an async stream
            if _is_async_stream(result):
                is_stream_result = True
                # For streams, reset the current span context but DON'T end the span
                # The stream wrapper will handle finalization when exhausted
                # Note: after_model hooks are not called for streaming responses
                if token is not None:
                    reset_current_span(token)
                return AsyncStreamWrapper(result, span, capture_mode, run)

            # Run after_model hooks (non-streaming only)
            if hooks is not None:
                model_ctx = ModelContext(
                    run=run,
                    span=span,
                    observe=observe,
                    provider=provider,
                    model=model,
                    messages=messages,
                    system_prompt=system_prompt,
                    args=current_args,
                    kwargs=current_kwargs,
                    timestamp_ms=now_ms(),
                )
                result = await hooks.run_after_hooks_async("after_model", model_ctx, result)

            # Non-streaming: hash and store output
            output_hash = hash_json(result)
            span.set_attribute("output_hash", output_hash)
            span.set_attribute("streaming", False)

            # Store output based on capture mode
            if capture_mode == CaptureMode.FULL:
                output_serialized = _serialize_for_storage(result)
                if output_serialized:
                    span.set_attribute("output", output_serialized)
            elif capture_mode == CaptureMode.EVIDENCE_ONLY:
                output_serialized = _serialize_for_storage(result, MAX_EVIDENCE_BYTES)
                if output_serialized:
                    span.set_attribute("output", output_serialized)

            # Extract token usage if available in result
            self._extract_token_usage(span, result)

            span.set_status(SpanStatus.OK)
            return result

        except Exception as e:
            # Run on_model_error hooks
            if hooks is not None:
                error_ctx = ErrorContext(
                    run=run,
                    span=span,
                    observe=observe,
                    error=e,
                    timestamp_ms=now_ms(),
                )
                await hooks.run_error_hooks_async("on_model_error", error_ctx, e)

            # Enhanced error context
            error_context = _format_error_context(e, capture_mode, input_for_hash)
            span.set_attribute("error_context", error_context)
            span.set_status(SpanStatus.ERROR, str(e))
            run.record_retry()
            raise

        finally:
            # Only finalize span if we didn't return a stream wrapper
            # (stream wrapper handles its own finalization)
            if not is_stream_result:
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
