"""
Hook registry for managing lifecycle hooks.

The HookRegistry stores hooks and provides methods for registration,
execution, and management.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from agent_observe.hooks.result import HookAction, HookResult

if TYPE_CHECKING:
    from agent_observe.hooks.circuit_breaker import CircuitBreakerConfig

logger = logging.getLogger(__name__)

# Lock for thread-safe hook registration
_registry_lock = threading.Lock()

# Type variable for hook functions
HookFunc = TypeVar("HookFunc", bound=Callable[..., Any])

# Valid hook phases
HOOK_PHASES = frozenset(
    {
        "before_tool",
        "after_tool",
        "on_tool_error",
        "before_model",
        "after_model",
        "on_model_error",
        "on_run_start",
        "on_run_end",
        "on_span_start",
        "on_span_end",
    }
)

# Default priority (lower = runs first)
DEFAULT_PRIORITY = 50


@dataclass
class RegisteredHook:
    """A registered hook with its metadata."""

    func: Callable[..., Any]
    name: str
    phase: str
    priority: int = DEFAULT_PRIORITY
    environments: list[str] | None = None  # None = all environments
    sample_rate: float = 1.0  # 1.0 = always run
    timeout_ms: int | None = None  # None = no timeout
    enabled: bool = True

    # Runtime stats (thread-safe via lock)
    _stats_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _call_count: int = field(default=0, repr=False)
    _error_count: int = field(default=0, repr=False)
    _total_latency_ms: float = field(default=0.0, repr=False)

    @property
    def call_count(self) -> int:
        """Thread-safe access to call count."""
        with self._stats_lock:
            return self._call_count

    @property
    def error_count(self) -> int:
        """Thread-safe access to error count."""
        with self._stats_lock:
            return self._error_count

    @property
    def total_latency_ms(self) -> float:
        """Thread-safe access to total latency."""
        with self._stats_lock:
            return self._total_latency_ms

    def record_call(self, latency_ms: float) -> None:
        """Thread-safe recording of a successful call."""
        with self._stats_lock:
            self._call_count += 1
            self._total_latency_ms += latency_ms

    def record_error(self) -> None:
        """Thread-safe recording of an error."""
        with self._stats_lock:
            self._error_count += 1

    def should_run(self, current_env: str | None = None) -> bool:
        """Check if this hook should run given current conditions."""
        if not self.enabled:
            return False

        # Check environment filter
        if self.environments is not None and current_env is not None:
            if current_env not in self.environments:
                return False

        # Check sample rate
        if self.sample_rate < 1.0:
            if random.random() > self.sample_rate:
                return False

        return True

    @property
    def is_async(self) -> bool:
        """Check if the hook function is async."""
        return asyncio.iscoroutinefunction(self.func)


class HookRegistry:
    """
    Registry for lifecycle hooks.

    Provides decorator syntax and explicit registration for hooks.
    Handles hook execution with priority ordering and error isolation.

    Usage:
        # Decorator syntax
        @observe.hooks.before_tool
        def my_hook(ctx):
            pass

        # With options
        @observe.hooks.before_tool(priority=0, environments=["prod"])
        def security_hook(ctx):
            pass

        # Explicit registration
        observe.hooks.register("before_tool", my_func, priority=10)
    """

    def __init__(
        self,
        current_env: str | None = None,
        hook_errors: str = "log",  # "log" or "raise"
        debug: bool = False,
        circuit_breaker: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize the hook registry.

        Args:
            current_env: Current environment for filtering hooks.
            hook_errors: How to handle hook errors - "log" or "raise".
            debug: Enable debug logging of hook execution.
            circuit_breaker: Configuration for circuit breaker protection.
        """
        self._hooks: dict[str, list[RegisteredHook]] = {phase: [] for phase in HOOK_PHASES}
        self._current_env = current_env
        self._hook_errors = hook_errors
        self._debug = debug

        # Initialize circuit breaker if configured
        self._circuit_breaker = None
        if circuit_breaker is not None:
            from agent_observe.hooks.circuit_breaker import CircuitBreakerRegistry

            self._circuit_breaker = CircuitBreakerRegistry(circuit_breaker)

    # -------------------------------------------------------------------------
    # Decorator properties for each phase
    # -------------------------------------------------------------------------

    @property
    def before_tool(self) -> _HookDecorator:
        """Decorator for before_tool hooks."""
        return _HookDecorator(self, "before_tool")

    @property
    def after_tool(self) -> _HookDecorator:
        """Decorator for after_tool hooks."""
        return _HookDecorator(self, "after_tool")

    @property
    def on_tool_error(self) -> _HookDecorator:
        """Decorator for on_tool_error hooks."""
        return _HookDecorator(self, "on_tool_error")

    @property
    def before_model(self) -> _HookDecorator:
        """Decorator for before_model hooks."""
        return _HookDecorator(self, "before_model")

    @property
    def after_model(self) -> _HookDecorator:
        """Decorator for after_model hooks."""
        return _HookDecorator(self, "after_model")

    @property
    def on_model_error(self) -> _HookDecorator:
        """Decorator for on_model_error hooks."""
        return _HookDecorator(self, "on_model_error")

    @property
    def on_run_start(self) -> _HookDecorator:
        """Decorator for on_run_start hooks."""
        return _HookDecorator(self, "on_run_start")

    @property
    def on_run_end(self) -> _HookDecorator:
        """Decorator for on_run_end hooks."""
        return _HookDecorator(self, "on_run_end")

    @property
    def on_span_start(self) -> _HookDecorator:
        """Decorator for on_span_start hooks."""
        return _HookDecorator(self, "on_span_start")

    @property
    def on_span_end(self) -> _HookDecorator:
        """Decorator for on_span_end hooks."""
        return _HookDecorator(self, "on_span_end")

    # -------------------------------------------------------------------------
    # Registration methods
    # -------------------------------------------------------------------------

    def register(
        self,
        phase: str,
        hook: Callable[..., Any],
        *,
        priority: int = DEFAULT_PRIORITY,
        environments: list[str] | None = None,
        sample_rate: float = 1.0,
        timeout_ms: int | None = None,
        name: str | None = None,
    ) -> None:
        """
        Register a hook for a specific phase.

        Args:
            phase: Hook phase (e.g., "before_tool", "on_run_end").
            hook: The hook function to register.
            priority: Execution priority (lower = runs first).
            environments: List of environments where this hook runs.
            sample_rate: Fraction of calls to run this hook (0.0-1.0).
            timeout_ms: Maximum execution time in milliseconds.
            name: Custom name for the hook (default: function name).

        Raises:
            ValueError: If phase is invalid or hook is not callable.
        """
        if phase not in HOOK_PHASES:
            raise ValueError(f"Invalid hook phase: {phase}. Valid phases: {HOOK_PHASES}")

        if not callable(hook):
            raise ValueError(f"Hook must be callable, got {type(hook).__name__}")

        # Validate sample_rate
        sample_rate = max(0.0, min(1.0, sample_rate))

        hook_name = name or getattr(hook, "__name__", str(hook))

        registered = RegisteredHook(
            func=hook,
            name=hook_name,
            phase=phase,
            priority=priority,
            environments=environments,
            sample_rate=sample_rate,
            timeout_ms=timeout_ms,
        )

        with _registry_lock:
            # Check for duplicate registration
            existing_names = {h.name for h in self._hooks[phase]}
            if hook_name in existing_names:
                logger.warning(f"Hook '{hook_name}' already registered for phase '{phase}', skipping")
                return

            self._hooks[phase].append(registered)
            # Sort by priority (lower first)
            self._hooks[phase].sort(key=lambda h: h.priority)

        logger.debug(f"Registered hook '{hook_name}' for phase '{phase}' (priority={priority})")

    def unregister(self, phase: str, hook: Callable[..., Any] | str) -> bool:
        """
        Unregister a hook.

        Args:
            phase: Hook phase.
            hook: The hook function or its name.

        Returns:
            True if hook was found and removed.
        """
        if phase not in HOOK_PHASES:
            return False

        with _registry_lock:
            hooks = self._hooks[phase]
            for i, registered in enumerate(hooks):
                if registered.func is hook or registered.name == hook:
                    hooks.pop(i)
                    logger.debug(f"Unregistered hook '{registered.name}' from phase '{phase}'")
                    return True
        return False

    def enable(self, hook_name: str) -> bool:
        """Enable a hook by name."""
        for phase_hooks in self._hooks.values():
            for hook in phase_hooks:
                if hook.name == hook_name:
                    hook.enabled = True
                    logger.debug(f"Enabled hook '{hook_name}'")
                    return True
        return False

    def disable(self, hook_name: str) -> bool:
        """Disable a hook by name."""
        for phase_hooks in self._hooks.values():
            for hook in phase_hooks:
                if hook.name == hook_name:
                    hook.enabled = False
                    logger.debug(f"Disabled hook '{hook_name}'")
                    return True
        return False

    def set_sample_rate(self, hook_name: str, sample_rate: float) -> bool:
        """Set the sample rate for a hook."""
        for phase_hooks in self._hooks.values():
            for hook in phase_hooks:
                if hook.name == hook_name:
                    hook.sample_rate = max(0.0, min(1.0, sample_rate))
                    logger.debug(f"Set sample rate for '{hook_name}' to {hook.sample_rate}")
                    return True
        return False

    def clear(self, phase: str | None = None) -> None:
        """Clear all hooks for a phase, or all hooks if phase is None."""
        if phase is None:
            for p in HOOK_PHASES:
                self._hooks[p].clear()
            logger.debug("Cleared all hooks")
        elif phase in HOOK_PHASES:
            self._hooks[phase].clear()
            logger.debug(f"Cleared hooks for phase '{phase}'")

    def list(self) -> dict[str, list[str]]:
        """List all registered hooks by phase."""
        return {phase: [h.name for h in hooks] for phase, hooks in self._hooks.items() if hooks}

    def status(self) -> dict[str, dict[str, Any]]:
        """Get status of all hooks."""
        result = {}
        for phase_hooks in self._hooks.values():
            for hook in phase_hooks:
                result[hook.name] = {
                    "phase": hook.phase,
                    "enabled": hook.enabled,
                    "priority": hook.priority,
                    "sample_rate": hook.sample_rate,
                    "call_count": hook.call_count,
                    "error_count": hook.error_count,
                    "avg_latency_ms": (
                        hook.total_latency_ms / hook.call_count if hook.call_count > 0 else 0
                    ),
                }
        return result

    # -------------------------------------------------------------------------
    # Execution methods
    # -------------------------------------------------------------------------

    def run_before_hooks(
        self,
        phase: str,
        ctx: Any,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any], HookResult | None]:
        """
        Run before hooks (before_tool, before_model).

        Returns:
            Tuple of (modified_args, modified_kwargs, blocking_result).
            If blocking_result is not None, execution should be blocked/skipped.
        """
        hooks = self._get_runnable_hooks(phase)

        for hook in hooks:
            # Check circuit breaker
            if self._circuit_breaker and not self._circuit_breaker.should_allow(hook.name):
                if self._debug:
                    logger.info(f"[HOOK] {phase}:{hook.name} SKIPPED (circuit breaker open)")
                continue

            try:
                start_time = time.perf_counter()

                if self._debug:
                    # Don't log args/kwargs - they may contain sensitive data
                    logger.info(f"[HOOK] {phase}:{hook.name} START")

                result = self._call_hook(hook, ctx)

                latency_ms = (time.perf_counter() - start_time) * 1000
                hook.record_call(latency_ms)

                # Record success with circuit breaker
                if self._circuit_breaker:
                    self._circuit_breaker.record_success(hook.name)

                if self._debug:
                    action = result.action.value if result else "proceed"
                    logger.info(f"[HOOK] {phase}:{hook.name} END action={action} latency={latency_ms:.2f}ms")

                if result is None:
                    continue  # Proceed to next hook

                result.hook_name = hook.name

                if result.action == HookAction.BLOCK:
                    return args, kwargs, result

                if result.action == HookAction.SKIP:
                    return args, kwargs, result

                if result.action == HookAction.MODIFY:
                    if result.args is not None:
                        args = result.args
                    if result.kwargs is not None:
                        kwargs = {**kwargs, **result.kwargs}

                if result.action == HookAction.PENDING:
                    # TODO: Implement pending/approval in Phase 5
                    logger.warning("PENDING action not yet implemented, blocking instead")
                    return args, kwargs, HookResult.block("Pending approval not implemented")

            except Exception as e:
                hook.record_error()
                # Record failure with circuit breaker
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure(hook.name, e)
                self._handle_hook_error(hook, e)

        return args, kwargs, None

    async def run_before_hooks_async(
        self,
        phase: str,
        ctx: Any,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any], HookResult | None]:
        """Async version of run_before_hooks."""
        hooks = self._get_runnable_hooks(phase)

        for hook in hooks:
            # Check circuit breaker
            if self._circuit_breaker and not self._circuit_breaker.should_allow(hook.name):
                if self._debug:
                    logger.info(f"[HOOK] {phase}:{hook.name} SKIPPED (circuit breaker open)")
                continue

            try:
                start_time = time.perf_counter()

                if self._debug:
                    logger.info(f"[HOOK] {phase}:{hook.name} START")

                result = await self._call_hook_async(hook, ctx)

                latency_ms = (time.perf_counter() - start_time) * 1000
                hook.record_call(latency_ms)

                if self._circuit_breaker:
                    self._circuit_breaker.record_success(hook.name)

                if self._debug:
                    action = result.action.value if result else "proceed"
                    logger.info(f"[HOOK] {phase}:{hook.name} END action={action} latency={latency_ms:.2f}ms")

                if result is None:
                    continue

                result.hook_name = hook.name

                if result.action == HookAction.BLOCK:
                    return args, kwargs, result

                if result.action == HookAction.SKIP:
                    return args, kwargs, result

                if result.action == HookAction.MODIFY:
                    if result.args is not None:
                        args = result.args
                    if result.kwargs is not None:
                        kwargs = {**kwargs, **result.kwargs}

                if result.action == HookAction.PENDING:
                    logger.warning("PENDING action not yet implemented, blocking instead")
                    return args, kwargs, HookResult.block("Pending approval not implemented")

            except Exception as e:
                hook.record_error()
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure(hook.name, e)
                self._handle_hook_error(hook, e)

        return args, kwargs, None

    def run_after_hooks(
        self,
        phase: str,
        ctx: Any,
        result: Any,
    ) -> Any:
        """
        Run after hooks (after_tool, after_model).

        After hooks can transform the result by returning a new value.

        Returns:
            The (possibly transformed) result.
        """
        hooks = self._get_runnable_hooks(phase)

        for hook in hooks:
            if self._circuit_breaker and not self._circuit_breaker.should_allow(hook.name):
                continue

            try:
                start_time = time.perf_counter()

                if self._debug:
                    logger.info(f"[HOOK] {phase}:{hook.name} START")

                new_result = self._call_hook(hook, ctx, result)

                latency_ms = (time.perf_counter() - start_time) * 1000
                hook.record_call(latency_ms)

                if self._circuit_breaker:
                    self._circuit_breaker.record_success(hook.name)

                if self._debug:
                    logger.info(f"[HOOK] {phase}:{hook.name} END latency={latency_ms:.2f}ms")

                if new_result is not None:
                    result = new_result

            except Exception as e:
                hook.record_error()
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure(hook.name, e)
                self._handle_hook_error(hook, e)

        return result

    async def run_after_hooks_async(
        self,
        phase: str,
        ctx: Any,
        result: Any,
    ) -> Any:
        """Async version of run_after_hooks."""
        hooks = self._get_runnable_hooks(phase)

        for hook in hooks:
            if self._circuit_breaker and not self._circuit_breaker.should_allow(hook.name):
                continue

            try:
                start_time = time.perf_counter()

                if self._debug:
                    logger.info(f"[HOOK] {phase}:{hook.name} START")

                new_result = await self._call_hook_async(hook, ctx, result)

                latency_ms = (time.perf_counter() - start_time) * 1000
                hook.record_call(latency_ms)

                if self._circuit_breaker:
                    self._circuit_breaker.record_success(hook.name)

                if self._debug:
                    logger.info(f"[HOOK] {phase}:{hook.name} END latency={latency_ms:.2f}ms")

                if new_result is not None:
                    result = new_result

            except Exception as e:
                hook.record_error()
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure(hook.name, e)
                self._handle_hook_error(hook, e)

        return result

    def run_error_hooks(
        self,
        phase: str,
        ctx: Any,
        error: Exception,
    ) -> None:
        """
        Run error hooks (on_tool_error, on_model_error).

        Error hooks cannot swallow the error - it will still be raised.
        """
        hooks = self._get_runnable_hooks(phase)

        for hook in hooks:
            if self._circuit_breaker and not self._circuit_breaker.should_allow(hook.name):
                continue

            try:
                start_time = time.perf_counter()

                if self._debug:
                    logger.info(f"[HOOK] {phase}:{hook.name} START (error: {type(error).__name__})")

                self._call_hook(hook, ctx, error)

                latency_ms = (time.perf_counter() - start_time) * 1000
                hook.record_call(latency_ms)

                if self._circuit_breaker:
                    self._circuit_breaker.record_success(hook.name)

                if self._debug:
                    logger.info(f"[HOOK] {phase}:{hook.name} END latency={latency_ms:.2f}ms")

            except Exception as e:
                hook.record_error()
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure(hook.name, e)
                self._handle_hook_error(hook, e)

    async def run_error_hooks_async(
        self,
        phase: str,
        ctx: Any,
        error: Exception,
    ) -> None:
        """Async version of run_error_hooks."""
        hooks = self._get_runnable_hooks(phase)

        for hook in hooks:
            if self._circuit_breaker and not self._circuit_breaker.should_allow(hook.name):
                continue

            try:
                start_time = time.perf_counter()
                await self._call_hook_async(hook, ctx, error)
                latency_ms = (time.perf_counter() - start_time) * 1000
                hook.record_call(latency_ms)
                if self._circuit_breaker:
                    self._circuit_breaker.record_success(hook.name)
            except Exception as e:
                hook.record_error()
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure(hook.name, e)
                self._handle_hook_error(hook, e)

    def run_lifecycle_hooks(self, phase: str, ctx: Any) -> None:
        """Run lifecycle hooks (on_run_start, on_run_end, on_span_start, on_span_end)."""
        hooks = self._get_runnable_hooks(phase)

        for hook in hooks:
            if self._circuit_breaker and not self._circuit_breaker.should_allow(hook.name):
                continue

            try:
                start_time = time.perf_counter()

                if self._debug:
                    logger.info(f"[HOOK] {phase}:{hook.name} START")

                self._call_hook(hook, ctx)

                latency_ms = (time.perf_counter() - start_time) * 1000
                hook.record_call(latency_ms)

                if self._circuit_breaker:
                    self._circuit_breaker.record_success(hook.name)

                if self._debug:
                    logger.info(f"[HOOK] {phase}:{hook.name} END latency={latency_ms:.2f}ms")

            except Exception as e:
                hook.record_error()
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure(hook.name, e)
                self._handle_hook_error(hook, e)

    async def run_lifecycle_hooks_async(self, phase: str, ctx: Any) -> None:
        """Async version of run_lifecycle_hooks."""
        hooks = self._get_runnable_hooks(phase)

        for hook in hooks:
            if self._circuit_breaker and not self._circuit_breaker.should_allow(hook.name):
                continue

            try:
                start_time = time.perf_counter()
                await self._call_hook_async(hook, ctx)
                latency_ms = (time.perf_counter() - start_time) * 1000
                hook.record_call(latency_ms)
                if self._circuit_breaker:
                    self._circuit_breaker.record_success(hook.name)
            except Exception as e:
                hook.record_error()
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure(hook.name, e)
                self._handle_hook_error(hook, e)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _get_runnable_hooks(self, phase: str) -> list[RegisteredHook]:
        """Get hooks that should run for this phase."""
        if phase not in self._hooks:
            return []
        # Take a snapshot to avoid issues if hooks are modified during iteration
        with _registry_lock:
            return [h for h in self._hooks[phase] if h.should_run(self._current_env)]

    def _call_hook(self, hook: RegisteredHook, *args: Any) -> Any:
        """Call a hook function (sync).

        Note: Async hooks called from sync context are run in a new event loop
        in a separate thread to avoid blocking the main thread and potential
        deadlocks.
        """
        if hook.is_async:
            # Run async hook in sync context using a thread pool
            # This avoids issues with nested event loops
            import concurrent.futures

            def run_async() -> Any:
                return asyncio.run(hook.func(*args))

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(run_async)
                return future.result(timeout=hook.timeout_ms / 1000 if hook.timeout_ms else None)
        else:
            return hook.func(*args)

    async def _call_hook_async(self, hook: RegisteredHook, *args: Any) -> Any:
        """Call a hook function (async).

        Sync hooks are called directly (they won't block the event loop
        unless they do I/O, which is the user's responsibility to avoid).
        """
        if hook.is_async:
            if hook.timeout_ms:
                return await asyncio.wait_for(
                    hook.func(*args),
                    timeout=hook.timeout_ms / 1000,
                )
            return await hook.func(*args)
        else:
            # Run sync hook - user is responsible for not blocking
            return hook.func(*args)

    def _handle_hook_error(self, hook: RegisteredHook, error: Exception) -> None:
        """Handle an error from a hook."""
        if self._hook_errors == "raise":
            raise error
        else:
            logger.warning(f"Hook '{hook.name}' raised error: {error}")


class _HookDecorator:
    """
    Decorator helper for registering hooks.

    Supports both @observe.hooks.before_tool and @observe.hooks.before_tool(priority=0).
    """

    def __init__(self, registry: HookRegistry, phase: str):
        self._registry = registry
        self._phase = phase

    def __call__(
        self,
        func: Callable[..., Any] | None = None,
        *,
        priority: int = DEFAULT_PRIORITY,
        environments: list[str] | None = None,
        sample_rate: float = 1.0,
        timeout_ms: int | None = None,
        name: str | None = None,
    ) -> Callable[..., Any]:
        """
        Register a hook function.

        Can be used as:
            @observe.hooks.before_tool
            def my_hook(ctx): ...

        Or with options:
            @observe.hooks.before_tool(priority=0)
            def my_hook(ctx): ...
        """
        if func is not None:
            # Called as @observe.hooks.before_tool (no parentheses)
            self._registry.register(self._phase, func)
            return func

        # Called as @observe.hooks.before_tool(...) (with options)
        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            self._registry.register(
                self._phase,
                f,
                priority=priority,
                environments=environments,
                sample_rate=sample_rate,
                timeout_ms=timeout_ms,
                name=name,
            )
            return f

        return decorator
