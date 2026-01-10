"""
Tests for agent-observe hooks.
"""

import pytest

from agent_observe.hooks import (
    HookAction,
    HookRegistry,
    HookResult,
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


# =============================================================================
# HookResult tests
# =============================================================================


class TestHookResult:
    """Tests for HookResult class."""

    def test_proceed(self):
        """Test HookResult.proceed()."""
        result = HookResult.proceed()
        assert result.action == HookAction.PROCEED
        assert not result.is_blocking()

    def test_block(self):
        """Test HookResult.block()."""
        result = HookResult.block("Dangerous operation")
        assert result.action == HookAction.BLOCK
        assert result.reason == "Dangerous operation"
        assert result.is_blocking()

    def test_skip(self):
        """Test HookResult.skip()."""
        cached_value = {"result": "cached"}
        result = HookResult.skip(cached_value)
        assert result.action == HookAction.SKIP
        assert result.result == cached_value
        assert result.is_blocking()

    def test_modify(self):
        """Test HookResult.modify()."""
        result = HookResult.modify(kwargs={"timeout": 30})
        assert result.action == HookAction.MODIFY
        assert result.kwargs == {"timeout": 30}
        assert not result.is_blocking()

    def test_pending(self):
        """Test HookResult.pending()."""
        result = HookResult.pending(timeout_seconds=60, on_timeout="proceed")
        assert result.action == HookAction.PENDING
        assert result.timeout_seconds == 60
        assert result.on_timeout == "proceed"


# =============================================================================
# HookRegistry tests
# =============================================================================


class TestHookRegistry:
    """Tests for HookRegistry class."""

    def test_register_hook(self):
        """Test registering a hook."""
        registry = HookRegistry()

        def my_hook(ctx):
            pass

        registry.register("before_tool", my_hook, priority=10)

        hooks = registry.list()
        assert "before_tool" in hooks
        assert "my_hook" in hooks["before_tool"]

    def test_decorator_syntax(self):
        """Test decorator-style hook registration."""
        registry = HookRegistry()

        @registry.before_tool
        def log_tool(ctx):
            pass

        @registry.after_model(priority=0)
        def process_response(ctx, result):
            return result

        hooks = registry.list()
        assert "log_tool" in hooks.get("before_tool", [])
        assert "process_response" in hooks.get("after_model", [])

    def test_hook_priority_ordering(self):
        """Test that hooks run in priority order."""
        registry = HookRegistry()
        call_order = []

        @registry.before_tool(priority=50)
        def middle(ctx):
            call_order.append("middle")

        @registry.before_tool(priority=0)
        def first(ctx):
            call_order.append("first")

        @registry.before_tool(priority=100)
        def last(ctx):
            call_order.append("last")

        ctx = mock_tool_context()
        registry.run_before_hooks("before_tool", ctx, [], {})

        assert call_order == ["first", "middle", "last"]

    def test_before_hooks_block(self):
        """Test that BLOCK action stops execution."""
        registry = HookRegistry()

        @registry.before_tool
        def blocker(ctx):
            return HookResult.block("Not allowed")

        @registry.before_tool
        def should_not_run(ctx):
            pytest.fail("This hook should not run")

        ctx = mock_tool_context()
        args, kwargs, result = registry.run_before_hooks("before_tool", ctx, [], {})

        assert result is not None
        assert result.action == HookAction.BLOCK
        assert result.reason == "Not allowed"

    def test_before_hooks_skip(self):
        """Test that SKIP action returns cached value."""
        registry = HookRegistry()
        cached = {"cached": True}

        @registry.before_tool
        def skipper(ctx):
            return HookResult.skip(cached)

        ctx = mock_tool_context()
        args, kwargs, result = registry.run_before_hooks("before_tool", ctx, [], {})

        assert result is not None
        assert result.action == HookAction.SKIP
        assert result.result == cached

    def test_before_hooks_modify(self):
        """Test that MODIFY action updates args/kwargs."""
        registry = HookRegistry()

        @registry.before_tool
        def add_timeout(ctx):
            return HookResult.modify(kwargs={"timeout": 30})

        ctx = mock_tool_context()
        args, kwargs, result = registry.run_before_hooks("before_tool", ctx, [], {"existing": "value"})

        assert result is None  # Proceeded
        assert kwargs["timeout"] == 30
        assert kwargs["existing"] == "value"

    def test_after_hooks_transform_result(self):
        """Test that after hooks can transform results."""
        registry = HookRegistry()

        @registry.after_tool
        def double_result(ctx, result):
            return result * 2

        ctx = mock_tool_context()
        result = registry.run_after_hooks("after_tool", ctx, 5)

        assert result == 10

    def test_lifecycle_hooks(self):
        """Test lifecycle hooks (on_run_start, on_run_end)."""
        registry = HookRegistry()
        events = []

        @registry.on_run_start
        def on_start(ctx):
            events.append(("start", ctx.run.name))

        @registry.on_run_end
        def on_end(ctx):
            events.append(("end", ctx.status))

        start_ctx = mock_run_start_context(name="test-run")
        end_ctx = mock_run_end_context(name="test-run", status="ok")

        registry.run_lifecycle_hooks("on_run_start", start_ctx)
        registry.run_lifecycle_hooks("on_run_end", end_ctx)

        assert events == [("start", "test-run"), ("end", "ok")]

    def test_environment_filtering(self):
        """Test that hooks respect environment filters."""
        registry = HookRegistry(current_env="prod")
        events = []

        @registry.before_tool(environments=["prod"])
        def prod_only(ctx):
            events.append("prod")

        @registry.before_tool(environments=["dev"])
        def dev_only(ctx):
            events.append("dev")

        @registry.before_tool
        def all_envs(ctx):
            events.append("all")

        ctx = mock_tool_context()
        registry.run_before_hooks("before_tool", ctx, [], {})

        assert "prod" in events
        assert "dev" not in events
        assert "all" in events

    def test_enable_disable_hooks(self):
        """Test enabling and disabling hooks."""
        registry = HookRegistry()
        events = []

        @registry.before_tool
        def my_hook(ctx):
            events.append("called")

        ctx = mock_tool_context()

        # Initially enabled
        registry.run_before_hooks("before_tool", ctx, [], {})
        assert events == ["called"]

        # Disable
        events.clear()
        registry.disable("my_hook")
        registry.run_before_hooks("before_tool", ctx, [], {})
        assert events == []

        # Re-enable
        registry.enable("my_hook")
        registry.run_before_hooks("before_tool", ctx, [], {})
        assert events == ["called"]

    def test_unregister_hook(self):
        """Test unregistering a hook."""
        registry = HookRegistry()

        @registry.before_tool
        def my_hook(ctx):
            pass

        assert "my_hook" in registry.list().get("before_tool", [])

        registry.unregister("before_tool", "my_hook")
        assert "my_hook" not in registry.list().get("before_tool", [])

    def test_hook_error_handling_log(self):
        """Test that hook errors are logged but don't crash."""
        registry = HookRegistry(hook_errors="log")
        events = []

        @registry.before_tool(priority=0)
        def failing_hook(ctx):
            raise ValueError("Hook error!")

        @registry.before_tool(priority=100)
        def should_still_run(ctx):
            events.append("ran")

        ctx = mock_tool_context()
        registry.run_before_hooks("before_tool", ctx, [], {})

        # Second hook should still run despite first failing
        assert "ran" in events

    def test_hook_error_handling_raise(self):
        """Test that hook errors can be raised."""
        registry = HookRegistry(hook_errors="raise")

        @registry.before_tool
        def failing_hook(ctx):
            raise ValueError("Hook error!")

        ctx = mock_tool_context()

        with pytest.raises(ValueError, match="Hook error!"):
            registry.run_before_hooks("before_tool", ctx, [], {})

    def test_hook_status(self):
        """Test getting hook status."""
        registry = HookRegistry()

        @registry.before_tool(priority=10)
        def my_hook(ctx):
            pass

        ctx = mock_tool_context()
        registry.run_before_hooks("before_tool", ctx, [], {})

        status = registry.status()
        assert "my_hook" in status
        assert status["my_hook"]["call_count"] == 1
        assert status["my_hook"]["priority"] == 10

    def test_clear_hooks(self):
        """Test clearing hooks."""
        registry = HookRegistry()

        @registry.before_tool
        def hook1(ctx):
            pass

        @registry.after_tool
        def hook2(ctx, result):
            return result

        # Clear specific phase
        registry.clear("before_tool")
        assert "before_tool" not in registry.list()
        assert "after_tool" in registry.list()

        # Clear all
        registry.clear()
        assert registry.list() == {}


# =============================================================================
# Testing utilities tests
# =============================================================================


class TestMockContexts:
    """Tests for mock context factories."""

    def test_mock_tool_context(self):
        """Test mock_tool_context creates valid context."""
        ctx = mock_tool_context(
            tool_name="query_db",
            tool_kind="db",
            kwargs={"sql": "SELECT *"},
        )

        assert ctx.tool_name == "query_db"
        assert ctx.tool_kind == "db"
        assert ctx.kwargs["sql"] == "SELECT *"
        assert ctx.run is not None
        assert ctx.span is not None

    def test_mock_model_context(self):
        """Test mock_model_context creates valid context."""
        ctx = mock_model_context(
            provider="anthropic",
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert ctx.provider == "anthropic"
        assert ctx.model == "claude-3"
        assert len(ctx.messages) == 1
        assert ctx.messages[0]["content"] == "Hello"

    def test_mock_run_end_context(self):
        """Test mock_run_end_context creates valid context."""
        ctx = mock_run_end_context(
            status="error",
            error=ValueError("Test error"),
            tool_calls=5,
        )

        assert ctx.status == "error"
        assert ctx.error is not None
        assert ctx.tool_calls == 5

    def test_mock_error_context(self):
        """Test mock_error_context creates valid context."""
        ctx = mock_error_context(error=RuntimeError("Boom!"))

        assert ctx.error_type == "RuntimeError"
        assert "Boom!" in ctx.error_message


class TestHookAssertions:
    """Tests for hook assertion helpers."""

    def test_assert_hook_blocks(self):
        """Test assert_hook_blocks works correctly."""

        def blocking_hook(ctx):
            return HookResult.block("Blocked!")

        ctx = mock_tool_context()
        assert_hook_blocks(blocking_hook, ctx, "Blocked")

    def test_assert_hook_proceeds(self):
        """Test assert_hook_proceeds works correctly."""

        def proceeding_hook(ctx):
            return None

        ctx = mock_tool_context()
        assert_hook_proceeds(proceeding_hook, ctx)

    def test_assert_hook_skips(self):
        """Test assert_hook_skips works correctly."""

        def skipping_hook(ctx):
            return HookResult.skip({"cached": True})

        ctx = mock_tool_context()
        result = assert_hook_skips(skipping_hook, ctx, {"cached": True})
        assert result == {"cached": True}

    def test_assert_hook_modifies(self):
        """Test assert_hook_modifies works correctly."""

        def modifying_hook(ctx):
            return HookResult.modify(kwargs={"timeout": 30})

        ctx = mock_tool_context()
        result = assert_hook_modifies(modifying_hook, ctx, expected_kwargs={"timeout": 30})
        assert result.action == HookAction.MODIFY


class TestRecordingHookRegistry:
    """Tests for RecordingHookRegistry."""

    def test_records_hook_calls(self):
        """Test that hook calls are recorded."""
        registry = RecordingHookRegistry()

        @registry.before_tool
        def my_hook(ctx):
            return HookResult.proceed()

        ctx = mock_tool_context()
        registry.run_before_hooks("before_tool", ctx, [], {})

        assert len(registry.calls) == 1
        assert registry.calls[0].phase == "before_tool"
        assert registry.calls[0].context is ctx

    def test_clear_calls(self):
        """Test clearing recorded calls."""
        registry = RecordingHookRegistry()

        @registry.before_tool
        def my_hook(ctx):
            pass

        ctx = mock_tool_context()
        registry.run_before_hooks("before_tool", ctx, [], {})

        assert len(registry.calls) == 1

        registry.clear_calls()
        assert len(registry.calls) == 0


# =============================================================================
# Integration-style tests
# =============================================================================


class TestHookPatterns:
    """Tests for common hook patterns."""

    def test_security_hook_pattern(self):
        """Test a security hook that blocks dangerous operations."""
        registry = HookRegistry()

        @registry.before_tool
        def security_hook(ctx):
            if ctx.tool_name == "execute_shell" and "--rm" in str(ctx.kwargs):
                return HookResult.block("Dangerous shell command blocked")
            return None

        # Safe call proceeds
        safe_ctx = mock_tool_context(tool_name="execute_shell", kwargs={"cmd": "ls"})
        _, _, result = registry.run_before_hooks("before_tool", safe_ctx, [], {})
        assert result is None

        # Dangerous call blocked
        dangerous_ctx = mock_tool_context(
            tool_name="execute_shell",
            kwargs={"cmd": "rm --rm -rf /"}
        )
        _, _, result = registry.run_before_hooks("before_tool", dangerous_ctx, [], {})
        assert result is not None
        assert result.action == HookAction.BLOCK

    def test_caching_hook_pattern(self):
        """Test a caching hook that skips repeated calls."""
        registry = HookRegistry()
        cache = {}

        @registry.before_tool
        def cache_hook(ctx):
            cache_key = f"{ctx.tool_name}:{ctx.kwargs}"
            if cache_key in cache:
                return HookResult.skip(cache[cache_key])
            return None

        @registry.after_tool
        def store_cache(ctx, result):
            cache_key = f"{ctx.tool_name}:{ctx.kwargs}"
            cache[cache_key] = result
            return result

        # First call - proceeds
        ctx1 = mock_tool_context(tool_name="get_user", kwargs={"id": 1})
        _, _, result1 = registry.run_before_hooks("before_tool", ctx1, [], {})
        assert result1 is None

        # Store result
        registry.run_after_hooks("after_tool", ctx1, {"name": "John"})

        # Second call - skips with cached result
        ctx2 = mock_tool_context(tool_name="get_user", kwargs={"id": 1})
        _, _, result2 = registry.run_before_hooks("before_tool", ctx2, [], {})
        assert result2 is not None
        assert result2.action == HookAction.SKIP
        assert result2.result == {"name": "John"}

    def test_logging_hook_pattern(self):
        """Test a logging hook that observes calls."""
        registry = HookRegistry()
        logs = []

        @registry.before_tool
        def log_before(ctx):
            logs.append(f"BEFORE: {ctx.tool_name}")

        @registry.after_tool
        def log_after(ctx, result):
            logs.append(f"AFTER: {ctx.tool_name} -> {result}")
            return result

        @registry.on_tool_error
        def log_error(ctx, error):
            logs.append(f"ERROR: {ctx.span.span_id}")

        ctx = mock_tool_context(tool_name="my_tool")
        registry.run_before_hooks("before_tool", ctx, [], {})
        registry.run_after_hooks("after_tool", ctx, "success")

        assert logs == ["BEFORE: my_tool", "AFTER: my_tool -> success"]

    def test_pii_redaction_hook_pattern(self):
        """Test a hook that redacts PII from arguments."""
        registry = HookRegistry()

        @registry.before_tool
        def redact_pii(ctx):
            # Check for PII patterns in kwargs
            modified_kwargs = {}
            for key, value in ctx.kwargs.items():
                if "email" in key.lower() and "@" in str(value):
                    modified_kwargs[key] = "[EMAIL_REDACTED]"
                elif "password" in key.lower():
                    modified_kwargs[key] = "[PASSWORD_REDACTED]"

            if modified_kwargs:
                return HookResult.modify(kwargs=modified_kwargs)
            return None

        ctx = mock_tool_context(
            tool_name="create_user",
            kwargs={"email": "user@example.com", "password": "secret123", "name": "John"}
        )
        _, modified_kwargs, _ = registry.run_before_hooks("before_tool", ctx, [], ctx.kwargs)

        assert modified_kwargs["email"] == "[EMAIL_REDACTED]"
        assert modified_kwargs["password"] == "[PASSWORD_REDACTED]"
        assert modified_kwargs["name"] == "John"  # Unchanged
