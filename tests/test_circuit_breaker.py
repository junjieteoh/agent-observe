"""
Tests for circuit breaker functionality.
"""

import time

import pytest

from agent_observe.hooks import (
    CircuitBreakerConfig,
    CircuitState,
    HookCircuitBreaker,
    HookRegistry,
    mock_tool_context,
)
from agent_observe.hooks.circuit_breaker import CircuitBreakerRegistry


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.enabled is True
        assert config.failure_threshold == 5
        assert config.window_seconds == 60
        assert config.recovery_seconds == 300
        assert config.action == "disable"


class TestHookCircuitBreaker:
    """Tests for individual hook circuit breaker."""

    def test_initial_state_is_closed(self):
        """Test that circuit starts in closed state."""
        config = CircuitBreakerConfig()
        breaker = HookCircuitBreaker(hook_name="test_hook", config=config)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.should_allow() is True

    def test_success_in_closed_state(self):
        """Test that success in closed state keeps it closed."""
        config = CircuitBreakerConfig()
        breaker = HookCircuitBreaker(hook_name="test_hook", config=config)

        breaker.record_success()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.should_allow() is True

    def test_single_failure_does_not_trip(self):
        """Test that a single failure doesn't trip the circuit."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = HookCircuitBreaker(hook_name="test_hook", config=config)

        breaker.record_failure(ValueError("Test error"))

        assert breaker.state == CircuitState.CLOSED
        assert breaker.should_allow() is True

    def test_reaching_threshold_trips_circuit(self):
        """Test that reaching failure threshold trips the circuit."""
        config = CircuitBreakerConfig(failure_threshold=3, window_seconds=60)
        breaker = HookCircuitBreaker(hook_name="test_hook", config=config)

        # Record failures up to threshold
        for i in range(3):
            breaker.record_failure(ValueError(f"Error {i}"))

        assert breaker.state == CircuitState.OPEN
        assert breaker.should_allow() is False

    def test_failures_outside_window_ignored(self):
        """Test that failures outside the time window are pruned."""
        config = CircuitBreakerConfig(failure_threshold=3, window_seconds=1)
        breaker = HookCircuitBreaker(hook_name="test_hook", config=config)

        # Record some failures
        breaker.record_failure(ValueError("Error 1"))
        breaker.record_failure(ValueError("Error 2"))

        # Wait for window to pass
        time.sleep(1.1)

        # These should be in a new window
        breaker.record_failure(ValueError("Error 3"))

        # Should still be closed (only 1 failure in current window)
        assert breaker.state == CircuitState.CLOSED

    def test_recovery_to_half_open(self):
        """Test that circuit enters half-open state after recovery period."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_seconds=1,  # Short for testing
        )
        breaker = HookCircuitBreaker(hook_name="test_hook", config=config)

        # Trip the circuit
        breaker.record_failure(ValueError("Error 1"))
        breaker.record_failure(ValueError("Error 2"))
        assert breaker.state == CircuitState.OPEN
        assert breaker.should_allow() is False

        # Wait for recovery
        time.sleep(1.1)

        # Should transition to half-open on next check
        assert breaker.should_allow() is True
        assert breaker.state == CircuitState.HALF_OPEN

    def test_success_in_half_open_closes_circuit(self):
        """Test that success in half-open state closes the circuit."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_seconds=0)
        breaker = HookCircuitBreaker(hook_name="test_hook", config=config)

        # Trip and recover to half-open
        breaker.record_failure(ValueError("Error"))
        breaker.should_allow()  # Triggers transition to half-open
        assert breaker.state == CircuitState.HALF_OPEN

        # Success should close
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens_circuit(self):
        """Test that failure in half-open state reopens the circuit."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_seconds=0)
        breaker = HookCircuitBreaker(hook_name="test_hook", config=config)

        # Trip and recover to half-open
        breaker.record_failure(ValueError("Error"))
        breaker.should_allow()  # Triggers transition to half-open
        assert breaker.state == CircuitState.HALF_OPEN

        # Failure should reopen
        breaker.record_failure(ValueError("Error again"))
        assert breaker.state == CircuitState.OPEN

    def test_get_state(self):
        """Test getting circuit breaker state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = HookCircuitBreaker(hook_name="test_hook", config=config)

        breaker.record_failure(ValueError("Error"))
        state = breaker.get_state()

        assert state["state"] == "closed"
        assert state["failures_in_window"] == 1


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_creates_breakers_on_demand(self):
        """Test that breakers are created when first accessed."""
        registry = CircuitBreakerRegistry(CircuitBreakerConfig())

        # Access should create
        assert registry.should_allow("hook1") is True

        status = registry.get_status()
        assert "hook1" in status

    def test_tracks_multiple_hooks_independently(self):
        """Test that each hook has its own circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2)
        registry = CircuitBreakerRegistry(config)

        # Trip hook1
        registry.record_failure("hook1", ValueError("Error 1"))
        registry.record_failure("hook1", ValueError("Error 2"))

        # hook1 should be open, hook2 should be closed
        assert registry.should_allow("hook1") is False
        assert registry.should_allow("hook2") is True

    def test_reset_specific_hook(self):
        """Test resetting a specific hook's circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        registry = CircuitBreakerRegistry(config)

        # Trip both hooks
        registry.record_failure("hook1", ValueError("Error"))
        registry.record_failure("hook2", ValueError("Error"))

        # Reset only hook1
        registry.reset("hook1")

        assert registry.should_allow("hook1") is True
        assert registry.should_allow("hook2") is False

    def test_reset_all_hooks(self):
        """Test resetting all circuit breakers."""
        config = CircuitBreakerConfig(failure_threshold=1)
        registry = CircuitBreakerRegistry(config)

        # Trip hooks
        registry.record_failure("hook1", ValueError("Error"))
        registry.record_failure("hook2", ValueError("Error"))

        # Reset all
        registry.reset()

        assert registry.should_allow("hook1") is True
        assert registry.should_allow("hook2") is True


class TestCircuitBreakerWithHookRegistry:
    """Integration tests for circuit breaker with HookRegistry."""

    def test_circuit_breaker_skips_failing_hooks(self):
        """Test that circuit breaker skips hooks that fail repeatedly."""
        config = CircuitBreakerConfig(failure_threshold=2)
        registry = HookRegistry(circuit_breaker=config)
        call_count = 0

        @registry.before_tool
        def failing_hook(ctx):
            nonlocal call_count
            call_count += 1
            raise ValueError("Hook error!")

        ctx = mock_tool_context()

        # First two calls should run (and fail)
        registry.run_before_hooks("before_tool", ctx, [], {})
        registry.run_before_hooks("before_tool", ctx, [], {})
        assert call_count == 2

        # Third call should be skipped (circuit open)
        registry.run_before_hooks("before_tool", ctx, [], {})
        assert call_count == 2  # Still 2, hook was skipped

    def test_disabled_circuit_breaker(self):
        """Test that disabled circuit breaker doesn't affect hooks."""
        config = CircuitBreakerConfig(enabled=False)
        registry = HookRegistry(circuit_breaker=config)
        call_count = 0

        @registry.before_tool
        def always_runs(ctx):
            nonlocal call_count
            call_count += 1

        ctx = mock_tool_context()

        for _ in range(5):
            registry.run_before_hooks("before_tool", ctx, [], {})

        assert call_count == 5
