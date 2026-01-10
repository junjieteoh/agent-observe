"""
Circuit breaker for hook resilience.

Protects production from misbehaving hooks by automatically disabling
hooks that fail repeatedly.

States:
    CLOSED (normal) -> OPEN (disabled) -> HALF-OPEN (testing)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_observe.hooks.registry import RegisteredHook

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Hook disabled due to failures
    HALF_OPEN = "half_open"  # Testing if hook has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    enabled: bool = True
    failure_threshold: int = 5  # Failures before tripping
    window_seconds: int = 60  # Time window for counting failures
    recovery_seconds: int = 300  # Time before retry (OPEN -> HALF_OPEN)
    action: str = "disable"  # "disable" | "log_only" | "raise"


@dataclass
class HookCircuitBreaker:
    """
    Circuit breaker for a single hook.

    Tracks failures within a time window and trips when threshold is exceeded.
    After recovery_seconds, allows one test call to check if hook has recovered.
    """

    hook_name: str
    config: CircuitBreakerConfig
    state: CircuitState = CircuitState.CLOSED

    # Failure tracking
    _failures: list[float] = field(default_factory=list)  # Timestamps of failures
    _last_failure_time: float = 0.0
    _opened_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_success(self) -> None:
        """Record a successful hook execution."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                # Successful test call - close the circuit
                self.state = CircuitState.CLOSED
                self._failures.clear()
                logger.info(f"Circuit breaker for hook '{self.hook_name}' closed (recovered)")

    def record_failure(self, error: Exception) -> None:
        """Record a failed hook execution."""
        with self._lock:
            now = time.time()
            self._last_failure_time = now

            if self.state == CircuitState.HALF_OPEN:
                # Test call failed - reopen circuit
                self.state = CircuitState.OPEN
                self._opened_at = now
                logger.warning(
                    f"Circuit breaker for hook '{self.hook_name}' reopened "
                    f"(test call failed: {error})"
                )
                return

            # Prune old failures outside the window
            window_start = now - self.config.window_seconds
            self._failures = [t for t in self._failures if t > window_start]

            # Add new failure
            self._failures.append(now)

            # Check if we should trip
            if len(self._failures) >= self.config.failure_threshold:
                self._trip(error)

    def should_allow(self) -> bool:
        """Check if the hook should be allowed to run."""
        if not self.config.enabled:
            return True  # Circuit breaker disabled

        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if recovery period has passed
                now = time.time()
                if now - self._opened_at >= self.config.recovery_seconds:
                    # Move to half-open state
                    self.state = CircuitState.HALF_OPEN
                    logger.info(
                        f"Circuit breaker for hook '{self.hook_name}' "
                        f"entering half-open state (testing)"
                    )
                    return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                # Allow one test call
                return True

            return True

    def _trip(self, error: Exception) -> None:
        """Trip the circuit breaker."""
        self.state = CircuitState.OPEN
        self._opened_at = time.time()

        logger.warning(
            f"Circuit breaker for hook '{self.hook_name}' tripped: "
            f"{len(self._failures)} failures in {self.config.window_seconds}s "
            f"(last error: {error})"
        )

    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        with self._lock:
            return {
                "state": self.state.value,
                "failures_in_window": len(self._failures),
                "last_failure_time": self._last_failure_time,
                "opened_at": self._opened_at if self.state != CircuitState.CLOSED else None,
            }


class CircuitBreakerRegistry:
    """
    Manages circuit breakers for all hooks.

    Usage:
        registry = CircuitBreakerRegistry(config)

        # Before running hook
        if registry.should_allow(hook_name):
            try:
                result = run_hook(...)
                registry.record_success(hook_name)
            except Exception as e:
                registry.record_failure(hook_name, e)
                raise
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        """Initialize the circuit breaker registry."""
        self.config = config or CircuitBreakerConfig()
        self._breakers: dict[str, HookCircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_or_create(self, hook_name: str) -> HookCircuitBreaker:
        """Get or create a circuit breaker for a hook."""
        with self._lock:
            if hook_name not in self._breakers:
                self._breakers[hook_name] = HookCircuitBreaker(
                    hook_name=hook_name,
                    config=self.config,
                )
            return self._breakers[hook_name]

    def should_allow(self, hook_name: str) -> bool:
        """Check if a hook should be allowed to run."""
        breaker = self.get_or_create(hook_name)
        return breaker.should_allow()

    def record_success(self, hook_name: str) -> None:
        """Record successful hook execution."""
        breaker = self.get_or_create(hook_name)
        breaker.record_success()

    def record_failure(self, hook_name: str, error: Exception) -> None:
        """Record failed hook execution."""
        breaker = self.get_or_create(hook_name)
        breaker.record_failure(error)

    def get_status(self) -> dict[str, dict]:
        """Get status of all circuit breakers."""
        with self._lock:
            return {name: breaker.get_state() for name, breaker in self._breakers.items()}

    def reset(self, hook_name: str | None = None) -> None:
        """Reset circuit breaker(s) to closed state."""
        with self._lock:
            if hook_name is None:
                for breaker in self._breakers.values():
                    breaker.state = CircuitState.CLOSED
                    breaker._failures.clear()
            elif hook_name in self._breakers:
                self._breakers[hook_name].state = CircuitState.CLOSED
                self._breakers[hook_name]._failures.clear()
