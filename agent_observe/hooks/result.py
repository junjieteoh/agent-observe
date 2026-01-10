"""
Hook result types for controlling execution flow.

HookAction defines what action to take after a hook runs.
HookResult is returned by before hooks to control execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HookAction(Enum):
    """Actions that a hook can request."""

    PROCEED = "proceed"  # Continue normal execution
    BLOCK = "block"  # Raise exception, stop execution
    SKIP = "skip"  # Skip execution, use provided result
    MODIFY = "modify"  # Use modified args/kwargs
    PENDING = "pending"  # Wait for external approval


@dataclass
class HookResult:
    """
    Result from a before hook that controls execution flow.

    Before hooks can return None (proceed) or a HookResult to control
    what happens next.

    Examples:
        # Proceed normally (same as returning None)
        return HookResult.proceed()

        # Block execution with reason
        return HookResult.block("Dangerous operation not allowed")

        # Skip execution and return a cached result
        return HookResult.skip(cached_value)

        # Modify the arguments before execution
        return HookResult.modify(kwargs={"timeout": 30})

        # Wait for external approval
        return HookResult.pending(timeout_seconds=300)
    """

    action: HookAction = HookAction.PROCEED

    # For MODIFY action
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None

    # For SKIP action
    result: Any = None

    # For BLOCK action
    reason: str | None = None

    # For PENDING action
    timeout_seconds: int = 300
    on_timeout: str = "block"  # "block" or "proceed"
    approval_id: str | None = None  # For external approval systems

    # Metadata
    hook_name: str | None = field(default=None, repr=False)

    @classmethod
    def proceed(cls) -> HookResult:
        """Continue with normal execution."""
        return cls(action=HookAction.PROCEED)

    @classmethod
    def block(cls, reason: str) -> HookResult:
        """
        Block execution and raise an exception.

        Args:
            reason: Human-readable reason for blocking.

        Returns:
            HookResult with BLOCK action.
        """
        return cls(action=HookAction.BLOCK, reason=reason)

    @classmethod
    def skip(cls, result: Any) -> HookResult:
        """
        Skip execution and return the provided result instead.

        Useful for caching or mocking tool calls.

        Args:
            result: The result to return instead of executing.

        Returns:
            HookResult with SKIP action.
        """
        return cls(action=HookAction.SKIP, result=result)

    @classmethod
    def modify(
        cls,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> HookResult:
        """
        Modify the arguments before execution.

        Args:
            args: New positional arguments (or None to keep original).
            kwargs: New/updated keyword arguments (merged with original).

        Returns:
            HookResult with MODIFY action.
        """
        return cls(action=HookAction.MODIFY, args=args, kwargs=kwargs)

    @classmethod
    def pending(
        cls,
        timeout_seconds: int = 300,
        on_timeout: str = "block",
        approval_id: str | None = None,
    ) -> HookResult:
        """
        Pause execution and wait for external approval.

        Args:
            timeout_seconds: How long to wait before timing out.
            on_timeout: What to do on timeout - "block" or "proceed".
            approval_id: Optional ID for external approval tracking.

        Returns:
            HookResult with PENDING action.
        """
        return cls(
            action=HookAction.PENDING,
            timeout_seconds=timeout_seconds,
            on_timeout=on_timeout,
            approval_id=approval_id,
        )

    def is_blocking(self) -> bool:
        """Check if this result will prevent execution."""
        return self.action in (HookAction.BLOCK, HookAction.SKIP)
