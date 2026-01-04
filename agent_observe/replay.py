"""
Tool replay system for agent-observe.

Provides deterministic tool result caching for:
- Testing with consistent tool outputs
- Debugging with recorded behavior
- Cost reduction by avoiding redundant API calls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from agent_observe.config import CaptureMode, ReplayMode
from agent_observe.context import now_ms
from agent_observe.hashing import hash_json

if TYPE_CHECKING:
    from agent_observe.sinks.base import Sink

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ReplayCacheEntry:
    """Entry in the replay cache."""

    key: str
    tool_name: str
    args_hash: str
    tool_version: str
    created_ts: int
    status: str  # "ok" or "error"
    result: Any | None = None
    result_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "tool_name": self.tool_name,
            "args_hash": self.args_hash,
            "tool_version": self.tool_version,
            "created_ts": self.created_ts,
            "status": self.status,
            "result": self.result,
            "result_hash": self.result_hash,
        }


class ReplayCache:
    """
    Manages tool result caching for replay.

    Cache key format: {tool_name}:{args_hash}:{tool_version}
    """

    def __init__(
        self,
        sink: Sink,
        mode: ReplayMode = ReplayMode.OFF,
        capture_mode: CaptureMode = CaptureMode.METADATA_ONLY,
    ):
        """
        Initialize replay cache.

        Args:
            sink: Storage sink for cache entries.
            mode: Replay mode (off, write, read).
            capture_mode: Controls what data is stored.
        """
        self.sink = sink
        self.mode = mode
        self.capture_mode = capture_mode

    def make_key(self, tool_name: str, args: Any, tool_version: str = "1") -> str:
        """
        Create cache key for a tool call.

        Args:
            tool_name: Name of the tool.
            args: Tool arguments.
            tool_version: Tool version.

        Returns:
            Cache key string.
        """
        args_hash = hash_json(args)
        return f"{tool_name}:{args_hash}:{tool_version}"

    def get(self, key: str) -> ReplayCacheEntry | None:
        """
        Get cached result for a tool call.

        Args:
            key: Cache key.

        Returns:
            ReplayCacheEntry if found, None otherwise.
        """
        if self.mode != ReplayMode.READ:
            return None

        entry_dict = self.sink.get_replay_cache_entry(key)
        if entry_dict is None:
            return None

        return ReplayCacheEntry(
            key=entry_dict["key"],
            tool_name=entry_dict["tool_name"],
            args_hash=entry_dict["args_hash"],
            tool_version=entry_dict.get("tool_version", "1"),
            created_ts=entry_dict["created_ts"],
            status=entry_dict["status"],
            result=entry_dict.get("result"),
            result_hash=entry_dict.get("result_hash"),
        )

    def put(
        self,
        tool_name: str,
        args: Any,
        result: Any,
        tool_version: str = "1",
        status: str = "ok",
    ) -> None:
        """
        Store result in cache.

        Args:
            tool_name: Name of the tool.
            args: Tool arguments.
            result: Tool result.
            tool_version: Tool version.
            status: Result status ("ok" or "error").
        """
        if self.mode != ReplayMode.WRITE:
            return

        args_hash = hash_json(args)
        key = f"{tool_name}:{args_hash}:{tool_version}"
        result_hash = hash_json(result)

        # Determine what to store based on capture mode
        stored_result: Any | None = None
        if self.capture_mode in (CaptureMode.EVIDENCE_ONLY, CaptureMode.FULL):
            stored_result = result

        entry = ReplayCacheEntry(
            key=key,
            tool_name=tool_name,
            args_hash=args_hash,
            tool_version=tool_version,
            created_ts=now_ms(),
            status=status,
            result=stored_result,
            result_hash=result_hash,
        )

        self.sink.write_replay_cache(entry.to_dict())

    def execute_with_cache(
        self,
        tool_name: str,
        args: Any,
        fn: Callable[[], T],
        tool_version: str = "1",
    ) -> tuple[T, bool]:
        """
        Execute a tool function with caching.

        Args:
            tool_name: Name of the tool.
            args: Tool arguments.
            fn: Function to execute if not cached.
            tool_version: Tool version.

        Returns:
            Tuple of (result, was_cached).
        """
        key = self.make_key(tool_name, args, tool_version)

        # Try to get from cache (only in read mode)
        if self.mode == ReplayMode.READ:
            entry = self.get(key)
            if entry is not None:
                if entry.result is not None:
                    logger.debug(f"Replay cache hit for {tool_name}")
                    return entry.result, True
                else:
                    # Metadata-only entry - no result stored
                    logger.warning(
                        f"Replay cache entry for {tool_name} has no result "
                        "(stored in metadata_only mode)"
                    )

        # Execute the function
        try:
            result = fn()
            status = "ok"
        except Exception as e:
            # Store error in cache
            if self.mode == ReplayMode.WRITE:
                self.put(tool_name, args, {"error": str(e)}, tool_version, status="error")
            raise

        # Store in cache (only in write mode)
        if self.mode == ReplayMode.WRITE:
            self.put(tool_name, args, result, tool_version, status=status)

        return result, False
