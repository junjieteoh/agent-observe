"""Tests for replay cache."""

from __future__ import annotations

import pytest

from agent_observe.config import CaptureMode, ReplayMode
from agent_observe.replay import ReplayCache
from agent_observe.sinks.sqlite_sink import SQLiteSink


class TestReplayCache:
    """Tests for ReplayCache."""

    def test_make_key(self, sqlite_sink: SQLiteSink) -> None:
        """Test cache key generation."""
        cache = ReplayCache(sqlite_sink, ReplayMode.WRITE, CaptureMode.METADATA_ONLY)

        key = cache.make_key("tool_name", {"arg": "value"}, "1.0")

        assert key.startswith("tool_name:")
        assert ":1.0" in key

    def test_make_key_deterministic(self, sqlite_sink: SQLiteSink) -> None:
        """Test that key generation is deterministic."""
        cache = ReplayCache(sqlite_sink, ReplayMode.WRITE, CaptureMode.METADATA_ONLY)

        key1 = cache.make_key("tool", {"a": 1, "b": 2}, "1")
        key2 = cache.make_key("tool", {"b": 2, "a": 1}, "1")

        assert key1 == key2

    def test_write_and_read_full_mode(self, sqlite_sink: SQLiteSink) -> None:
        """Test writing and reading cache in full mode."""
        # Write
        write_cache = ReplayCache(sqlite_sink, ReplayMode.WRITE, CaptureMode.FULL)
        write_cache.put(
            tool_name="my_tool",
            args={"input": "test"},
            result={"output": "result"},
            tool_version="1",
            status="ok",
        )

        # Read
        read_cache = ReplayCache(sqlite_sink, ReplayMode.READ, CaptureMode.FULL)
        key = read_cache.make_key("my_tool", {"input": "test"}, "1")
        entry = read_cache.get(key)

        assert entry is not None
        assert entry.tool_name == "my_tool"
        assert entry.status == "ok"
        assert entry.result == {"output": "result"}

    def test_metadata_only_stores_hash(self, sqlite_sink: SQLiteSink) -> None:
        """Test that metadata_only mode stores hash but not result."""
        cache = ReplayCache(sqlite_sink, ReplayMode.WRITE, CaptureMode.METADATA_ONLY)
        cache.put(
            tool_name="my_tool",
            args={"input": "test"},
            result={"output": "result"},
            tool_version="1",
        )

        # Read back
        read_cache = ReplayCache(sqlite_sink, ReplayMode.READ, CaptureMode.METADATA_ONLY)
        key = read_cache.make_key("my_tool", {"input": "test"}, "1")
        entry = read_cache.get(key)

        assert entry is not None
        assert entry.result is None  # No result stored
        assert entry.result_hash is not None  # But hash is stored

    def test_get_returns_none_when_off(self, sqlite_sink: SQLiteSink) -> None:
        """Test that get returns None when mode is OFF."""
        cache = ReplayCache(sqlite_sink, ReplayMode.OFF, CaptureMode.FULL)

        # Even if entry exists, OFF mode returns None
        key = cache.make_key("my_tool", {"input": "test"}, "1")
        entry = cache.get(key)

        assert entry is None

    def test_put_does_nothing_when_read_mode(self, sqlite_sink: SQLiteSink) -> None:
        """Test that put is no-op in READ mode."""
        cache = ReplayCache(sqlite_sink, ReplayMode.READ, CaptureMode.FULL)

        # This should not write anything
        cache.put(
            tool_name="my_tool",
            args={"input": "test"},
            result={"output": "result"},
        )

        # Switch to write mode to verify nothing was written
        write_cache = ReplayCache(sqlite_sink, ReplayMode.WRITE, CaptureMode.FULL)
        key = write_cache.make_key("my_tool", {"input": "test"}, "1")
        entry = sqlite_sink.get_replay_cache_entry(key)

        assert entry is None

    def test_execute_with_cache_hit(self, sqlite_sink: SQLiteSink) -> None:
        """Test execute_with_cache returns cached result."""
        # First, write to cache
        write_cache = ReplayCache(sqlite_sink, ReplayMode.WRITE, CaptureMode.FULL)
        write_cache.put(
            tool_name="expensive_tool",
            args={"x": 1},
            result="cached_result",
        )

        # Now execute with cache
        read_cache = ReplayCache(sqlite_sink, ReplayMode.READ, CaptureMode.FULL)

        call_count = 0

        def expensive_fn() -> str:
            nonlocal call_count
            call_count += 1
            return "new_result"

        result, was_cached = read_cache.execute_with_cache(
            tool_name="expensive_tool",
            args={"x": 1},
            fn=expensive_fn,
        )

        assert result == "cached_result"
        assert was_cached is True
        assert call_count == 0  # Function was not called

    def test_execute_with_cache_miss(self, sqlite_sink: SQLiteSink) -> None:
        """Test execute_with_cache calls function on miss."""
        cache = ReplayCache(sqlite_sink, ReplayMode.READ, CaptureMode.FULL)

        call_count = 0

        def my_fn() -> str:
            nonlocal call_count
            call_count += 1
            return "computed_result"

        result, was_cached = cache.execute_with_cache(
            tool_name="new_tool",
            args={"y": 2},
            fn=my_fn,
        )

        assert result == "computed_result"
        assert was_cached is False
        assert call_count == 1

    def test_execute_with_cache_writes_on_write_mode(self, sqlite_sink: SQLiteSink) -> None:
        """Test execute_with_cache writes result in WRITE mode."""
        cache = ReplayCache(sqlite_sink, ReplayMode.WRITE, CaptureMode.FULL)

        def my_fn() -> str:
            return "result_to_cache"

        result, was_cached = cache.execute_with_cache(
            tool_name="tool_to_cache",
            args={"z": 3},
            fn=my_fn,
        )

        assert result == "result_to_cache"
        assert was_cached is False

        # Verify it was written
        key = cache.make_key("tool_to_cache", {"z": 3}, "1")
        entry = sqlite_sink.get_replay_cache_entry(key)

        assert entry is not None
        assert entry["status"] == "ok"

    def test_execute_with_cache_handles_exception(self, sqlite_sink: SQLiteSink) -> None:
        """Test execute_with_cache handles exceptions."""
        cache = ReplayCache(sqlite_sink, ReplayMode.WRITE, CaptureMode.FULL)

        def failing_fn() -> str:
            raise ValueError("Tool failed")

        with pytest.raises(ValueError, match="Tool failed"):
            cache.execute_with_cache(
                tool_name="failing_tool",
                args={"fail": True},
                fn=failing_fn,
            )

        # Verify error was cached
        key = cache.make_key("failing_tool", {"fail": True}, "1")
        entry = sqlite_sink.get_replay_cache_entry(key)

        assert entry is not None
        assert entry["status"] == "error"
