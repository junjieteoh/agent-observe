"""Tests for SQLite sink."""

from __future__ import annotations

from pathlib import Path

from agent_observe.context import now_ms
from agent_observe.sinks.sqlite_sink import SQLiteSink


class TestSQLiteSink:
    """Tests for SQLiteSink."""

    def test_initialize_creates_tables(self, temp_db: Path) -> None:
        """Test that initialization creates required tables."""
        sink = SQLiteSink(path=temp_db, async_writes=False)
        sink.initialize()

        # Check tables exist
        conn = sink._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}

        assert "runs" in tables
        assert "spans" in tables
        assert "events" in tables
        assert "replay_cache" in tables
        assert "agent_observe_schema_version" in tables

        sink.close()

    def test_write_and_read_run(self, sqlite_sink: SQLiteSink) -> None:
        """Test writing and reading a run."""
        run_data = {
            "run_id": "test-run-123",
            "trace_id": "trace-456",
            "name": "test-agent",
            "ts_start": now_ms(),
            "ts_end": now_ms() + 1000,
            "status": "ok",
            "risk_score": 25,
            "eval_tags": ["RETRY_STORM"],
            "tool_calls": 5,
            "model_calls": 2,
            "policy_violations": 0,
            "latency_ms": 1000,
            "project": "test-project",
            "env": "dev",
            "agent_version": "1.0.0",
        }

        sqlite_sink.write_run(run_data)

        # Read back
        run = sqlite_sink.get_run("test-run-123")

        assert run is not None
        assert run["run_id"] == "test-run-123"
        assert run["name"] == "test-agent"
        assert run["status"] == "ok"
        assert run["risk_score"] == 25
        assert run["eval_tags"] == ["RETRY_STORM"]

    def test_write_and_read_span(self, sqlite_sink: SQLiteSink) -> None:
        """Test writing and reading spans."""
        # First create a run
        sqlite_sink.write_run({
            "run_id": "run-1",
            "name": "test",
            "ts_start": now_ms(),
            "status": "ok",
        })

        span_data = {
            "span_id": "span-1",
            "run_id": "run-1",
            "parent_span_id": None,
            "kind": "tool",
            "name": "my_tool",
            "ts_start": now_ms(),
            "ts_end": now_ms() + 100,
            "status": "ok",
            "attrs": {"key": "value"},
        }

        sqlite_sink.write_span(span_data)

        # Read back
        spans = sqlite_sink.get_spans("run-1")

        assert len(spans) == 1
        assert spans[0]["span_id"] == "span-1"
        assert spans[0]["name"] == "my_tool"
        assert spans[0]["attrs"] == {"key": "value"}

    def test_write_and_read_event(self, sqlite_sink: SQLiteSink) -> None:
        """Test writing and reading events."""
        sqlite_sink.write_run({
            "run_id": "run-1",
            "name": "test",
            "ts_start": now_ms(),
            "status": "ok",
        })

        event_data = {
            "event_id": "event-1",
            "run_id": "run-1",
            "ts": now_ms(),
            "type": "custom.event",
            "payload": {"data": [1, 2, 3]},
        }

        sqlite_sink.write_event(event_data)

        # Read back
        events = sqlite_sink.get_events("run-1")

        assert len(events) == 1
        assert events[0]["type"] == "custom.event"
        assert events[0]["payload"] == {"data": [1, 2, 3]}

    def test_get_runs_with_filters(self, sqlite_sink: SQLiteSink) -> None:
        """Test querying runs with filters."""
        # Create multiple runs
        for i in range(5):
            sqlite_sink.write_run({
                "run_id": f"run-{i}",
                "name": f"agent-{'alpha' if i < 3 else 'beta'}",
                "ts_start": now_ms() - i * 1000,
                "status": "ok" if i % 2 == 0 else "error",
                "risk_score": i * 10,
            })

        # Filter by name
        runs = sqlite_sink.get_runs(name="alpha")
        assert len(runs) == 3

        # Filter by status
        runs = sqlite_sink.get_runs(status="error")
        assert len(runs) == 2

        # Filter by min_risk
        runs = sqlite_sink.get_runs(min_risk=20)
        assert len(runs) == 3

        # Combine filters
        runs = sqlite_sink.get_runs(name="alpha", status="ok")
        assert len(runs) == 2

    def test_get_runs_pagination(self, sqlite_sink: SQLiteSink) -> None:
        """Test run pagination."""
        for i in range(20):
            sqlite_sink.write_run({
                "run_id": f"run-{i:02d}",
                "name": "test",
                "ts_start": now_ms() - i * 1000,
                "status": "ok",
            })

        # First page
        runs = sqlite_sink.get_runs(limit=5, offset=0)
        assert len(runs) == 5

        # Second page
        runs = sqlite_sink.get_runs(limit=5, offset=5)
        assert len(runs) == 5

        # Beyond data
        runs = sqlite_sink.get_runs(limit=5, offset=25)
        assert len(runs) == 0

    def test_replay_cache(self, sqlite_sink: SQLiteSink) -> None:
        """Test replay cache operations."""
        entry = {
            "key": "tool:abc:1",
            "tool_name": "tool",
            "args_hash": "abc",
            "tool_version": "1",
            "created_ts": now_ms(),
            "status": "ok",
            "result": {"data": "cached"},
            "result_hash": "xyz",
        }

        sqlite_sink.write_replay_cache(entry)

        # Read back
        cached = sqlite_sink.get_replay_cache_entry("tool:abc:1")

        assert cached is not None
        assert cached["tool_name"] == "tool"
        assert cached["status"] == "ok"
        assert cached["result"] == {"data": "cached"}

    def test_run_not_found(self, sqlite_sink: SQLiteSink) -> None:
        """Test getting non-existent run."""
        run = sqlite_sink.get_run("nonexistent")
        assert run is None

    def test_cascade_delete(self, sqlite_sink: SQLiteSink) -> None:
        """Test that deleting run cascades to spans and events."""
        # Create run with spans and events
        sqlite_sink.write_run({
            "run_id": "run-to-delete",
            "name": "test",
            "ts_start": now_ms(),
            "status": "ok",
        })

        sqlite_sink.write_span({
            "span_id": "span-1",
            "run_id": "run-to-delete",
            "kind": "tool",
            "name": "tool",
            "ts_start": now_ms(),
            "status": "ok",
        })

        sqlite_sink.write_event({
            "event_id": "event-1",
            "run_id": "run-to-delete",
            "ts": now_ms(),
            "type": "test",
            "payload": {},
        })

        # Verify they exist
        assert len(sqlite_sink.get_spans("run-to-delete")) == 1
        assert len(sqlite_sink.get_events("run-to-delete")) == 1

        # Delete the run
        conn = sqlite_sink._get_connection()
        conn.execute("DELETE FROM runs WHERE run_id = ?", ("run-to-delete",))
        conn.commit()

        # Verify cascade
        assert len(sqlite_sink.get_spans("run-to-delete")) == 0
        assert len(sqlite_sink.get_events("run-to-delete")) == 0

    def test_thread_safety(self, temp_db: Path) -> None:
        """Test that sink handles multiple threads."""
        import threading

        sink = SQLiteSink(path=temp_db, async_writes=False)
        sink.initialize()

        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(10):
                    sink.write_run({
                        "run_id": f"run-{thread_id}-{i}",
                        "name": f"thread-{thread_id}",
                        "ts_start": now_ms(),
                        "status": "ok",
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify all runs were written
        runs = sink.get_runs(limit=100)
        assert len(runs) == 50

        sink.close()
