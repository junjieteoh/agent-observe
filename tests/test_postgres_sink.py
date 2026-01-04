"""Tests for PostgreSQL sink.

These are integration tests that require a running PostgreSQL database.
They are skipped if DATABASE_URL is not set.
"""

from __future__ import annotations

import os

import pytest

# Skip all tests in this module if DATABASE_URL is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set - skipping Postgres integration tests",
)


@pytest.fixture
def postgres_sink():
    """Create a PostgreSQL sink for testing."""
    from agent_observe.sinks.postgres_sink import PostgresSink

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set")

    sink = PostgresSink(
        database_url=database_url,
        async_writes=False,
    )
    sink.initialize()

    # Clean up test data before test
    if sink._pool:
        with sink._pool.connection() as conn:
            conn.execute("DELETE FROM events WHERE run_id LIKE 'test-%'")
            conn.execute("DELETE FROM spans WHERE run_id LIKE 'test-%'")
            conn.execute("DELETE FROM runs WHERE run_id LIKE 'test-%'")
            conn.commit()

    yield sink

    # Clean up after test
    if sink._pool:
        with sink._pool.connection() as conn:
            conn.execute("DELETE FROM events WHERE run_id LIKE 'test-%'")
            conn.execute("DELETE FROM spans WHERE run_id LIKE 'test-%'")
            conn.execute("DELETE FROM runs WHERE run_id LIKE 'test-%'")
            conn.commit()

    sink.close()


@pytest.mark.integration
class TestPostgresSink:
    """Integration tests for PostgresSink."""

    def test_write_and_read_run(self, postgres_sink) -> None:
        """Test writing and reading a run."""
        import time

        ts = int(time.time() * 1000)

        run_data = {
            "run_id": "test-pg-run-1",
            "trace_id": "trace-pg-1",
            "name": "test-postgres-agent",
            "ts_start": ts,
            "ts_end": ts + 1000,
            "status": "ok",
            "risk_score": 15,
            "eval_tags": ["TEST_TAG"],
            "tool_calls": 3,
            "model_calls": 1,
            "policy_violations": 0,
            "latency_ms": 1000,
            "project": "test-project",
            "env": "test",
            "agent_version": "1.0.0",
        }

        postgres_sink._do_write_runs([run_data])

        # Read back
        run = postgres_sink.get_run("test-pg-run-1")

        assert run is not None
        assert run["name"] == "test-postgres-agent"
        assert run["status"] == "ok"
        assert run["risk_score"] == 15

    def test_write_and_read_span(self, postgres_sink) -> None:
        """Test writing and reading spans."""
        import time

        ts = int(time.time() * 1000)

        # Create run first
        postgres_sink._do_write_runs([{
            "run_id": "test-pg-run-2",
            "name": "test",
            "ts_start": ts,
            "status": "ok",
        }])

        # Create span
        postgres_sink._do_write_spans([{
            "span_id": "test-span-1",
            "run_id": "test-pg-run-2",
            "kind": "tool",
            "name": "test_tool",
            "ts_start": ts,
            "ts_end": ts + 100,
            "status": "ok",
            "attrs": {"key": "value"},
        }])

        # Read back
        spans = postgres_sink.get_spans("test-pg-run-2")

        assert len(spans) == 1
        assert spans[0]["name"] == "test_tool"

    def test_get_runs_with_filters(self, postgres_sink) -> None:
        """Test querying runs with filters."""
        import time

        ts = int(time.time() * 1000)

        # Create multiple runs
        runs = []
        for i in range(5):
            runs.append({
                "run_id": f"test-pg-filter-{i}",
                "name": f"agent-{'alpha' if i < 3 else 'beta'}",
                "ts_start": ts - i * 1000,
                "status": "ok" if i % 2 == 0 else "error",
                "risk_score": i * 10,
            })

        postgres_sink._do_write_runs(runs)

        # Filter by name
        result = postgres_sink.get_runs(name="alpha")
        assert len(result) >= 3

        # Filter by status
        result = postgres_sink.get_runs(status="error")
        assert len([r for r in result if r["run_id"].startswith("test-pg-filter")]) >= 2

    def test_replay_cache(self, postgres_sink) -> None:
        """Test replay cache operations."""
        import time

        ts = int(time.time() * 1000)

        entry = {
            "key": "test-pg-tool:abc:1",
            "tool_name": "test-pg-tool",
            "args_hash": "abc",
            "tool_version": "1",
            "created_ts": ts,
            "status": "ok",
            "result": {"data": "cached"},
            "result_hash": "xyz",
        }

        postgres_sink._do_write_replay_cache([entry])

        # Read back
        cached = postgres_sink.get_replay_cache_entry("test-pg-tool:abc:1")

        assert cached is not None
        assert cached["tool_name"] == "test-pg-tool"
        assert cached["status"] == "ok"
