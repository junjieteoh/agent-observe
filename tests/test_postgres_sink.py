"""Tests for PostgreSQL sink.

Unit tests run without a database (using mocks).
Integration tests require a running PostgreSQL database and are skipped if DATABASE_URL is not set.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestPostgresSinkUnit:
    """Unit tests for PostgresSink (no database required)."""

    def test_check_tables_exist_all_present(self) -> None:
        """Test _check_tables_exist when all tables exist."""
        from agent_observe.sinks.postgres_sink import PostgresSink

        sink = PostgresSink(database_url="postgresql://test", async_writes=False)

        mock_conn = MagicMock()
        # New implementation uses a single query returning all existing table names
        mock_conn.execute.return_value.fetchall.return_value = [
            ("runs",),
            ("spans",),
            ("events",),
        ]

        result = sink._check_tables_exist(mock_conn)
        assert result is True
        assert mock_conn.execute.call_count == 1  # Single efficient query

    def test_check_tables_exist_missing_table(self) -> None:
        """Test _check_tables_exist when a table is missing."""
        from agent_observe.sinks.postgres_sink import PostgresSink

        sink = PostgresSink(database_url="postgresql://test", async_writes=False)

        mock_conn = MagicMock()
        # Only runs and events exist, spans is missing
        mock_conn.execute.return_value.fetchall.return_value = [
            ("runs",),
            ("events",),
        ]

        result = sink._check_tables_exist(mock_conn)
        assert result is False

    def test_initialization_skips_schema_when_tables_exist(self) -> None:
        """Test that schema creation is skipped when tables already exist."""
        from agent_observe.sinks.postgres_sink import PostgresSink

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        # Tables exist
        mock_conn.execute.return_value.fetchone.return_value = (True,)

        with patch.object(PostgresSink, "_get_connection", return_value=mock_conn):
            with patch.object(PostgresSink, "_check_tables_exist", return_value=True):
                sink = PostgresSink(database_url="postgresql://test", async_writes=False)

                # Mock the psycopg import
                with patch.dict("sys.modules", {"psycopg": MagicMock()}):
                    sink._do_initialize()

                assert sink._initialized is True

    def test_schema_sql_has_text_span_id(self) -> None:
        """Verify span_id uses TEXT type for OpenTelemetry compatibility."""
        from agent_observe.sinks.postgres_sink import SCHEMA_SQL

        assert "span_id TEXT PRIMARY KEY" in SCHEMA_SQL
        assert "parent_span_id TEXT" in SCHEMA_SQL
        # Should NOT have UUID for span_id
        assert "span_id UUID" not in SCHEMA_SQL

    def test_write_spans_uses_text_not_uuid_cast(self) -> None:
        """Verify span writes don't use ::uuid cast for span_id."""
        from agent_observe.sinks.postgres_sink import PostgresSink

        sink = PostgresSink(database_url="postgresql://test", async_writes=False)
        sink._initialized = True

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(sink, "_get_connection", return_value=mock_conn):
            # Write a span with 16-char span_id (OpenTelemetry format)
            sink._do_write_spans([{
                "span_id": "abc123def456789",  # 16 chars
                "run_id": "550e8400-e29b-41d4-a716-446655440000",  # Full UUID
                "parent_span_id": None,
                "kind": "tool",
                "name": "test",
                "ts_start": 1704067200000,
                "ts_end": 1704067201000,
                "status": "ok",
                "attrs": {},
            }])

        # Check that executemany was called on the cursor (batch insert)
        mock_cursor.executemany.assert_called()
        call_args = mock_cursor.executemany.call_args
        sql = call_args[0][0]

        # The span_id should NOT be cast to uuid
        # Pattern: VALUES (%s, %s::uuid, %s, ... where first %s is span_id (no cast),
        # second is run_id (::uuid), third is parent_span_id (no cast)
        normalized_sql = sql.replace("\n", "").replace(" ", "")
        # Should have %s,%s::uuid,%s at the start of VALUES
        assert "VALUES(%s,%s::uuid,%s," in normalized_sql


# Integration tests - skip if no DATABASE_URL
_skip_without_db = pytest.mark.skipif(
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
@_skip_without_db
class TestPostgresSinkIntegration:
    """Integration tests for PostgresSink (requires DATABASE_URL)."""

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
