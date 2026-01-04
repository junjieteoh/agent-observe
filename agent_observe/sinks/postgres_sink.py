"""
PostgreSQL sink for agent-observe.

Production-ready sink for multi-instance deployments.
Uses psycopg3 with connection pooling for optimal performance.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from agent_observe.sinks.base import Sink

logger = logging.getLogger(__name__)

# Allowed values for validation
ALLOWED_STATUSES = {"ok", "error", "blocked"}

# Maximum lengths for input validation
MAX_NAME_LENGTH = 256
MAX_TAG_LENGTH = 64


def _sanitize_identifier(value: str, max_length: int = 256) -> str:
    """
    Sanitize a string for safe use in queries.

    Removes potentially dangerous characters and enforces length limits.
    """
    if not value:
        return ""
    # Remove null bytes and control characters
    sanitized = "".join(c for c in value if c.isprintable() and c != "\x00")
    # Truncate to max length
    return sanitized[:max_length]


# Schema version for migrations
SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS agent_observe_schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT now()
);

-- Runs table
CREATE TABLE IF NOT EXISTS runs (
    run_id UUID PRIMARY KEY,
    trace_id TEXT,
    name TEXT NOT NULL,
    ts_start TIMESTAMPTZ NOT NULL,
    ts_end TIMESTAMPTZ,
    task JSONB,
    agent_version TEXT,
    project TEXT,
    env TEXT,
    status TEXT CHECK (status IN ('ok', 'error', 'blocked')),
    risk_score INTEGER,
    eval_tags JSONB,
    policy_violations INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    model_calls INTEGER DEFAULT 0,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_runs_ts_start ON runs(ts_start DESC);
CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_risk_score ON runs(risk_score);
CREATE INDEX IF NOT EXISTS idx_runs_project_env ON runs(project, env);

-- Spans table
CREATE TABLE IF NOT EXISTS spans (
    span_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    parent_span_id UUID,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    ts_start TIMESTAMPTZ NOT NULL,
    ts_end TIMESTAMPTZ,
    status TEXT CHECK (status IN ('ok', 'error', 'blocked')),
    attrs JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_spans_run_id ON spans(run_id);
CREATE INDEX IF NOT EXISTS idx_spans_kind_name ON spans(kind, name);
CREATE INDEX IF NOT EXISTS idx_spans_ts_start ON spans(ts_start);

-- Events table
CREATE TABLE IF NOT EXISTS events (
    event_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    type TEXT NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_events_run_id_type ON events(run_id, type);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

-- Replay cache table
CREATE TABLE IF NOT EXISTS replay_cache (
    key TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    tool_version TEXT,
    created_ts TIMESTAMPTZ DEFAULT now(),
    status TEXT CHECK (status IN ('ok', 'error')),
    result BYTEA,
    result_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_replay_tool_args ON replay_cache(tool_name, args_hash);
"""


class PostgresSink(Sink):
    """
    PostgreSQL-based sink for production deployments.

    Features:
    - Connection pooling via psycopg_pool
    - Neon-compatible
    - Full query support for viewer
    - Automatic schema creation and migrations
    """

    def __init__(
        self,
        database_url: str,
        async_writes: bool = True,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ):
        """
        Initialize PostgreSQL sink.

        Args:
            database_url: PostgreSQL connection string.
            async_writes: If True, writes are queued and flushed in background.
            min_pool_size: Minimum connection pool size.
            max_pool_size: Maximum connection pool size.
        """
        super().__init__(async_writes=async_writes)
        self.database_url = database_url
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self._pool: Any = None

    def _do_initialize(self) -> None:
        """Create connection pool and schema."""
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as e:
            raise ImportError(
                "psycopg[pool] is required for PostgreSQL sink. "
                "Install with: pip install 'agent-observe[postgres]'"
            ) from e

        pool = None
        try:
            pool = ConnectionPool(
                self.database_url,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                open=True,
            )

            # Create schema
            with pool.connection() as conn:
                conn.execute(SCHEMA_SQL)

                # Record schema version if not present
                result = conn.execute(
                    "SELECT version FROM agent_observe_schema_version "
                    "ORDER BY version DESC LIMIT 1"
                ).fetchone()

                if result is None:
                    conn.execute(
                        "INSERT INTO agent_observe_schema_version (version) VALUES (%s)",
                        (SCHEMA_VERSION,),
                    )
                conn.commit()

            # Only assign to instance after successful initialization
            self._pool = pool
            logger.info("PostgreSQL sink initialized")

        except Exception:
            # Clean up pool on failure to prevent resource leak
            if pool is not None:
                try:
                    pool.close()
                except Exception as cleanup_error:
                    logger.warning(f"Error closing pool during cleanup: {cleanup_error}")
            raise

    def _ms_to_datetime(self, ms: int | None) -> datetime | None:
        """Convert milliseconds since epoch to datetime."""
        if ms is None:
            return None
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

    def _do_write_runs(self, runs: list[dict[str, Any]]) -> None:
        """Write runs to PostgreSQL."""
        if not self._pool:
            return

        with self._pool.connection() as conn:
            for run in runs:
                conn.execute(
                    """
                    INSERT INTO runs (
                        run_id, trace_id, name, ts_start, ts_end, task,
                        agent_version, project, env, status,
                        risk_score, eval_tags, policy_violations,
                        tool_calls, model_calls, latency_ms
                    ) VALUES (
                        %s::uuid, %s, %s, %s, %s, %s::jsonb,
                        %s, %s, %s, %s,
                        %s, %s::jsonb, %s,
                        %s, %s, %s
                    )
                    ON CONFLICT (run_id) DO UPDATE SET
                        ts_end = EXCLUDED.ts_end,
                        status = EXCLUDED.status,
                        risk_score = EXCLUDED.risk_score,
                        eval_tags = EXCLUDED.eval_tags,
                        policy_violations = EXCLUDED.policy_violations,
                        tool_calls = EXCLUDED.tool_calls,
                        model_calls = EXCLUDED.model_calls,
                        latency_ms = EXCLUDED.latency_ms
                    """,
                    (
                        run.get("run_id"),
                        run.get("trace_id"),
                        run.get("name"),
                        self._ms_to_datetime(run.get("ts_start")),
                        self._ms_to_datetime(run.get("ts_end")),
                        json.dumps(run.get("task")) if run.get("task") else None,
                        run.get("agent_version"),
                        run.get("project"),
                        run.get("env"),
                        run.get("status"),
                        run.get("risk_score"),
                        json.dumps(run.get("eval_tags")) if run.get("eval_tags") else None,
                        run.get("policy_violations", 0),
                        run.get("tool_calls", 0),
                        run.get("model_calls", 0),
                        run.get("latency_ms"),
                    ),
                )
            conn.commit()

    def _do_write_spans(self, spans: list[dict[str, Any]]) -> None:
        """Write spans to PostgreSQL."""
        if not self._pool:
            return

        with self._pool.connection() as conn:
            for span in spans:
                conn.execute(
                    """
                    INSERT INTO spans (
                        span_id, run_id, parent_span_id, kind, name,
                        ts_start, ts_end, status, attrs, error_message
                    ) VALUES (
                        %s::uuid, %s::uuid, %s::uuid, %s, %s,
                        %s, %s, %s, %s::jsonb, %s
                    )
                    ON CONFLICT (span_id) DO UPDATE SET
                        ts_end = EXCLUDED.ts_end,
                        status = EXCLUDED.status,
                        attrs = EXCLUDED.attrs,
                        error_message = EXCLUDED.error_message
                    """,
                    (
                        span.get("span_id"),
                        span.get("run_id"),
                        span.get("parent_span_id"),
                        span.get("kind"),
                        span.get("name"),
                        self._ms_to_datetime(span.get("ts_start")),
                        self._ms_to_datetime(span.get("ts_end")),
                        span.get("status"),
                        json.dumps(span.get("attrs")) if span.get("attrs") else None,
                        span.get("error_message"),
                    ),
                )
            conn.commit()

    def _do_write_events(self, events: list[dict[str, Any]]) -> None:
        """Write events to PostgreSQL."""
        if not self._pool:
            return

        with self._pool.connection() as conn:
            for event in events:
                conn.execute(
                    """
                    INSERT INTO events (
                        event_id, run_id, ts, type, payload
                    ) VALUES (
                        %s::uuid, %s::uuid, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    (
                        event.get("event_id"),
                        event.get("run_id"),
                        self._ms_to_datetime(event.get("ts")),
                        event.get("type"),
                        json.dumps(event.get("payload")) if event.get("payload") else None,
                    ),
                )
            conn.commit()

    def _do_write_replay_cache(self, entries: list[dict[str, Any]]) -> None:
        """Write replay cache entries to PostgreSQL."""
        if not self._pool:
            return

        with self._pool.connection() as conn:
            for entry in entries:
                result = entry.get("result")
                if result is not None and not isinstance(result, bytes):
                    result = json.dumps(result).encode("utf-8")

                conn.execute(
                    """
                    INSERT INTO replay_cache (
                        key, tool_name, args_hash, tool_version,
                        created_ts, status, result, result_hash
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (key) DO UPDATE SET
                        status = EXCLUDED.status,
                        result = EXCLUDED.result,
                        result_hash = EXCLUDED.result_hash
                    """,
                    (
                        entry.get("key"),
                        entry.get("tool_name"),
                        entry.get("args_hash"),
                        entry.get("tool_version"),
                        self._ms_to_datetime(entry.get("created_ts")),
                        entry.get("status"),
                        result,
                        entry.get("result_hash"),
                    ),
                )
            conn.commit()

    def _do_close(self) -> None:
        """Close connection pool."""
        if self._pool:
            self._pool.close()
            self._pool = None

    # Query methods for viewer

    def get_runs(
        self,
        name: str | None = None,
        status: str | None = None,
        min_risk: int | None = None,
        tag: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query runs with optional filters."""
        if not self._pool:
            return []

        query = "SELECT * FROM runs WHERE 1=1"
        params: list[Any] = []

        # Validate and sanitize name input
        if name:
            sanitized_name = _sanitize_identifier(name, MAX_NAME_LENGTH)
            if sanitized_name:
                query += " AND name ILIKE %s"
                params.append(f"%{sanitized_name}%")

        # Validate status against allowed values
        if status and status.lower() in ALLOWED_STATUSES:
            query += " AND status = %s"
            params.append(status.lower())
            # Silently ignore invalid status values

        # Validate min_risk is in valid range
        if min_risk is not None:
            # Clamp to valid range 0-100
            clamped_risk = max(0, min(100, int(min_risk)))
            query += " AND risk_score >= %s"
            params.append(clamped_risk)

        # Validate and sanitize tag input - use JSONB containment for proper search
        if tag:
            sanitized_tag = _sanitize_identifier(tag, MAX_TAG_LENGTH)
            if sanitized_tag:
                # Use JSONB @> operator for proper array containment check
                query += " AND eval_tags @> %s::jsonb"
                params.append(json.dumps([sanitized_tag]))

        # Validate limit and offset
        validated_limit = max(1, min(1000, limit))  # Max 1000 results
        validated_offset = max(0, offset)

        query += " ORDER BY ts_start DESC LIMIT %s OFFSET %s"
        params.extend([validated_limit, validated_offset])

        with self._pool.connection() as conn:
            result = conn.execute(query, params)
            columns = [desc.name for desc in result.description or []]
            return [self._row_to_dict(dict(zip(columns, row))) for row in result.fetchall()]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a single run by ID."""
        if not self._pool:
            return None

        with self._pool.connection() as conn:
            result = conn.execute(
                "SELECT * FROM runs WHERE run_id = %s::uuid",
                (run_id,),
            )
            columns = [desc.name for desc in result.description or []]
            row = result.fetchone()
            return self._row_to_dict(dict(zip(columns, row))) if row else None

    def get_spans(self, run_id: str) -> list[dict[str, Any]]:
        """Get all spans for a run."""
        if not self._pool:
            return []

        with self._pool.connection() as conn:
            result = conn.execute(
                "SELECT * FROM spans WHERE run_id = %s::uuid ORDER BY ts_start",
                (run_id,),
            )
            columns = [desc.name for desc in result.description or []]
            return [self._row_to_dict(dict(zip(columns, row))) for row in result.fetchall()]

    def get_events(self, run_id: str) -> list[dict[str, Any]]:
        """Get all events for a run."""
        if not self._pool:
            return []

        with self._pool.connection() as conn:
            result = conn.execute(
                "SELECT * FROM events WHERE run_id = %s::uuid ORDER BY ts",
                (run_id,),
            )
            columns = [desc.name for desc in result.description or []]
            return [self._row_to_dict(dict(zip(columns, row))) for row in result.fetchall()]

    def get_replay_cache_entry(self, key: str) -> dict[str, Any] | None:
        """Get a replay cache entry by key."""
        if not self._pool:
            return None

        with self._pool.connection() as conn:
            result = conn.execute(
                "SELECT * FROM replay_cache WHERE key = %s",
                (key,),
            )
            columns = [desc.name for desc in result.description or []]
            row = result.fetchone()
            if row is None:
                return None

            entry = self._row_to_dict(dict(zip(columns, row)))
            # Decode result if present
            if entry.get("result") and isinstance(entry["result"], (bytes, memoryview)):
                result_bytes = (
                    bytes(entry["result"])
                    if isinstance(entry["result"], memoryview)
                    else entry["result"]
                )
                try:
                    entry["result"] = json.loads(result_bytes.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    entry["result"] = result_bytes
            return entry

    @staticmethod
    def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
        """Convert database row to dict with timestamp handling."""
        result = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                # Convert to milliseconds since epoch
                result[key] = int(value.timestamp() * 1000)
            else:
                result[key] = value
        return result
