"""
PostgreSQL sink for agent-observe.

Production-ready sink for multi-instance deployments.
Uses psycopg3 for PostgreSQL connectivity.

Design decisions:
- Simple connection-per-operation (no pool) for minimal dependencies
- Parameterized queries throughout (SQL injection safe)
- Retry logic for transient connection failures
- Batch inserts using executemany for performance
- Graceful degradation when tables pre-exist (no CREATE permission needed)
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from agent_observe.sinks.base import Sink

logger = logging.getLogger(__name__)

# Connection settings
CONNECTION_TIMEOUT_SECONDS = 10
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 0.5

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
SCHEMA_VERSION = 2  # v0.1.7: Added user_id, session_id, input/output, etc.

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
    capture_mode TEXT CHECK (capture_mode IN ('off', 'metadata_only', 'evidence_only', 'full')),
    status TEXT CHECK (status IN ('ok', 'error', 'blocked')),
    risk_score INTEGER CHECK (risk_score >= 0 AND risk_score <= 100),
    eval_tags JSONB,
    policy_violations INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    model_calls INTEGER DEFAULT 0,
    latency_ms INTEGER,
    -- v0.1.7: Attribution fields
    user_id TEXT,
    session_id TEXT,
    prompt_version TEXT,
    prompt_hash TEXT,
    model_config JSONB,
    experiment_id TEXT,
    -- v0.1.7: Content fields (Wide Event)
    input_json TEXT,
    input_text TEXT,
    output_json TEXT,
    output_text TEXT,
    -- v0.1.7: Custom metadata
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_runs_ts_start ON runs(ts_start DESC);
CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_risk_score ON runs(risk_score);
CREATE INDEX IF NOT EXISTS idx_runs_project_env ON runs(project, env);
CREATE INDEX IF NOT EXISTS idx_runs_eval_tags ON runs USING GIN(eval_tags);
-- v0.1.7: Indexes for new fields
CREATE INDEX IF NOT EXISTS idx_runs_user_id ON runs(user_id);
CREATE INDEX IF NOT EXISTS idx_runs_session_id ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_runs_prompt_version ON runs(prompt_version);
CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON runs(experiment_id);

-- Spans table
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    parent_span_id TEXT,
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
CREATE INDEX IF NOT EXISTS idx_spans_parent ON spans(parent_span_id) WHERE parent_span_id IS NOT NULL;
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
CREATE INDEX IF NOT EXISTS idx_replay_created ON replay_cache(created_ts);
"""


class PostgresSink(Sink):
    """
    PostgreSQL-based sink for production deployments.

    Features:
    - Simple psycopg3 connections (no pool dependency)
    - Neon-compatible
    - Full query support for viewer
    - Automatic schema creation and migrations
    """

    # Valid PostgreSQL identifier pattern (letters, digits, underscores, starting with letter or underscore)
    _VALID_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    def __init__(
        self,
        database_url: str,
        async_writes: bool = True,
        schema: str = "public",
    ):
        """
        Initialize PostgreSQL sink.

        Args:
            database_url: PostgreSQL connection string.
            async_writes: If True, writes are queued and flushed in background.
            schema: PostgreSQL schema name (default: public).

        Raises:
            ValueError: If schema name contains invalid characters.
        """
        super().__init__(async_writes=async_writes)
        self.database_url = database_url

        # SECURITY: Validate schema name to prevent SQL injection
        if not self._VALID_IDENTIFIER_PATTERN.match(schema):
            raise ValueError(
                f"Invalid schema name '{schema}': must contain only letters, digits, "
                "and underscores, and start with a letter or underscore"
            )
        self.schema = schema
        self._initialized = False

    def _get_connection(self) -> Any:
        """
        Get a new database connection with timeout.

        Uses autocommit=False for explicit transaction control.
        Sets search_path to the configured schema.
        """
        import psycopg
        from psycopg import sql

        conn = psycopg.connect(
            self.database_url,
            connect_timeout=CONNECTION_TIMEOUT_SECONDS,
            autocommit=False,
        )
        # Set search_path to use the configured schema
        # Note: SET doesn't support parameterized queries, use sql.Identifier for safety
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
        return conn

    def _execute_with_retry(
        self,
        operation: str,
        func: Any,
    ) -> Any:
        """
        Execute a database operation with retry logic for transient failures.

        Args:
            operation: Description of the operation (for logging).
            func: Callable that performs the database operation.

        Returns:
            Result of the operation, or None if all retries failed.
        """
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                return func()
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if this is a transient/retryable error
                is_transient = any(
                    keyword in error_str
                    for keyword in [
                        "connection",
                        "timeout",
                        "temporary",
                        "unavailable",
                        "too many connections",
                        "connection refused",
                        "network",
                    ]
                )

                if is_transient and attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_SECONDS * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"Transient error in {operation} (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    # Non-transient error or final attempt
                    logger.error(f"Failed {operation} after {attempt + 1} attempts: {e}")
                    raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        return None

    def _check_tables_exist(self, conn: Any) -> bool:
        """
        Check if required tables already exist in the database.

        Uses a single query for efficiency.
        """
        required_tables = {"runs", "spans", "events"}
        result = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = ANY(%s)",
            (self.schema, list(required_tables)),
        ).fetchall()

        existing_tables = {row[0] for row in result}
        return required_tables.issubset(existing_tables)

    def _migrate_v1_to_v2(self, conn: Any) -> None:
        """Migrate schema from v1 to v2 (add v0.1.7 fields)."""
        # Add new columns to runs table
        new_columns = [
            ("user_id", "TEXT"),
            ("session_id", "TEXT"),
            ("prompt_version", "TEXT"),
            ("prompt_hash", "TEXT"),
            ("model_config", "JSONB"),
            ("experiment_id", "TEXT"),
            ("input_json", "TEXT"),
            ("input_text", "TEXT"),
            ("output_json", "TEXT"),
            ("output_text", "TEXT"),
            ("metadata", "JSONB"),
        ]

        for col_name, col_type in new_columns:
            try:
                conn.execute(f"ALTER TABLE runs ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
            except Exception as e:
                logger.warning(f"Failed to add column {col_name}: {e}")

        # Create new indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_runs_user_id ON runs(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_runs_session_id ON runs(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_runs_prompt_version ON runs(prompt_version)",
            "CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON runs(experiment_id)",
        ]
        for idx_sql in indexes:
            try:
                conn.execute(idx_sql)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")

        conn.commit()

    def _do_initialize(self) -> None:
        """Initialize connection and schema."""
        try:
            import psycopg  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "psycopg is required for PostgreSQL sink. "
                "Install with: pip install 'agent-observe[postgres]'"
            ) from e

        try:
            with self._get_connection() as conn:
                # Check current schema version
                current_version = 0
                try:
                    result = conn.execute(
                        "SELECT version FROM agent_observe_schema_version "
                        "ORDER BY version DESC LIMIT 1"
                    ).fetchone()
                    if result:
                        current_version = result[0]
                except Exception:
                    # Table doesn't exist yet
                    pass

                # Check if tables already exist
                tables_exist = self._check_tables_exist(conn)

                if current_version == 0 and not tables_exist:
                    # Fresh install - create full schema
                    try:
                        conn.execute(SCHEMA_SQL)
                        conn.execute(
                            "INSERT INTO agent_observe_schema_version (version) VALUES (%s)",
                            (SCHEMA_VERSION,),
                        )
                        conn.commit()
                        logger.info("PostgreSQL schema created successfully")
                    except Exception as schema_error:
                        conn.rollback()
                        raise RuntimeError(
                            f"Cannot create schema (tables don't exist and no CREATE permission). "
                            f"Please create tables manually - see AGENTS.md for SQL schema. "
                            f"Error: {schema_error}"
                        ) from schema_error
                elif current_version < 2:
                    # Migration needed from v1 to v2
                    logger.info(f"Migrating PostgreSQL schema from version {current_version} to {SCHEMA_VERSION}")
                    self._migrate_v1_to_v2(conn)
                    try:
                        conn.execute(
                            "INSERT INTO agent_observe_schema_version (version) VALUES (%s)",
                            (SCHEMA_VERSION,),
                        )
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        logger.debug("Schema version tracking skipped")
                    logger.info(f"PostgreSQL schema migrated to version {SCHEMA_VERSION}")
                else:
                    logger.info("PostgreSQL tables already up to date")

            self._initialized = True
            logger.info("PostgreSQL sink initialized")

        except Exception:
            raise

    def _ms_to_datetime(self, ms: int | None) -> datetime | None:
        """Convert milliseconds since epoch to datetime."""
        if ms is None:
            return None
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

    def _do_write_runs(self, runs: list[dict[str, Any]]) -> None:
        """Write runs to PostgreSQL using batch insert."""
        if not self._initialized or not runs:
            return

        def _write() -> None:
            with self._get_connection() as conn:
                try:
                    with conn.cursor() as cur:
                        # Prepare batch data
                        params_list = [
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
                                run.get("capture_mode"),
                                run.get("status"),
                                run.get("risk_score"),
                                json.dumps(run.get("eval_tags")) if run.get("eval_tags") else None,
                                run.get("policy_violations", 0),
                                run.get("tool_calls", 0),
                                run.get("model_calls", 0),
                                run.get("latency_ms"),
                                # v0.1.7: Attribution fields
                                run.get("user_id"),
                                run.get("session_id"),
                                run.get("prompt_version"),
                                run.get("prompt_hash"),
                                json.dumps(run.get("model_config")) if run.get("model_config") else None,
                                run.get("experiment_id"),
                                # v0.1.7: Content fields
                                run.get("input_json"),
                                run.get("input_text"),
                                run.get("output_json"),
                                run.get("output_text"),
                                json.dumps(run.get("metadata")) if run.get("metadata") else None,
                            )
                            for run in runs
                        ]

                        cur.executemany(
                            """
                            INSERT INTO runs (
                                run_id, trace_id, name, ts_start, ts_end, task,
                                agent_version, project, env, capture_mode, status,
                                risk_score, eval_tags, policy_violations,
                                tool_calls, model_calls, latency_ms,
                                user_id, session_id, prompt_version, prompt_hash,
                                model_config, experiment_id,
                                input_json, input_text, output_json, output_text,
                                metadata
                            ) VALUES (
                                %s::uuid, %s, %s, %s, %s, %s::jsonb,
                                %s, %s, %s, %s, %s,
                                %s, %s::jsonb, %s,
                                %s, %s, %s,
                                %s, %s, %s, %s,
                                %s::jsonb, %s,
                                %s, %s, %s, %s,
                                %s::jsonb
                            )
                            ON CONFLICT (run_id) DO UPDATE SET
                                ts_end = EXCLUDED.ts_end,
                                status = EXCLUDED.status,
                                risk_score = EXCLUDED.risk_score,
                                eval_tags = EXCLUDED.eval_tags,
                                policy_violations = EXCLUDED.policy_violations,
                                tool_calls = EXCLUDED.tool_calls,
                                model_calls = EXCLUDED.model_calls,
                                latency_ms = EXCLUDED.latency_ms,
                                user_id = EXCLUDED.user_id,
                                session_id = EXCLUDED.session_id,
                                prompt_version = EXCLUDED.prompt_version,
                                prompt_hash = EXCLUDED.prompt_hash,
                                model_config = EXCLUDED.model_config,
                                experiment_id = EXCLUDED.experiment_id,
                                input_json = EXCLUDED.input_json,
                                input_text = EXCLUDED.input_text,
                                output_json = EXCLUDED.output_json,
                                output_text = EXCLUDED.output_text,
                                metadata = EXCLUDED.metadata
                            """,
                            params_list,
                        )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

        self._execute_with_retry("write_runs", _write)

    def _do_write_spans(self, spans: list[dict[str, Any]]) -> None:
        """Write spans to PostgreSQL using batch insert."""
        if not self._initialized or not spans:
            return

        def _write() -> None:
            with self._get_connection() as conn:
                try:
                    with conn.cursor() as cur:
                        params_list = [
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
                            )
                            for span in spans
                        ]

                        cur.executemany(
                            """
                            INSERT INTO spans (
                                span_id, run_id, parent_span_id, kind, name,
                                ts_start, ts_end, status, attrs, error_message
                            ) VALUES (
                                %s, %s::uuid, %s, %s, %s,
                                %s, %s, %s, %s::jsonb, %s
                            )
                            ON CONFLICT (span_id) DO UPDATE SET
                                ts_end = EXCLUDED.ts_end,
                                status = EXCLUDED.status,
                                attrs = EXCLUDED.attrs,
                                error_message = EXCLUDED.error_message
                            """,
                            params_list,
                        )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

        self._execute_with_retry("write_spans", _write)

    def _do_write_events(self, events: list[dict[str, Any]]) -> None:
        """Write events to PostgreSQL using batch insert."""
        if not self._initialized or not events:
            return

        def _write() -> None:
            with self._get_connection() as conn:
                try:
                    with conn.cursor() as cur:
                        params_list = [
                            (
                                event.get("event_id"),
                                event.get("run_id"),
                                self._ms_to_datetime(event.get("ts")),
                                event.get("type"),
                                json.dumps(event.get("payload")) if event.get("payload") else None,
                            )
                            for event in events
                        ]

                        cur.executemany(
                            """
                            INSERT INTO events (
                                event_id, run_id, ts, type, payload
                            ) VALUES (
                                %s::uuid, %s::uuid, %s, %s, %s::jsonb
                            )
                            ON CONFLICT (event_id) DO NOTHING
                            """,
                            params_list,
                        )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

        self._execute_with_retry("write_events", _write)

    def _do_write_replay_cache(self, entries: list[dict[str, Any]]) -> None:
        """Write replay cache entries to PostgreSQL using batch insert."""
        if not self._initialized or not entries:
            return

        def _write() -> None:
            with self._get_connection() as conn:
                try:
                    with conn.cursor() as cur:
                        params_list = []
                        for entry in entries:
                            result = entry.get("result")
                            if result is not None and not isinstance(result, bytes):
                                result = json.dumps(result).encode("utf-8")

                            params_list.append(
                                (
                                    entry.get("key"),
                                    entry.get("tool_name"),
                                    entry.get("args_hash"),
                                    entry.get("tool_version"),
                                    self._ms_to_datetime(entry.get("created_ts")),
                                    entry.get("status"),
                                    result,
                                    entry.get("result_hash"),
                                )
                            )

                        cur.executemany(
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
                            params_list,
                        )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

        self._execute_with_retry("write_replay_cache", _write)

    def _do_close(self) -> None:
        """Mark sink as closed."""
        self._initialized = False

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
        if not self._initialized:
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

        with self._get_connection() as conn:
            result = conn.execute(query, params)
            columns = [desc.name for desc in result.description or []]
            return [self._row_to_dict(dict(zip(columns, row))) for row in result.fetchall()]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a single run by ID."""
        if not self._initialized:
            return None

        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT * FROM runs WHERE run_id = %s::uuid",
                (run_id,),
            )
            columns = [desc.name for desc in result.description or []]
            row = result.fetchone()
            return self._row_to_dict(dict(zip(columns, row))) if row else None

    def get_spans(self, run_id: str) -> list[dict[str, Any]]:
        """Get all spans for a run."""
        if not self._initialized:
            return []

        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT * FROM spans WHERE run_id = %s::uuid ORDER BY ts_start",
                (run_id,),
            )
            columns = [desc.name for desc in result.description or []]
            return [self._row_to_dict(dict(zip(columns, row))) for row in result.fetchall()]

    def get_events(self, run_id: str) -> list[dict[str, Any]]:
        """Get all events for a run."""
        if not self._initialized:
            return []

        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT * FROM events WHERE run_id = %s::uuid ORDER BY ts",
                (run_id,),
            )
            columns = [desc.name for desc in result.description or []]
            return [self._row_to_dict(dict(zip(columns, row))) for row in result.fetchall()]

    def get_replay_cache_entry(self, key: str) -> dict[str, Any] | None:
        """Get a replay cache entry by key."""
        if not self._initialized:
            return None

        with self._get_connection() as conn:
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
