"""
SQLite sink for agent-observe.

Primary sink for local development with full query support.
Uses WAL mode for better concurrent access.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
from pathlib import Path
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
SCHEMA_VERSION = 2  # v0.1.7: Added user_id, session_id, input/output, etc.

SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS agent_observe_schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now'))
);

-- Runs table
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    trace_id TEXT,
    name TEXT NOT NULL,
    ts_start INTEGER NOT NULL,
    ts_end INTEGER,
    task TEXT,  -- JSON
    agent_version TEXT,
    project TEXT,
    env TEXT,
    capture_mode TEXT CHECK (capture_mode IN ('off', 'metadata_only', 'evidence_only', 'full')),
    status TEXT CHECK (status IN ('ok', 'error', 'blocked')),
    risk_score INTEGER,
    eval_tags TEXT,  -- JSON array
    policy_violations INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    model_calls INTEGER DEFAULT 0,
    latency_ms INTEGER,
    -- v0.1.7: Attribution fields
    user_id TEXT,
    session_id TEXT,
    prompt_version TEXT,
    prompt_hash TEXT,
    model_config TEXT,  -- JSON
    experiment_id TEXT,
    -- v0.1.7: Content fields (Wide Event)
    input_json TEXT,
    input_text TEXT,
    output_json TEXT,
    output_text TEXT,
    -- v0.1.7: Custom metadata
    metadata TEXT,  -- JSON
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_runs_ts_start ON runs(ts_start DESC);
CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_risk_score ON runs(risk_score);
CREATE INDEX IF NOT EXISTS idx_runs_project_env ON runs(project, env);
-- v0.1.7: Indexes for new fields
CREATE INDEX IF NOT EXISTS idx_runs_user_id ON runs(user_id);
CREATE INDEX IF NOT EXISTS idx_runs_session_id ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_runs_prompt_version ON runs(prompt_version);
CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON runs(experiment_id);

-- Spans table
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    parent_span_id TEXT,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    ts_start INTEGER NOT NULL,
    ts_end INTEGER,
    status TEXT CHECK (status IN ('ok', 'error', 'blocked')),
    attrs TEXT,  -- JSON
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_spans_run_id ON spans(run_id);
CREATE INDEX IF NOT EXISTS idx_spans_kind_name ON spans(kind, name);
CREATE INDEX IF NOT EXISTS idx_spans_ts_start ON spans(ts_start);

-- Events table
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ts INTEGER NOT NULL,
    type TEXT NOT NULL,
    payload TEXT,  -- JSON
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_run_id_type ON events(run_id, type);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

-- Replay cache table
CREATE TABLE IF NOT EXISTS replay_cache (
    key TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    tool_version TEXT,
    created_ts INTEGER NOT NULL,
    status TEXT CHECK (status IN ('ok', 'error')),
    result BLOB,  -- Serialized result (only in evidence_only/full mode)
    result_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_replay_tool_args ON replay_cache(tool_name, args_hash);
"""


class SQLiteSink(Sink):
    """
    SQLite-based sink for local development.

    Features:
    - WAL mode for concurrent reads/writes
    - Full query support for viewer
    - Thread-safe connection handling
    - Automatic schema creation
    """

    def __init__(
        self,
        path: Path | str = ".riff/observe.db",
        async_writes: bool = True,
    ):
        """
        Initialize SQLite sink.

        Args:
            path: Path to SQLite database file.
            async_writes: If True, writes are queued and flushed in background.
        """
        super().__init__(async_writes=async_writes)
        self.path = Path(path)
        self._local = threading.local()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _do_initialize(self) -> None:
        """Create database directory and schema."""
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        conn = self._get_connection()

        # Check if tables exist and what version we have
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_observe_schema_version'"
        )
        has_schema_table = cursor.fetchone() is not None

        current_version = 0
        if has_schema_table:
            cursor = conn.execute(
                "SELECT version FROM agent_observe_schema_version ORDER BY version DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                current_version = row[0]

        # If fresh database, create schema
        if current_version == 0:
            conn.executescript(SCHEMA_SQL)
            conn.execute(
                "INSERT INTO agent_observe_schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            conn.commit()
            logger.info(f"SQLite sink initialized at {self.path}")
            return

        # Migrate from version 1 to version 2
        if current_version < 2:
            logger.info(f"Migrating SQLite schema from version {current_version} to {SCHEMA_VERSION}")
            self._migrate_v1_to_v2(conn)
            conn.execute(
                "INSERT INTO agent_observe_schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            conn.commit()
            logger.info(f"SQLite schema migrated to version {SCHEMA_VERSION}")

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        """Migrate schema from v1 to v2 (add v0.1.7 fields)."""
        # Add new columns to runs table (SQLite requires ALTER TABLE for each column)
        new_columns = [
            ("user_id", "TEXT"),
            ("session_id", "TEXT"),
            ("prompt_version", "TEXT"),
            ("prompt_hash", "TEXT"),
            ("model_config", "TEXT"),
            ("experiment_id", "TEXT"),
            ("input_json", "TEXT"),
            ("input_text", "TEXT"),
            ("output_json", "TEXT"),
            ("output_text", "TEXT"),
            ("metadata", "TEXT"),
        ]

        for col_name, col_type in new_columns:
            try:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
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
            except sqlite3.OperationalError as e:
                logger.warning(f"Failed to create index: {e}")

    def _do_write_runs(self, runs: list[dict[str, Any]]) -> None:
        """Write runs to SQLite."""
        conn = self._get_connection()

        for run in runs:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, trace_id, name, ts_start, ts_end, task,
                    agent_version, project, env, capture_mode, status,
                    risk_score, eval_tags, policy_violations,
                    tool_calls, model_calls, latency_ms,
                    user_id, session_id, prompt_version, prompt_hash,
                    model_config, experiment_id,
                    input_json, input_text, output_json, output_text,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.get("run_id"),
                    run.get("trace_id"),
                    run.get("name"),
                    run.get("ts_start"),
                    run.get("ts_end"),
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
                ),
            )

        conn.commit()

    def _do_write_spans(self, spans: list[dict[str, Any]]) -> None:
        """Write spans to SQLite."""
        conn = self._get_connection()

        for span in spans:
            conn.execute(
                """
                INSERT OR REPLACE INTO spans (
                    span_id, run_id, parent_span_id, kind, name,
                    ts_start, ts_end, status, attrs, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    span.get("span_id"),
                    span.get("run_id"),
                    span.get("parent_span_id"),
                    span.get("kind"),
                    span.get("name"),
                    span.get("ts_start"),
                    span.get("ts_end"),
                    span.get("status"),
                    json.dumps(span.get("attrs")) if span.get("attrs") else None,
                    span.get("error_message"),
                ),
            )

        conn.commit()

    def _do_write_events(self, events: list[dict[str, Any]]) -> None:
        """Write events to SQLite."""
        conn = self._get_connection()

        for event in events:
            conn.execute(
                """
                INSERT OR REPLACE INTO events (
                    event_id, run_id, ts, type, payload
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.get("event_id"),
                    event.get("run_id"),
                    event.get("ts"),
                    event.get("type"),
                    json.dumps(event.get("payload")) if event.get("payload") else None,
                ),
            )

        conn.commit()

    def _do_write_replay_cache(self, entries: list[dict[str, Any]]) -> None:
        """Write replay cache entries to SQLite."""
        conn = self._get_connection()

        for entry in entries:
            result = entry.get("result")
            if result is not None and not isinstance(result, bytes):
                result = json.dumps(result).encode("utf-8")

            conn.execute(
                """
                INSERT OR REPLACE INTO replay_cache (
                    key, tool_name, args_hash, tool_version,
                    created_ts, status, result, result_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.get("key"),
                    entry.get("tool_name"),
                    entry.get("args_hash"),
                    entry.get("tool_version"),
                    entry.get("created_ts"),
                    entry.get("status"),
                    result,
                    entry.get("result_hash"),
                ),
            )

        conn.commit()

    def _do_close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

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
        conn = self._get_connection()

        query = "SELECT * FROM runs WHERE 1=1"
        params: list[Any] = []

        # Validate and sanitize name input
        if name:
            sanitized_name = _sanitize_identifier(name, MAX_NAME_LENGTH)
            if sanitized_name:
                query += " AND name LIKE ?"
                params.append(f"%{sanitized_name}%")

        # Validate status against allowed values
        if status and status.lower() in ALLOWED_STATUSES:
            query += " AND status = ?"
            params.append(status.lower())
            # Silently ignore invalid status values

        # Validate min_risk is in valid range
        if min_risk is not None:
            # Clamp to valid range 0-100
            clamped_risk = max(0, min(100, int(min_risk)))
            query += " AND risk_score >= ?"
            params.append(clamped_risk)

        # Validate and sanitize tag input - use safe LIKE with escaped characters
        if tag:
            sanitized_tag = _sanitize_identifier(tag, MAX_TAG_LENGTH)
            if sanitized_tag:
                # Escape special LIKE characters to prevent injection
                escaped_tag = (
                    sanitized_tag
                    .replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                    .replace('"', '\\"')
                    .replace("'", "\\'")
                )
                query += " AND eval_tags LIKE ? ESCAPE '\\'"
                params.append(f'%"{escaped_tag}"%')

        # Validate limit and offset
        validated_limit = max(1, min(1000, limit))  # Max 1000 results
        validated_offset = max(0, offset)

        query += " ORDER BY ts_start DESC LIMIT ? OFFSET ?"
        params.extend([validated_limit, validated_offset])

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a single run by ID."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def get_spans(self, run_id: str) -> list[dict[str, Any]]:
        """Get all spans for a run."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM spans WHERE run_id = ? ORDER BY ts_start",
            (run_id,),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_events(self, run_id: str) -> list[dict[str, Any]]:
        """Get all events for a run."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY ts",
            (run_id,),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_replay_cache_entry(self, key: str) -> dict[str, Any] | None:
        """Get a replay cache entry by key."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM replay_cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row is None:
            return None

        entry = self._row_to_dict(row)
        # Decode result if present (keep as bytes on decode failure)
        if entry.get("result") and isinstance(entry["result"], bytes):
            with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError):
                entry["result"] = json.loads(entry["result"].decode("utf-8"))
        return entry

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert sqlite3.Row to dict with JSON parsing."""
        d = dict(row)

        # Parse JSON fields
        json_fields = (
            "task", "attrs", "payload", "eval_tags",
            # v0.1.7 fields
            "model_config", "input_json", "output_json", "metadata",
        )
        for field in json_fields:
            if field in d and d[field] is not None and isinstance(d[field], str):
                with contextlib.suppress(json.JSONDecodeError):
                    d[field] = json.loads(d[field])

        return d
