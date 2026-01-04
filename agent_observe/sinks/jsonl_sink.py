"""
JSONL sink for agent-observe.

Simple file-based sink that writes newline-delimited JSON.
Good for debugging and simple deployments.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_observe.sinks.base import Sink

logger = logging.getLogger(__name__)

# Allowed category values to prevent path traversal
ALLOWED_CATEGORIES = frozenset({"runs", "spans", "events", "replay_cache"})


class JSONLSink(Sink):
    """
    JSONL file-based sink.

    Writes runs, spans, and events to separate JSONL files organized by date.
    Structure:
        {directory}/
            runs/
                2024-01-15.jsonl
            spans/
                2024-01-15.jsonl
            events/
                2024-01-15.jsonl
    """

    def __init__(
        self,
        directory: Path | str = ".riff/traces/",
        async_writes: bool = True,
    ):
        """
        Initialize JSONL sink.

        Args:
            directory: Base directory for JSONL files.
            async_writes: If True, writes are queued and flushed in background.
        """
        super().__init__(async_writes=async_writes)
        self.directory = Path(directory)

    def _do_initialize(self) -> None:
        """Create directory structure."""
        for subdir in ("runs", "spans", "events", "replay_cache"):
            (self.directory / subdir).mkdir(parents=True, exist_ok=True)

        logger.info(f"JSONL sink initialized at {self.directory}")

    def _get_file_path(self, category: str, date: datetime | None = None) -> Path:
        """
        Get file path for a category and date.

        Args:
            category: One of 'runs', 'spans', 'events', 'replay_cache'.
            date: Date for the file (defaults to today).

        Returns:
            Safe path within the sink directory.

        Raises:
            ValueError: If category is not in allowed list.
        """
        # Validate category to prevent path traversal
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_CATEGORIES))}"
            )

        if date is None:
            date = datetime.now()

        # Use strftime for safe filename (no user input)
        filename = date.strftime("%Y-%m-%d.jsonl")
        target_path = self.directory / category / filename

        # Additional safety: verify path is within expected directory
        try:
            resolved = target_path.resolve()
            base_resolved = self.directory.resolve()
            if not str(resolved).startswith(str(base_resolved)):
                raise ValueError(f"Path traversal detected: {target_path}")
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid path: {e}") from e

        return target_path

    def _append_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        """Append records to a JSONL file."""
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                line = json.dumps(record, default=str, ensure_ascii=False)
                f.write(line + "\n")

    def _do_write_runs(self, runs: list[dict[str, Any]]) -> None:
        """Write runs to JSONL."""
        path = self._get_file_path("runs")
        self._append_jsonl(path, runs)

    def _do_write_spans(self, spans: list[dict[str, Any]]) -> None:
        """Write spans to JSONL."""
        path = self._get_file_path("spans")
        self._append_jsonl(path, spans)

    def _do_write_events(self, events: list[dict[str, Any]]) -> None:
        """Write events to JSONL."""
        path = self._get_file_path("events")
        self._append_jsonl(path, events)

    def _do_write_replay_cache(self, entries: list[dict[str, Any]]) -> None:
        """Write replay cache entries to JSONL."""
        path = self._get_file_path("replay_cache")

        # Convert bytes to base64 for JSON serialization
        import base64

        processed = []
        for entry in entries:
            e = entry.copy()
            if e.get("result") and isinstance(e["result"], bytes):
                e["result"] = base64.b64encode(e["result"]).decode("ascii")
                e["result_encoding"] = "base64"
            processed.append(e)

        self._append_jsonl(path, processed)

    def get_runs(
        self,
        name: str | None = None,
        status: str | None = None,
        min_risk: int | None = None,
        tag: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Query runs from JSONL files.

        Note: This is not optimized for large datasets.
        For production use, consider SQLite or Postgres.
        """
        runs_dir = self.directory / "runs"
        if not runs_dir.exists():
            return []

        all_runs = []

        # Read all JSONL files, sorted by name (date) descending
        for jsonl_file in sorted(runs_dir.glob("*.jsonl"), reverse=True):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        run = json.loads(line)
                        all_runs.append(run)
                    except json.JSONDecodeError:
                        continue

        # Apply filters
        filtered = []
        for run in all_runs:
            if name and name.lower() not in run.get("name", "").lower():
                continue
            if status and run.get("status") != status:
                continue
            if min_risk is not None and (run.get("risk_score") or 0) < min_risk:
                continue
            if tag:
                eval_tags = run.get("eval_tags") or []
                if tag not in eval_tags:
                    continue
            filtered.append(run)

        # Sort by ts_start descending
        filtered.sort(key=lambda r: r.get("ts_start", 0), reverse=True)

        # Apply pagination
        return filtered[offset : offset + limit]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a single run by ID."""
        runs_dir = self.directory / "runs"
        if not runs_dir.exists():
            return None

        for jsonl_file in runs_dir.glob("*.jsonl"):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        run: dict[str, Any] = json.loads(line)
                        if run.get("run_id") == run_id:
                            return run
                    except json.JSONDecodeError:
                        continue

        return None

    def get_spans(self, run_id: str) -> list[dict[str, Any]]:
        """Get all spans for a run."""
        spans_dir = self.directory / "spans"
        if not spans_dir.exists():
            return []

        spans = []
        for jsonl_file in spans_dir.glob("*.jsonl"):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        span = json.loads(line)
                        if span.get("run_id") == run_id:
                            spans.append(span)
                    except json.JSONDecodeError:
                        continue

        # Sort by ts_start
        spans.sort(key=lambda s: s.get("ts_start", 0))
        return spans

    def get_events(self, run_id: str) -> list[dict[str, Any]]:
        """Get all events for a run."""
        events_dir = self.directory / "events"
        if not events_dir.exists():
            return []

        events = []
        for jsonl_file in events_dir.glob("*.jsonl"):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("run_id") == run_id:
                            events.append(event)
                    except json.JSONDecodeError:
                        continue

        # Sort by ts
        events.sort(key=lambda e: e.get("ts", 0))
        return events

    def get_replay_cache_entry(self, key: str) -> dict[str, Any] | None:
        """Get a replay cache entry by key."""
        import base64

        cache_dir = self.directory / "replay_cache"
        if not cache_dir.exists():
            return None

        for jsonl_file in cache_dir.glob("*.jsonl"):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry: dict[str, Any] = json.loads(line)
                        if entry.get("key") == key:
                            # Decode base64 result if present
                            if entry.get("result_encoding") == "base64":
                                entry["result"] = base64.b64decode(entry["result"])
                                del entry["result_encoding"]
                            return entry
                    except json.JSONDecodeError:
                        continue

        return None
