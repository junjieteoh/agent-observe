"""
FastAPI viewer application for agent-observe.

Provides a minimal web UI for browsing runs and debugging agent behavior.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from agent_observe.sinks.base import Sink

# Input validation constants
ALLOWED_STATUSES = {"ok", "error", "blocked"}
MAX_NAME_LENGTH = 256
MAX_TAG_LENGTH = 64
MAX_SPAN_DEPTH = 100  # Maximum recursion depth for span tree

# Regex for valid UUID format (accepts both with and without dashes)
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$",
    re.IGNORECASE,
)


def validate_uuid(value: str) -> bool:
    """Validate that a string is a valid UUID format."""
    if not value or len(value) > 40:
        return False
    return bool(UUID_PATTERN.match(value))


def sanitize_string(value: str | None, max_length: int = 256) -> str | None:
    """Sanitize a string input."""
    if value is None:
        return None
    # Remove null bytes and control characters
    sanitized = "".join(c for c in value if c.isprintable() and c != "\x00")
    return sanitized[:max_length] if sanitized else None

# Template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


def format_timestamp(ts: int | None) -> str:
    """Format milliseconds timestamp to readable string."""
    if ts is None:
        return "-"
    dt = datetime.fromtimestamp(ts / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(ms: int | None) -> str:
    """Format duration in milliseconds to readable string."""
    if ms is None:
        return "-"
    if ms < 1000:
        return f"{ms}ms"
    elif ms < 60000:
        return f"{ms / 1000:.1f}s"
    else:
        return f"{ms / 60000:.1f}m"


def risk_color(score: int | None) -> str:
    """Get color class for risk score."""
    if score is None:
        return "gray"
    if score >= 40:
        return "red"
    elif score >= 15:
        return "orange"
    else:
        return "green"


_STATUS_COLORS = {"ok": "green", "error": "red", "blocked": "orange"}


def status_color(status: str | None) -> str:
    """Get color class for status."""
    return _STATUS_COLORS.get(status or "", "gray")


def create_app(backend: str = "sqlite", backend_url: str = ".riff/observe.db") -> FastAPI:
    """
    Create the FastAPI viewer application.

    Args:
        backend: Backend type ("sqlite" or "postgres").
        backend_url: Database path or URL.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="agent-observe Viewer",
        description="Observability viewer for AI agent applications",
        version="0.1.0",
    )

    # Initialize sink based on backend
    sink: Sink
    if backend == "postgres":
        from agent_observe.sinks.postgres_sink import PostgresSink

        sink = PostgresSink(database_url=backend_url, async_writes=False)
    else:
        from agent_observe.sinks.sqlite_sink import SQLiteSink

        sink = SQLiteSink(path=backend_url, async_writes=False)

    sink.initialize()

    # Set up templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Add custom filters
    templates.env.filters["format_timestamp"] = format_timestamp
    templates.env.filters["format_duration"] = format_duration
    templates.env.globals["risk_color"] = risk_color
    templates.env.globals["status_color"] = status_color

    @app.get("/", response_class=HTMLResponse)
    async def runs_list(
        request: Request,
        name: str | None = Query(None, description="Filter by run name"),
        status: str | None = Query(None, description="Filter by status"),
        min_risk: int | None = Query(None, description="Filter by minimum risk score"),
        tag: str | None = Query(None, description="Filter by eval tag"),
        page: int = Query(1, ge=1, description="Page number"),
        limit: int = Query(50, ge=1, le=100, description="Results per page"),
    ) -> HTMLResponse:
        """List runs with optional filters."""
        offset = (page - 1) * limit

        runs = sink.get_runs(
            name=name,
            status=status,
            min_risk=min_risk,
            tag=tag,
            limit=limit + 1,  # Get one extra to check if there's a next page
            offset=offset,
        )

        has_next = len(runs) > limit
        if has_next:
            runs = runs[:limit]

        return templates.TemplateResponse(
            "runs.html",
            {
                "request": request,
                "runs": runs,
                "filters": {
                    "name": name,
                    "status": status,
                    "min_risk": min_risk,
                    "tag": tag,
                },
                "page": page,
                "limit": limit,
                "has_next": has_next,
                "has_prev": page > 1,
            },
        )

    @app.get("/run/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str) -> HTMLResponse:
        """Show run detail with spans and events."""
        # Validate run_id format
        if not validate_uuid(run_id):
            return templates.TemplateResponse(
                "run_detail.html",
                {
                    "request": request,
                    "run": None,
                    "spans": [],
                    "events": [],
                    "error": "Invalid run ID format",
                },
                status_code=400,
            )

        run = sink.get_run(run_id)
        if run is None:
            return templates.TemplateResponse(
                "run_detail.html",
                {
                    "request": request,
                    "run": None,
                    "spans": [],
                    "events": [],
                    "error": "Run not found",
                },
                status_code=404,
            )

        spans = sink.get_spans(run_id)
        events = sink.get_events(run_id)

        # Build span tree with depth limit
        span_tree = build_span_tree(spans)

        return templates.TemplateResponse(
            "run_detail.html",
            {
                "request": request,
                "run": run,
                "spans": spans,
                "span_tree": span_tree,
                "events": events,
                "error": None,
            },
        )

    @app.get("/health")
    async def health_check() -> JSONResponse:
        """Health check endpoint for load balancers."""
        return JSONResponse({"status": "ok"})

    @app.get("/ready")
    async def readiness_check() -> JSONResponse:
        """Readiness check endpoint - verifies database connectivity."""
        try:
            # Try to query the database
            sink.get_runs(limit=1)
            return JSONResponse({"status": "ready"})
        except Exception as e:
            return JSONResponse(
                {"status": "not ready", "error": str(e)[:100]},
                status_code=503,
            )

    @app.get("/api/runs")
    async def api_runs(
        name: str | None = Query(None, max_length=MAX_NAME_LENGTH),
        status: str | None = Query(None),
        min_risk: int | None = Query(None, ge=0, le=100),
        tag: str | None = Query(None, max_length=MAX_TAG_LENGTH),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> list[dict[str, Any]]:
        """API endpoint for runs."""
        # Validate and sanitize inputs
        validated_status = status.lower() if status and status.lower() in ALLOWED_STATUSES else None

        return sink.get_runs(
            name=sanitize_string(name, MAX_NAME_LENGTH),
            status=validated_status,
            min_risk=min_risk,
            tag=sanitize_string(tag, MAX_TAG_LENGTH),
            limit=limit,
            offset=offset,
        )

    @app.get("/api/runs/{run_id}")
    async def api_run(run_id: str) -> dict[str, Any]:
        """API endpoint for single run."""
        # Validate run_id format
        if not validate_uuid(run_id):
            raise HTTPException(status_code=400, detail="Invalid run ID format")

        run = sink.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "run": run,
            "spans": sink.get_spans(run_id),
            "events": sink.get_events(run_id),
        }

    return app


def build_span_tree(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build hierarchical span tree from flat list.

    Includes cycle detection and depth limiting for safety.
    """
    if not spans:
        return []

    # Index spans by ID
    span_map = {s["span_id"]: {**s, "children": []} for s in spans}

    # Build tree with cycle detection
    roots = []
    seen_ids: set[str] = set()

    for span in span_map.values():
        span_id = span["span_id"]
        if span_id in seen_ids:
            continue  # Skip cycles
        seen_ids.add(span_id)

        parent_id = span.get("parent_span_id")
        if parent_id and parent_id in span_map and parent_id != span_id:
            span_map[parent_id]["children"].append(span)
        else:
            roots.append(span)

    # Sort by start time with depth limit
    def sort_by_time(nodes: list[dict[str, Any]], depth: int = 0) -> list[dict[str, Any]]:
        if depth > MAX_SPAN_DEPTH:
            return nodes  # Stop recursion at max depth

        nodes.sort(key=lambda s: s.get("ts_start", 0))
        for node in nodes:
            sort_by_time(node["children"], depth + 1)
        return nodes

    return sort_by_time(roots)
