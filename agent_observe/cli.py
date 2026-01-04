"""
CLI for agent-observe.

Provides commands:
- agent-observe view: Start the local viewer
- agent-observe export-jsonl: Export data to JSONL
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_view(args: argparse.Namespace) -> int:
    """Start the local viewer."""
    try:
        from agent_observe.viewer.app import create_app
    except ImportError as e:
        print(
            f"Viewer dependencies not installed: {e}\n"
            "Install with: pip install 'agent-observe[viewer]'",
            file=sys.stderr,
        )
        return 1

    # Determine database source
    database_url = args.database_url or os.environ.get("DATABASE_URL")
    sqlite_path = args.db

    if database_url:
        print(f"Using PostgreSQL: {database_url[:50]}...")
        backend = "postgres"
        backend_url = database_url
    elif sqlite_path:
        path = Path(sqlite_path)
        if not path.exists():
            print(f"SQLite database not found: {path}", file=sys.stderr)
            return 1
        print(f"Using SQLite: {path}")
        backend = "sqlite"
        backend_url = str(path)
    else:
        # Default SQLite path
        default_path = Path(".riff/observe.db")
        if default_path.exists():
            print(f"Using SQLite: {default_path}")
            backend = "sqlite"
            backend_url = str(default_path)
        else:
            print(
                "No database found. Run your agent with observe.install() first,\n"
                "or specify --db or --database-url",
                file=sys.stderr,
            )
            return 1

    # Create and run the app
    app = create_app(backend=backend, backend_url=backend_url)

    try:
        import uvicorn

        print(f"\nStarting viewer at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop\n")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        print("\nShutting down...")
    except ImportError:
        print(
            "uvicorn not installed. Install with: pip install 'agent-observe[viewer]'",
            file=sys.stderr,
        )
        return 1

    return 0


def cmd_export_jsonl(args: argparse.Namespace) -> int:
    """Export data to JSONL."""
    from agent_observe.sinks.base import Sink

    # Determine source
    database_url = args.database_url or os.environ.get("DATABASE_URL")
    sink: Sink

    if database_url:
        # Use Postgres
        try:
            from agent_observe.sinks.postgres_sink import PostgresSink

            sink = PostgresSink(database_url=database_url, async_writes=False)
            sink.initialize()
        except ImportError as e:
            print(
                f"Postgres dependencies not installed: {e}\n"
                "Install with: pip install 'agent-observe[postgres]'",
                file=sys.stderr,
            )
            return 1
    elif args.db:
        # Use SQLite
        path = Path(args.db)
        if not path.exists():
            print(f"SQLite database not found: {path}", file=sys.stderr)
            return 1
        from agent_observe.sinks.sqlite_sink import SQLiteSink

        sink = SQLiteSink(path=path, async_writes=False)
        sink.initialize()
    else:
        # Default SQLite path
        default_path = Path(".riff/observe.db")
        if default_path.exists():
            from agent_observe.sinks.sqlite_sink import SQLiteSink

            sink = SQLiteSink(path=default_path, async_writes=False)
            sink.initialize()
        else:
            print("No database found. Specify --db or --database-url", file=sys.stderr)
            return 1

    # Export
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_file = output_dir / "runs.jsonl"
    spans_file = output_dir / "spans.jsonl"
    events_file = output_dir / "events.jsonl"

    # Get all runs
    print(f"Exporting to {output_dir}...")

    runs = sink.get_runs(limit=10000)
    print(f"Found {len(runs)} runs")

    with open(runs_file, "w", encoding="utf-8") as f:
        for run in runs:
            f.write(json.dumps(run, default=str) + "\n")

    span_count = 0
    event_count = 0

    with open(spans_file, "w", encoding="utf-8") as sf, open(
        events_file, "w", encoding="utf-8"
    ) as ef:
        for run in runs:
            run_id = run["run_id"]

            for span in sink.get_spans(run_id):
                sf.write(json.dumps(span, default=str) + "\n")
                span_count += 1

            for event in sink.get_events(run_id):
                ef.write(json.dumps(event, default=str) + "\n")
                event_count += 1

    print(f"Exported {len(runs)} runs, {span_count} spans, {event_count} events")
    print(f"Files: {runs_file}, {spans_file}, {events_file}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    from agent_observe import __version__

    parser = argparse.ArgumentParser(
        prog="agent-observe",
        description="Observability, audit, and eval for AI agent applications",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # view command
    view_parser = subparsers.add_parser("view", help="Start the local viewer")
    view_parser.add_argument(
        "--db",
        type=str,
        help="Path to SQLite database",
    )
    view_parser.add_argument(
        "--database-url",
        type=str,
        help="PostgreSQL connection URL",
    )
    view_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    view_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to (default: 8765)",
    )

    # export-jsonl command
    export_parser = subparsers.add_parser("export-jsonl", help="Export data to JSONL")
    export_parser.add_argument(
        "--db",
        type=str,
        help="Path to SQLite database",
    )
    export_parser.add_argument(
        "--database-url",
        type=str,
        help="PostgreSQL connection URL",
    )
    export_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./export",
        help="Output directory (default: ./export)",
    )

    args = parser.parse_args(argv)

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Dispatch command
    if args.command == "view":
        return cmd_view(args)
    elif args.command == "export-jsonl":
        return cmd_export_jsonl(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
