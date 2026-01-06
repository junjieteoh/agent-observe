# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.7] - 2026-01-05

### Added
- **Wide Event trace capture**: Comprehensive trace model inspired by "Logging sucks" philosophy
  - `set_input()` / `set_output()`: Capture run-level input/output for complete context
  - Auto-inference: Input/output automatically inferred from spans if not explicitly set
  - `prompt_hash`: Auto-calculated hash of system prompt for version tracking

- **Run attribution fields**: Better context for debugging and analytics
  - `user_id`: Associate runs with users/accounts
  - `session_id`: Link runs within a conversation/session
  - `prompt_version`: Explicit prompt version (e.g., "v2.3")
  - `experiment_id`: A/B test cohort tracking
  - `model_config`: Model configuration (model name, temperature, etc.)
  - `metadata`: Custom metadata dictionary via `run.add_metadata()`

- **Full LLM context capture**: `@model_call` now captures complete LLM context
  - System prompt, full message history
  - Model configuration (temperature, max_tokens, etc.)
  - Tools/functions definitions
  - Stored in `llm_context` span attribute

- **SQLite schema migration**: Automatic migration from v1 to v2 schema
  - New columns added via ALTER TABLE for existing databases
  - No manual migration required

### Changed
- **Default mode changed**: `full` is now the default capture mode (was `metadata_only`)
  - Users install observability libraries because they want to see what happened
  - Use `mode="metadata_only"` for minimal storage if needed

- **Schema version bumped to 2**: New fields for Wide Event support
  - `user_id`, `session_id`, `prompt_version`, `prompt_hash`
  - `input_json`, `input_text`, `output_json`, `output_text`
  - `model_config`, `experiment_id`, `metadata`

### Migration from v0.1.6

**Behavior change**: Default mode is now `full` capture.

To keep old behavior:
```python
observe.install(mode="metadata_only")
```

**Schema changes**: New columns are nullable - no migration script needed. Old data has `NULL` for new fields.

## [0.1.6] - 2025-01-05

### Added
- **Sink health metrics**: `SinkMetrics` dataclass tracks writes_total, writes_failed, retries_total, write latency, and queue high watermark
- **Queue depth monitoring**: `sink.queue_depth` property exposes current backlog size
- **Health check API**: `observe.health()` returns system status (healthy/degraded/unhealthy) with queue depth and write metrics
- **Retry logic with exponential backoff**: Failed writes now retry up to 3 times with 1s, 2s, 4s backoff
- **Span memory management**: `RunContext` now limits in-memory spans (default 10,000) and flushes to sink when exceeded

### Changed
- **Improved error handling**: Silent policy check returns now log at DEBUG level for better debuggability
- **Safer cleanup on interrupts**: `run()` and `arun()` now catch `BaseException` to ensure cleanup on KeyboardInterrupt/SystemExit

## [0.1.5] - 2025-01-05

### Added
- **Per-run config overrides**: Override settings per-run instead of only at `install()` time
  - `mode`: Override capture mode for debugging specific runs - `observe.run("agent", mode="full")`
  - `policy_file`: Apply different policy rules per agent - `observe.run("agent", policy_file="strict.yml")`
  - `fail_on_violation`: Enable/disable strict mode per run - `observe.run("agent", fail_on_violation=True)`
  - `latency_budget_ms`: Different SLAs per agent - `observe.run("agent", latency_budget_ms=5000)`
- **RunContext accessor methods**: `get_capture_mode()`, `get_policy_engine()`, `get_fail_on_violation()`, `get_latency_budget_ms()` for consistent access to effective settings
- **Backward compatible**: All new parameters are optional with `None` defaults; existing code works unchanged

## [0.1.4] - 2025-01-05

### Added
- **Capture mode support in decorators**: `@tool` and `@model_call` now respect `capture_mode` setting
  - `full` mode: stores actual `args`/`result` (tools) and `input`/`output` (model calls)
  - `evidence_only` mode: stores content up to 64KB limit
  - `metadata_only` mode: stores only hashes (existing behavior)
- **pg_schema config**: PostgreSQL schema support - `Config(pg_schema="myschema")` or `AGENT_OBSERVE_PG_SCHEMA` env var (default: public)
- **Streaming LLM response support**: `@model_call` now detects and wraps streaming responses
  - Captures Time to First Token (TTFT) via `ts_first_token` attribute
  - Captures Time to Last Token via `ts_last_token` attribute
  - Records `chunk_count` for stream length
  - Accumulates output and stores based on `capture_mode`
  - Properly finalizes spans on stream errors (not just on exhaustion)
  - Works with OpenAI, Anthropic, and generic streaming formats
- **Enhanced error context**: Errors now include structured context based on `capture_mode`
  - `full` mode: includes error type, message, full traceback, and input that caused the error
  - `evidence_only` mode: includes error type, message, and truncated traceback (4KB limit)
  - `metadata_only` mode: includes only error type and message

### Changed
- Updated AGENTS.md to prefer `Config` object for configuration
- Error handling in `@tool` and `@model_call` now uses structured error context

## [0.1.3] - 2025-01-04

### Added
- **Examples folder**: Runnable scripts for basic usage, capture modes, policies, async, and querying
- **Policy roadmap**: Documented planned enhancements (argument-level policies, conditional policies, etc.)

## [0.1.2] - 2025-01-04

### Fixed
- **PostgreSQL executemany bug**: Fixed `cursor.executemany()` call (was incorrectly called on connection object)
- **Config accepts strings**: `Config(mode="full", env="dev")` now works (auto-converts to enums)

### Added
- **capture_mode stored in runs**: Each run now records which capture mode was used (full, metadata_only, etc.)
- **GIN index on eval_tags**: Fast JSONB containment queries for tag filtering
- **Partial index on parent_span_id**: Efficient hierarchical span lookups
- **Index on replay_cache.created_ts**: Enables TTL-based cache cleanup
- **CHECK constraint on risk_score**: Database-level validation (0-100 range)
- **Usage Guide documentation**: Comprehensive guide with policies, risk scoring, querying examples

### Changed
- PostgreSQL dependency will use `psycopg[binary]` to include precompiled binaries (fixes "libpq not found" errors)
- Updated AGENTS.md with schema design notes and rationale

## [0.1.1] - 2025-01-04

### Fixed
- **PostgreSQL span_id type mismatch**: Changed `span_id` and `parent_span_id` from UUID to TEXT for OpenTelemetry compatibility (OTEL uses 16-char hex IDs)
- **PostgreSQL permission handling**: Now gracefully detects pre-existing tables and skips schema creation when CREATE permission is missing

### Added
- **PostgreSQL retry logic**: Transient connection errors now retry with exponential backoff (3 attempts)
- **PostgreSQL batch inserts**: Uses `executemany()` for efficient bulk writes
- **PostgreSQL connection timeout**: 10-second timeout prevents hanging connections
- **Efficient table checks**: Single query to verify all required tables exist
- **Dynamic versioning**: Version is now read from package metadata across all modules
- **Unit tests for PostgreSQL sink**: New tests that don't require a real database

### Changed
- PostgreSQL dependency now uses `psycopg[binary]` to include precompiled binaries (fixes "no libpq" errors)
- Improved documentation with PostgreSQL best practices and manual schema SQL

## [0.1.0] - 2025-01-04

### Added

- Initial release of agent-observe
- Core observability runtime with `observe.install()` and `observe.run()` context manager
- `@tool` decorator for wrapping external tool calls with automatic tracing
- `@model_call` decorator for wrapping LLM/model invocations
- Policy engine with YAML-based configuration for allow/deny patterns
- Automatic risk scoring (0-100) based on behavioral signals
- Multiple sink backends:
  - SQLite sink for local development
  - PostgreSQL sink for production (via DATABASE_URL)
  - JSONL sink as fallback
  - OpenTelemetry (OTLP) sink for enterprise export
- Tool replay system for deterministic testing
- FastAPI-based local viewer for debugging agent runs
- CLI with `view` and `export-jsonl` commands
- Capture modes: `off`, `metadata_only`, `evidence_only`, `full`
- Enterprise-safe defaults (metadata-only capture)
- Comprehensive test suite
- Example agents demonstrating usage

[Unreleased]: https://github.com/junjieteoh/agent-observe/compare/v0.1.7...HEAD
[0.1.7]: https://github.com/junjieteoh/agent-observe/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/junjieteoh/agent-observe/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/junjieteoh/agent-observe/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/junjieteoh/agent-observe/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/junjieteoh/agent-observe/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/junjieteoh/agent-observe/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/junjieteoh/agent-observe/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/junjieteoh/agent-observe/releases/tag/v0.1.0
