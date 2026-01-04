# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2025-01-04

### Added
- **Examples folder**: Runnable scripts for basic usage, capture modes, policies, async, and querying
- **Policy roadmap**: Documented planned enhancements (argument-level policies, conditional policies, etc.)

*No schema changes - safe upgrade from v0.1.2*

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

[Unreleased]: https://github.com/junjieteoh/agent-observe/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/junjieteoh/agent-observe/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/junjieteoh/agent-observe/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/junjieteoh/agent-observe/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/junjieteoh/agent-observe/releases/tag/v0.1.0
