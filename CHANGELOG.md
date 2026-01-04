# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/junjieteoh/agent-observe/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/junjieteoh/agent-observe/releases/tag/v0.1.0
