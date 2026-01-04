# agent-observe

**Framework-agnostic observability, audit, and eval for AI agent applications.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`agent-observe` is a lightweight runtime layer that provides:

- **Observability** - Track agent runs, tool calls, and model invocations
- **Audit/Compliance** - Policy engine with deny/allow patterns for tools
- **Label-free Eval** - Automatic risk scoring based on behavioral signals
- **Tool Replay** - Cache tool results for deterministic testing
- **Local Viewer** - FastAPI UI for browsing and debugging runs

Designed to be **enterprise-safe by default** - stores only metadata (hashes, sizes, timings), not raw content.

## Installation

```bash
# Core package
pip install agent-observe

# With viewer UI
pip install agent-observe[viewer]

# With PostgreSQL support
pip install agent-observe[postgres]

# All extras
pip install agent-observe[all]
```

## Quick Start

```python
from agent_observe import observe, tool, model_call

# Initialize (zero-config, auto-detects environment)
observe.install()

# Define tools
@tool(name="search", kind="http")
def search_web(query: str) -> list[dict]:
    # Your implementation
    return [{"title": "Result", "url": "https://..."}]

@model_call(provider="openai", model="gpt-4")
def call_llm(prompt: str) -> str:
    # Your LLM call
    return "Response..."

# Run your agent
with observe.run("my-agent", task={"goal": "Research topic"}):
    results = search_web("AI agents")
    analysis = call_llm(f"Analyze: {results}")
    observe.emit_artifact("analysis", analysis)
```

View the results:

```bash
agent-observe view
```

### Async Support

Full async/await support for modern agent frameworks:

```python
@tool(name="fetch_data", kind="http")
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        return await client.get(url)

@model_call(provider="anthropic", model="claude-3")
async def call_claude(prompt: str) -> str:
    return await anthropic.messages.create(...)

# Use async context manager
async with observe.arun("async-agent"):
    data = await fetch_data("https://api.example.com")
    response = await call_claude(f"Analyze: {data}")
```

## Framework Integration

See **[AGENTS.md](AGENTS.md)** for detailed integration examples with:
- OpenAI Function Calling
- Anthropic Claude
- Google Vertex AI / Gemini
- LangChain
- Custom ReAct agents

## Features

### Zero-Config Defaults

Just call `observe.install()` - it automatically:
- Selects the right sink based on environment
- Uses SQLite for local dev, Postgres if `DATABASE_URL` is set
- Captures metadata only (enterprise-safe)

### Automatic Sink Selection

| Condition | Sink |
|-----------|------|
| `DATABASE_URL` set | PostgreSQL |
| `OTEL_EXPORTER_OTLP_ENDPOINT` set | OTLP (OpenTelemetry) |
| `AGENT_OBSERVE_ENV=dev` | SQLite |
| Default | JSONL |

### Policy Engine

Create `.riff/observe.policy.yml`:

```yaml
tools:
  allow:
    - "db.*"
    - "http.*"
  deny:
    - "shell.*"
    - "*.destructive"

limits:
  max_tool_calls: 100
  max_retries: 10
  max_model_calls: 50
```

> **Coming Soon:** SQL query validation and network domain restrictions are planned for a future release.

### Risk Scoring

Automatic risk scoring (0-100) based on:

| Signal | Weight | Tag |
|--------|--------|-----|
| Policy violations | +40 | `POLICY_VIOLATION` |
| Tool success rate < 90% | +25 | `TOOL_FAILURE` |
| Repeated tool calls (loops) | +15 | `LOOP_SUSPECTED` |
| 5+ retries | +10 | `RETRY_STORM` |
| Latency exceeds budget | +10 | `LATENCY_BREACH` |

### Capture Modes

| Mode | Description |
|------|-------------|
| `off` | Disable observability |
| `metadata_only` | Store hashes, sizes, timings only (default) |
| `evidence_only` | Store small blobs with redaction |
| `full` | Store all content (with caps) |

## Environment Variables

```bash
# Core
AGENT_OBSERVE_MODE=metadata_only    # off|metadata_only|evidence_only|full
AGENT_OBSERVE_ENV=prod              # dev|staging|prod
AGENT_OBSERVE_PROJECT=my-app        # Project name
AGENT_OBSERVE_AGENT_VERSION=1.0.0   # Agent version

# Sink selection
AGENT_OBSERVE_SINK=auto             # auto|sqlite|jsonl|postgres|otlp
DATABASE_URL=postgresql://...     # Enables Postgres sink

# Policy
AGENT_OBSERVE_POLICY_FILE=.riff/observe.policy.yml
AGENT_OBSERVE_FAIL_ON_VIOLATION=0   # 1 to raise on violations

# Replay
AGENT_OBSERVE_REPLAY=off            # off|write|read

# Performance
AGENT_OBSERVE_LATENCY_BUDGET_MS=20000
```

## CLI

```bash
# Start the viewer
agent-observe view
agent-observe view --port 8080

# Export to JSONL
agent-observe export-jsonl -o ./export

# With specific database
agent-observe view --db .riff/observe.db
agent-observe view --database-url postgresql://...
```

## API Reference

### Core

```python
from agent_observe import observe

# Initialize
observe.install(mode="metadata_only")

# Create a run context
with observe.run("agent-name", task={"goal": "..."}) as run:
    pass

# Emit events
observe.emit_event("custom.event", {"key": "value"})

# Emit artifacts
observe.emit_artifact("report", {"data": "..."}, provenance=["tool1", "tool2"])
```

### Decorators

```python
from agent_observe import tool, model_call

@tool(name="my_tool", kind="db", version="1")
def my_tool(arg: str) -> dict:
    pass

@model_call(provider="openai", model="gpt-4")
def call_model(prompt: str) -> str:
    pass
```

## Architecture

```
agent_observe/
├── observe.py      # Core runtime (install, run, emit_*)
├── decorators.py   # @tool, @model_call (sync and async)
├── policy.py       # YAML policy engine
├── metrics.py      # Risk scoring and eval
├── replay.py       # Tool result caching
├── sinks/
│   ├── sqlite_sink.py   # Local dev
│   ├── jsonl_sink.py    # Fallback
│   ├── postgres_sink.py # Production
│   └── otel_sink.py     # OTLP export (Jaeger, Honeycomb, Datadog, etc.)
└── viewer/
    └── app.py      # FastAPI viewer
```

## Roadmap

- [ ] Auto-instrumentation for OpenAI SDK
- [ ] Auto-instrumentation for Anthropic SDK
- [ ] SQL query validation policies
- [ ] Network domain restriction policies
- [ ] Streaming support for LLM responses
- [ ] Sampling for high-volume production

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Type checking
mypy agent_observe
```

## License

MIT License - see LICENSE file for details.
