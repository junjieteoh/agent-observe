# agent-observe

**Framework-agnostic observability, audit, and eval for AI agent applications.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is this?

`agent-observe` is a lightweight runtime layer that wraps your AI agent code to capture:

- **What tools were called** and when
- **What LLM calls were made** and how long they took
- **Policy violations** (blocked operations)
- **Risk scores** based on behavioral signals

Designed to be **enterprise-safe by default** - stores only metadata (hashes, sizes, timings), not raw content.

## Installation

```bash
pip install agent-observe

# With PostgreSQL support
pip install agent-observe[postgres]

# With viewer UI
pip install agent-observe[viewer]
```

## Quick Start

```python
from agent_observe import observe, tool, model_call

# Initialize (zero-config)
observe.install()

# Wrap your tools
@tool(name="search", kind="http")
def search_web(query: str) -> list:
    return requests.get(f"https://api.search.com?q={query}").json()

# Wrap your LLM calls
@model_call(provider="openai", model="gpt-4")
def call_llm(prompt: str) -> str:
    return openai.chat.completions.create(...).choices[0].message.content

# Run your agent
with observe.run("my-agent", task={"goal": "Research AI"}):
    results = search_web("AI agents")
    analysis = call_llm(f"Analyze: {results}")
```

View results:
```bash
agent-observe view
# Open http://localhost:8765
```

## Documentation

| Document | Description |
|----------|-------------|
| **[Examples](examples/)** | Runnable code examples (basic usage, async, policies) |
| **[Data Model](docs/DATA_MODEL.md)** | What are Runs, Spans, Events, and Replay Cache? |
| **[Capture Modes](docs/CAPTURE_MODES.md)** | What data is stored? Hashes vs full content |
| **[Configuration](docs/CONFIGURATION.md)** | Environment variables and Config options |
| **[Usage Guide](docs/USAGE_GUIDE.md)** | Policies, risk scoring, querying, real-world examples |
| **[Integration Guide](AGENTS.md)** | How to integrate with OpenAI, Anthropic, LangChain, etc. |

## Key Concepts

### Runs, Spans, and Events

```
┌─────────────────────────────────────────────────────────────┐
│                        observe.run()                         │
│                           (Run)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ @tool       │  │ @model_call │  │ emit_event  │          │
│  │  (Span)     │  │   (Span)    │  │  (Event)    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

- **Run** = One agent execution (start to finish)
- **Span** = One tool or model call within a run
- **Event** = Custom occurrence you emit

See [Data Model](docs/DATA_MODEL.md) for details.

### Capture Modes

| Mode | What's Stored | Use Case |
|------|---------------|----------|
| `metadata_only` | Hashes, timings | Production (default) |
| `evidence_only` | Small content + hashes | Debugging |
| `full` | Everything | Development |

**Default is `metadata_only`** - enterprise-safe, no PII leakage.

See [Capture Modes](docs/CAPTURE_MODES.md) for details.

### Risk Scoring

Automatic risk scoring (0-100) based on:

| Signal | Weight |
|--------|--------|
| Policy violations | +40 |
| Tool success rate < 90% | +25 |
| Repeated tool calls (loops) | +15 |
| 5+ retries | +10 |
| Latency exceeds budget | +10 |

## Configuration

### Zero-Config (Recommended)

```python
observe.install()  # Reads from environment variables
```

### Environment Variables

```bash
AGENT_OBSERVE_MODE=metadata_only    # Capture mode
AGENT_OBSERVE_ENV=prod              # Environment
DATABASE_URL=postgresql://...       # Enables Postgres sink
```

See [Configuration](docs/CONFIGURATION.md) for all options.

### Explicit Config

```python
from agent_observe.config import Config, CaptureMode, SinkType

config = Config(
    mode=CaptureMode.FULL,
    sink_type=SinkType.POSTGRES,
    database_url=os.environ.get("DATABASE_URL"),
)
observe.install(config=config)
```

## Sinks (Storage Backends)

| Sink | Use Case |
|------|----------|
| SQLite | Local development |
| PostgreSQL | Production |
| JSONL | Simple fallback |
| OTLP | OpenTelemetry export (Jaeger, Honeycomb, Datadog) |

Auto-selected based on available connections.

## Policy Engine

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
  max_model_calls: 50
```

## CLI

```bash
# Start viewer
agent-observe view

# Export to JSONL
agent-observe export-jsonl -o ./export
```

## Architecture

```
agent_observe/
├── observe.py      # Core runtime
├── decorators.py   # @tool, @model_call
├── policy.py       # YAML policy engine
├── metrics.py      # Risk scoring
├── replay.py       # Tool result caching
├── sinks/          # Storage backends
└── viewer/         # FastAPI UI
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT License
