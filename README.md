# agent-observe

**Framework-agnostic observability, audit, and eval for AI agent applications.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is this?

`agent-observe` is a lightweight runtime layer that wraps your AI agent code to capture:

- **What tools were called** and when
- **What LLM calls were made** and how long they took
- **Full LLM context** - system prompts, message history, tool definitions (v0.1.7+)
- **Run input/output** - what the user asked, what the agent responded (v0.1.7+)
- **Session continuity** - link runs in a conversation (v0.1.7+)
- **Policy violations** (blocked operations)
- **Risk scores** based on behavioral signals

As of v0.1.7, default mode is **full capture** - stores complete traces for debugging and audit.

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

# Initialize (zero-config, defaults to full capture as of v0.1.7)
observe.install()

# Wrap your tools
@tool(name="search", kind="http")
def search_web(query: str) -> list:
    return requests.get(f"https://api.search.com?q={query}").json()

# Wrap your LLM calls
@model_call(provider="openai", model="gpt-4")
def call_llm(messages: list) -> str:
    return openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
    ).choices[0].message.content

# Run your agent with context (v0.1.7+)
with observe.run(
    "my-agent",
    user_id="jane",              # Who triggered this?
    session_id="conv_123",       # Part of which conversation?
) as run:
    run.set_input("Research AI agents")  # Capture user request

    results = search_web("AI agents")
    analysis = call_llm([
        {"role": "system", "content": "You are a research assistant"},
        {"role": "user", "content": f"Analyze: {results}"},
    ])

    run.set_output(analysis)  # Capture final response
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
| `full` | Everything (default as of v0.1.7) | Development, debugging |
| `evidence_only` | Small content + hashes (64KB limit) | Production with audit needs |
| `metadata_only` | Hashes, timings only | High-security production |

**Default is `full`** as of v0.1.7 - you install observability because you want to see what happened.

For minimal storage: `observe.install(mode="metadata_only")`

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
AGENT_OBSERVE_MODE=full             # Capture mode (default: full as of v0.1.7)
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
