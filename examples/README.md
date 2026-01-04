# Examples

Practical examples for using `agent-observe`.

## Quick Start

| Example | Description |
|---------|-------------|
| [basic_usage.py](basic_usage.py) | Minimal setup with `@tool` and `@model_call` |
| [capture_modes.py](capture_modes.py) | Different capture modes (full, metadata_only, etc.) |
| [with_policy.py](with_policy.py) | Policy engine with allow/deny patterns |
| [async_agent.py](async_agent.py) | Async/await usage with `observe.arun()` |
| [query_runs.py](query_runs.py) | Query stored runs, spans, and events |

## Running Examples

```bash
# Install agent-observe
pip install agent-observe

# Run any example
python examples/basic_usage.py

# View results
agent-observe view
```

## Configuration

All examples read from environment variables by default:

```bash
# Capture mode (off, metadata_only, evidence_only, full)
export AGENT_OBSERVE_MODE=full

# Environment (dev, staging, prod)
export AGENT_OBSERVE_ENV=dev

# PostgreSQL (optional, uses SQLite by default)
export DATABASE_URL=postgresql://user:pass@host/db
```

Or configure explicitly in code:

```python
from agent_observe.config import Config

config = Config(mode="full", env="dev")
observe.install(config=config)
```
