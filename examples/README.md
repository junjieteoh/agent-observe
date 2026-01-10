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
| [wide_event_traces.py](wide_event_traces.py) | Wide event traces with attribution |

## v0.2 Features

| Example | Description |
|---------|-------------|
| [hooks_example.py](hooks_example.py) | Lifecycle hooks for security, caching, cost tracking |
| [pii_handling.py](pii_handling.py) | PII redaction, hashing, tokenization, and flagging |

## Real-World Examples

Production-ready patterns for common agent architectures:

| Example | Description |
|---------|-------------|
| [customer_support_agent.py](customer_support_agent.py) | Full support workflow: customer lookup, ticketing, email, PII protection, audit trail |
| [rag_agent_with_budget.py](rag_agent_with_budget.py) | RAG pipeline with budget enforcement, semantic caching, cost tracking, circuit breaker |

### Customer Support Agent

Demonstrates a realistic customer support workflow:
- Database lookups (customer, orders, tickets)
- Ticket creation and escalation
- Email notifications
- PII auto-redaction (emails, phone numbers)
- Security hooks (block dangerous operations)
- LLM cost tracking
- Audit events for compliance

```bash
python examples/customer_support_agent.py
```

### RAG Agent with Budget

Demonstrates a production RAG system:
- Query rewriting for better retrieval
- Vector database search (simulated)
- Context assembly and answer generation
- **Budget enforcement** - stops before overspending
- **Semantic caching** - skip LLM for repeated queries
- **Circuit breaker** - auto-disable failing hooks
- Per-query cost tracking

```bash
python examples/rag_agent_with_budget.py
```

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

## More Resources

- [Guide](../docs/GUIDE.md) - Data model, capture modes, policies, risk scoring
- [Patterns & Recipes](../docs/PATTERNS.md) - Real-world patterns for production use
- [Integration Guide](../AGENTS.md) - Integrate with OpenAI, Anthropic, LangChain, etc.
- [Configuration](../docs/CONFIGURATION.md) - All configuration options
