# agent-observe

**Enterprise-grade observability for AI agents. Deploy with confidence. Improve continuously.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

You're deploying AI agents in production. But:

- **What is it doing?** You can't see inside the black box
- **Is it safe?** No way to enforce policies or block dangerous operations
- **Is it improving?** No systematic way to evaluate and iterate
- **Can you prove it?** No audit trail for compliance
- **What does it cost?** No visibility into token usage and spend

## The Solution

`agent-observe` is an **embeddable observability layer** for AI agents. Not a platform you connect to—a library you embed directly in your agent code.

```python
from agent_observe import observe, tool, model_call

observe.install()

@tool(name="query_database", kind="db")
def query_database(sql: str) -> list:
    return db.execute(sql)

with observe.run("my-agent", user_id="jane") as run:
    run.set_input(user_request)
    result = agent.run(user_request)
    run.set_output(result)

# Now you have: traces, policy enforcement, audit logs, extensibility hooks
```

---

## Why agent-observe?

### For Enterprise Deployment

| Challenge | How agent-observe Helps |
|-----------|-------------------------|
| **Visibility** | Full tracing of every tool call, LLM request, and decision |
| **Control** | Policy engine blocks dangerous operations before they execute |
| **Compliance** | Immutable audit trail with user attribution and timestamps |
| **Improvement** | Extensible hooks for eval, feedback, and iteration |
| **Cost Visibility** | Track token usage and spend via span attributes |

### vs. Platforms like Langfuse

| Aspect | Langfuse | agent-observe |
|--------|----------|---------------|
| **Architecture** | External SaaS platform | Embeddable library |
| **Data location** | Their cloud or self-hosted service | Your database (SQLite, Postgres, OTLP) |
| **Policy enforcement** | ❌ None | ✅ Block/allow rules, call limits |
| **Extensibility** | Limited | ✅ Full lifecycle hooks (v0.2) |
| **PII handling** | ❌ None | ✅ Pre-storage redaction (v0.2) |
| **Replay testing** | ❌ None | ✅ Deterministic replay |
| **UI/Dashboard** | ✅ Built-in | ❌ Bring your own (or use OTLP → Jaeger/Grafana) |

**Use Langfuse** if you want an all-in-one platform with UI, prompt management, and built-in evals.

**Use agent-observe** if you want:
- Full control over your data
- Policy enforcement and safety guardrails
- Extensibility to build your own workflows
- Lightweight library without external dependencies

## Installation

```bash
pip install agent-observe

# With PostgreSQL support
pip install agent-observe[postgres]

# With viewer UI
pip install agent-observe[viewer]
```

---

## Enterprise Deployment Lifecycle

agent-observe supports the full lifecycle of deploying and improving agents in enterprise settings:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 TRUSTABILITY + OBSERVABILITY FOR ENTERPRISE                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TRUSTABILITY: Can you trust the agent?                                    │
│   ──────────────────────────────────────                                    │
│      ├── Policy engine blocks dangerous operations before execution         │
│      ├── Call limits prevent runaway loops and cost explosions              │
│      ├── Approval workflows for high-risk actions (v0.2)                    │
│      ├── PII redaction before data leaves your control (v0.2)               │
│      └── Immutable audit trail proves what happened                         │
│                                                                             │
│   OBSERVABILITY: Can you see what's happening?                              │
│   ─────────────────────────────────────────────                             │
│      ├── Full traces: every tool call, model request, decision              │
│      ├── Lifecycle hooks: inject logic at any point (v0.2)                  │
│      ├── User & session attribution for multi-tenant systems                │
│      ├── Error tracking with full context                                   │
│      └── Export to any backend: Postgres, OTLP, Jaeger, Grafana             │
│                                                                             │
│   IMPROVEMENT: Can you make it better?                                      │
│   ────────────────────────────────────                                      │
│      ├── Hooks for eval, feedback, custom metrics (v0.2)                    │
│      ├── Replay testing for deterministic agent testing                     │
│      └── Query traces to find patterns and regressions                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```python
from agent_observe import observe, tool, model_call

# Initialize (zero-config, defaults to full capture)
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

# Run your agent with full context
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

View traces:
```bash
agent-observe view
# Open http://localhost:8765
```

## Documentation

| Document | Description |
|----------|-------------|
| **[Examples](examples/)** | Runnable code examples (basic usage, async, policies, hooks, PII) |
| **[Guide](docs/GUIDE.md)** | Data model, capture modes, policies, risk scoring, querying |
| **[Configuration](docs/CONFIGURATION.md)** | Environment variables and Config options |
| **[Patterns](docs/PATTERNS.md)** | Enterprise patterns and recipes |
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

See [Guide](docs/GUIDE.md) for details.

### Capture Modes

| Mode | What's Stored | Use Case |
|------|---------------|----------|
| `full` | Everything (default as of v0.1.7) | Development, debugging |
| `evidence_only` | Small content + hashes (64KB limit) | Production with audit needs |
| `metadata_only` | Hashes, timings only | High-security production |

**Default is `full`** as of v0.1.7 - you install observability because you want to see what happened.

For minimal storage: `observe.install(mode="metadata_only")`

See [Guide](docs/GUIDE.md#capture-modes) for details.

### Risk Scoring

Automatic risk scoring (0-100) based on:

| Signal | Weight |
|--------|--------|
| Policy violations | +40 |
| Tool success rate < 90% | +25 |
| Repeated tool calls (loops) | +15 |
| 5+ retries | +10 |
| Latency exceeds budget | +10 |

---

## How to Improve Your Agent

agent-observe provides the **data and hooks** you need to continuously improve your agents. Here's how:

### 1. Track Token Usage & Costs

Add token/cost attributes to any span:

```python
from agent_observe.context import get_current_span

# Inside your model call wrapper
response = openai.chat.completions.create(model="gpt-4", messages=messages)

span = get_current_span()
span.set_attribute("tokens_input", response.usage.prompt_tokens)
span.set_attribute("tokens_output", response.usage.completion_tokens)
span.set_attribute("cost_usd", calculate_cost(response.usage))  # Your pricing logic
```

### 2. Run Evaluations

Emit eval events to track quality:

```python
# After agent completes
score = my_evaluator(run.input, run.output)
observe.emit_event("eval", {
    "score": score.overall,
    "correctness": score.correctness,
    "helpfulness": score.helpfulness,
    "passed": score.overall > 0.7,
})
```

### 3. Collect User Feedback

Capture ratings and feedback:

```python
# When user provides feedback
observe.emit_event("feedback", {
    "rating": 5,
    "comment": "This was helpful!",
    "run_id": run.run_id,
})
```

### 4. Audit Data Access

Log sensitive operations:

```python
observe.emit_event("audit", {
    "action": "data_access",
    "resource": "users_table",
    "actor": run.user_id,
    "query": sanitized_query,
})
```

### 5. Use Lifecycle Hooks

Automate tasks with hooks that run at key points in the execution lifecycle:

```python
from agent_observe import observe, HookResult

# Block dangerous operations
@observe.hooks.before_tool
def security_check(ctx):
    if "DROP" in str(ctx.args).upper():
        return HookResult.block("SQL DROP statements are blocked")
    return HookResult.proceed()

# Auto-eval after each run
@observe.hooks.on_run_end
def auto_eval(ctx):
    if ctx.status == "ok":
        score = evaluate(ctx.run.output)
        observe.emit_event("eval", {"score": score})

# Track cost on every model call
@observe.hooks.after_model
def track_cost(ctx, result):
    cost = calculate_cost(result.usage)
    ctx.span.set_attribute("cost_usd", cost)
    return result

# Modify inputs before execution
@observe.hooks.before_tool
def sanitize_inputs(ctx):
    if ctx.tool_name == "search":
        cleaned_query = sanitize(ctx.args[0])
        return HookResult.modify(args=(cleaned_query,), kwargs=ctx.kwargs)
    return HookResult.proceed()
```

### 6. Circuit Breaker for Hook Resilience

Protect your agent from failing hooks with automatic circuit breakers:

```python
from agent_observe import observe, CircuitBreakerConfig

observe.install()

# Configure circuit breaker for hooks
observe.hooks.set_circuit_breaker(CircuitBreakerConfig(
    enabled=True,
    failure_threshold=5,    # Open circuit after 5 failures
    window_seconds=60,      # Within 60 seconds
    recovery_seconds=300,   # Try again after 5 minutes
))

@observe.hooks.before_tool
def flaky_external_check(ctx):
    # If this fails 5 times in 60s, it's automatically skipped
    # until the circuit breaker recovers
    return external_service.validate(ctx.tool_name)
```

---

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

## Policy Engine (Safety Guardrails)

Enterprises need guardrails. The policy engine lets you enforce rules **before execution**:

```yaml
# .riff/observe.policy.yml
tools:
  allow:
    - "db.read_*"      # Allow read operations
    - "http.get_*"     # Allow GET requests
  deny:
    - "shell.*"        # Block all shell commands
    - "db.drop_*"      # Block destructive DB ops
    - "*.delete"       # Block anything ending in delete

limits:
  max_tool_calls: 100   # Prevent infinite loops
  max_model_calls: 50   # Cap LLM spend
```

When a policy violation occurs:

```python
from agent_observe import PolicyViolationError

try:
    dangerous_tool()  # Blocked by policy
except PolicyViolationError as e:
    print(f"Blocked: {e.reason}")
    # Log for audit, alert security team, etc.
```

---

## Compliance & Audit

### User Attribution

Every run tracks who triggered it:

```python
with observe.run("agent", user_id="jane@company.com", session_id="conv_123"):
    # All spans and events are attributed to this user
```

### Immutable Audit Trail

All traces are stored with:
- Timestamps (ms precision)
- User ID
- Session ID
- Full input/output (configurable)
- Policy violations

### Query for Compliance

```sql
-- Find all runs by a specific user
SELECT * FROM runs WHERE user_id = 'jane@company.com';

-- Find all policy violations
SELECT * FROM spans WHERE violation_type IS NOT NULL;

-- Find all data access events
SELECT * FROM events WHERE event_type = 'audit';
```

### PII Handling

Automatically redact or hash PII before it's stored:

```python
from agent_observe import observe, PIIConfig

# Configure PII handling at install time
observe.install(
    pii=PIIConfig(
        enabled=True,
        action="redact",  # "redact", "hash", "tokenize", or "flag"
        patterns={
            "email": True,      # Built-in pattern
            "phone": True,      # Built-in pattern
            "ssn": True,        # Built-in pattern
            "credit_card": True,
            # Custom patterns
            "employee_id": r"EMP-\d{6}",
        },
    )
)

# All data is automatically processed before storage
with observe.run("support-agent", user_id="jane"):
    # Emails in tool args/results are redacted as [EMAIL_REDACTED]
    send_email("user@example.com", "Hello!")  # Stored as [EMAIL_REDACTED]
```

**PII Actions:**
| Action | Description |
|--------|-------------|
| `redact` | Replace with `[EMAIL_REDACTED]`, `[PHONE_REDACTED]`, etc. |
| `hash` | Replace with consistent hash: `[EMAIL:a1b2c3d4...]` |
| `tokenize` | Replace with reversible token (for later recovery) |
| `flag` | Keep original but mark as `[PII:email]user@example.com[/PII]` |

---

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

---

## Roadmap

### v0.1.x - Observability Foundation ✅
- Full tracing (tools, models, runs)
- Multiple sinks (SQLite, Postgres, JSONL, OTLP)
- Policy engine with allow/deny rules
- Risk scoring
- Replay mode for testing

### v0.2 - Trustability + Extensibility ✅
- Lifecycle hooks (before/after tool, model, run)
- Hook actions (block, skip, modify execution)
- Circuit breaker (auto-disable failing hooks)
- PII handling (redact/hash/tokenize before storage)

### v0.3 (Next) - Production Hardening
- 🔄 Approval workflows (pause for human approval)
- 📋 Enhanced policy engine (dynamic rules)
- 📋 Session analytics
- 📋 OpenTelemetry semantic conventions

---

## Philosophy

**We build what you can't do yourself.**

You CAN add cost tracking, run evals, collect feedback, log audits using our existing APIs (`set_attribute`, `emit_event`). So we don't build those as features—we give you the hooks to build them your way.

What you CAN'T do yourself:
- **Intercept before execution** → We provide `before_tool`, `before_model` hooks
- **Block or skip operations** → We provide `HookResult.block()`, `HookResult.skip()`
- **Redact PII before storage** → We provide pre-sink interception
- **Pause for approval** → We provide `HookResult.pending()`

This keeps the library lightweight and flexible while giving you the power to build exactly what your enterprise needs.

---

## License

MIT License - Use it, embed it, extend it. No restrictions.
