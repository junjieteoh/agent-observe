# agent-observe Guide

Comprehensive guide to using `agent-observe` effectively.

> **Looking for code examples?** See the [examples/](../examples/) folder for runnable scripts.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Data Model](#data-model)
3. [Capture Modes](#capture-modes)
4. [Environments](#environments)
5. [Policies](#policies)
6. [Risk Scoring](#risk-scoring)
7. [Querying Your Data](#querying-your-data)
8. [Real-World Use Cases](#real-world-use-cases)

---

## Quick Reference

| Concept | What It Does |
|---------|--------------|
| **Run** | One agent execution (start to finish) |
| **Span** | One tool or model call within a run |
| **Event** | Custom occurrence you emit |
| **Capture Mode** | Controls what data is stored |
| **Policy** | Rules for what agents can/cannot do |
| **Risk Score** | Automatic 0-100 score based on behavior |

---

## Data Model

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

### Runs

A **Run** represents a single agent execution from start to finish.

```python
with observe.run("my-agent", user_id="jane", session_id="conv_123"):
    search_web("AI news")
    call_llm("Summarize findings")
```

**Key Run Fields:**

| Field | Description |
|-------|-------------|
| `run_id` | Unique identifier |
| `name` | Agent name you provided |
| `status` | `ok`, `error`, or `blocked` |
| `risk_score` | Automatic score 0-100 |
| `user_id` | User attribution |
| `session_id` | Conversation/session linking |
| `input_json/text` | What user asked |
| `output_json/text` | Agent's response |

### Spans

A **Span** represents a single operation within a run.

```python
@tool(name="search", kind="http")
def search_web(query: str) -> list:  # Creates a span each time
    return requests.get(f"https://api.search.com?q={query}").json()
```

**Key Span Fields:**

| Field | Description |
|-------|-------------|
| `span_id` | Unique identifier (16-char hex, OTEL compatible) |
| `kind` | `tool` or `model` |
| `name` | Tool/model name from decorator |
| `status` | `ok`, `error`, or `blocked` |
| `attrs` | Operation-specific attributes |

### Events

An **Event** is a custom occurrence you emit during a run.

```python
observe.emit_event("user.query", {"query": "What is AI?"})
observe.emit_artifact("report", {"content": "..."})
```

### Replay Cache

Stores tool results for deterministic testing:

```python
# Record results
observe.install(replay="write")
result = search_web("AI news")  # Executed and cached

# Replay cached results
observe.install(replay="read")
result = search_web("AI news")  # Returns cached result (no execution)
```

---

## Capture Modes

Controls what data is stored:

| Mode | What's Stored | Use Case |
|------|---------------|----------|
| `off` | Nothing | Disable observability |
| `metadata_only` | Hashes, timings, counts | Production (default) |
| `evidence_only` | Small content + hashes | Debugging |
| `full` | Everything | Development/testing |

### Setting the Capture Mode

```python
# Via environment variable
export AGENT_OBSERVE_MODE=full

# Or in code
observe.install(mode="full")
```

### `metadata_only` - Enterprise-Safe (Default)

Stores **hashes and metadata only**, not actual content:

```json
{
  "args_hash": "4758640e...",
  "result_hash": "be42ae69...",
  "result_size": 2048
}
```

**What you CAN audit:**
- What tools were called and in what order
- Success/failure status
- Duration of each operation
- Whether same inputs were used (matching hashes)
- Policy violations and risk scores

**What you CANNOT see:**
- Actual input arguments
- Actual output content

### `evidence_only` - Partial Content

Stores small content directly, hashes large content:

| Content Size | What's Stored |
|--------------|---------------|
| < 1 KB | Full content |
| >= 1 KB | Hash + size |

### `full` - Everything

Stores all content (with size caps: 100KB args, 1MB results).

### Recommendations

| Environment | Recommended Mode |
|-------------|------------------|
| Production | `metadata_only` |
| Staging | `evidence_only` |
| Development | `full` |
| CI/CD Tests | `full` |
| Load Testing | `off` |

---

## Environments

The `env` setting tells agent-observe which environment your agent is running in:

```python
observe.install(env="dev")      # Development
observe.install(env="staging")  # Staging/QA
observe.install(env="prod")     # Production
```

### How it affects behavior

| Environment | Default Behavior |
|-------------|------------------|
| `dev` | Uses SQLite (local file) |
| `staging` | Uses Postgres if available |
| `prod` | Uses Postgres |

### Best practice

Set via environment variable:

```bash
export AGENT_OBSERVE_ENV=prod
```

---

## Policies

Policies define rules for what your agent can and cannot do.

### Creating a policy file

Create `.riff/observe.policy.yml`:

```yaml
tools:
  allow:
    - "db.query"
    - "db.read"
    - "calculator.*"    # Wildcard

  deny:
    - "shell.*"         # Block all shell commands
    - "file.delete"
    - "db.drop"

limits:
  max_tool_calls: 100
  max_model_calls: 50
  max_retries: 10
```

### Using policies

```python
observe.install(
    policy_file=".riff/observe.policy.yml",
    fail_on_violation=False,  # Log but don't block
)
```

### Policy violation behavior

| Setting | Behavior |
|---------|----------|
| `fail_on_violation=False` | Log violation, let tool run |
| `fail_on_violation=True` | Raise `PolicyViolationError`, stop execution |

### Checking violations

```python
run = observe.sink.get_run(run_id)
print(f"Violations: {run['policy_violations']}")
print(f"Risk score: {run['risk_score']}")  # Violations add +40
```

---

## Risk Scoring

Every run gets an automatic risk score from 0-100:

| Signal | Points |
|--------|--------|
| Policy violation | +40 |
| Tool success rate < 90% | +25 |
| Repeated identical tool calls | +15 |
| 5+ retries | +10 |
| Latency exceeds budget | +10 |

### Example scenarios

**Normal run (score: 0)**
- 5 tool calls, all successful
- 2 model calls
- 3 seconds total

**Suspicious run (score: 55)**
- 10 tool calls, 2 failed
- Policy violation (tried shell command)
- Repeated same search 5 times

### Querying by risk

```sql
SELECT run_id, name, risk_score, eval_tags
FROM runs
WHERE risk_score > 30
ORDER BY risk_score DESC;
```

---

## Querying Your Data

### Using the Viewer UI

```bash
agent-observe view
# Open http://localhost:8765
```

### SQL Queries

**Recent runs:**
```sql
SELECT run_id, name, status, risk_score, latency_ms
FROM runs
ORDER BY ts_start DESC
LIMIT 20;
```

**Failed runs with errors:**
```sql
SELECT r.run_id, r.name, s.name as failed_tool, s.error_message
FROM runs r
JOIN spans s ON r.run_id = s.run_id
WHERE r.status = 'error' AND s.status = 'error';
```

**Slowest tools:**
```sql
SELECT name, AVG(EXTRACT(EPOCH FROM (ts_end - ts_start)) * 1000) as avg_ms
FROM spans
WHERE kind = 'tool'
GROUP BY name
ORDER BY avg_ms DESC;
```

### Python API

```python
# Get recent runs
runs = observe.sink.get_runs(limit=10)

# Get details for a specific run
run = observe.sink.get_run("run-id")
spans = observe.sink.get_spans(run["run_id"])
events = observe.sink.get_events(run["run_id"])
```

### Exporting data

```bash
agent-observe export-jsonl -o ./export/
```

---

## Real-World Use Cases

### 1. Debugging a Failing Agent

```python
observe.install(mode="full", env="dev")

with observe.run("debug-agent") as ctx:
    try:
        result = run_agent(query)
    except Exception as e:
        print(f"Failed! Run ID: {ctx.run_id}")
        raise

# Query the run
spans = observe.sink.get_spans(ctx.run_id)
for span in spans:
    if span['status'] == 'error':
        print(f"Error in {span['name']}: {span['error_message']}")
```

### 2. Compliance Audit

```sql
SELECT r.run_id, r.ts_start, s.name as tool, s.status, r.policy_violations
FROM runs r
JOIN spans s ON r.run_id = s.run_id
WHERE r.name = 'customer-service-agent'
  AND r.ts_start BETWEEN '2024-01-01' AND '2024-01-02'
ORDER BY s.ts_start;
```

### 3. Performance Monitoring

```sql
SELECT name, kind,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY
    EXTRACT(EPOCH FROM (ts_end - ts_start)) * 1000
  ) as p95_ms
FROM spans
WHERE ts_start > NOW() - INTERVAL '7 days'
GROUP BY name, kind
ORDER BY p95_ms DESC;
```

### 4. Detecting Anomalies

```python
def run_with_monitoring(query: str):
    with observe.run("monitored-agent") as ctx:
        result = run_agent(query)

    run = observe.sink.get_run(ctx.run_id)

    if run["risk_score"] > 50:
        send_alert(f"High risk: {run['risk_score']}")

    if run["policy_violations"] > 0:
        send_alert(f"Policy violations: {run['policy_violations']}")

    return result
```

### 5. Deterministic Testing with Replay

```python
# Record golden responses
observe.install(mode="full", replay="write")

with observe.run("record-test"):
    result = my_agent("What is 2+2?")
    assert "4" in result

# Run tests with cached responses
observe.install(replay="read")

with observe.run("replay-test"):
    result = my_agent("What is 2+2?")  # No API calls!
    assert "4" in result
```

---

## Next Steps

- [Configuration Reference](CONFIGURATION.md) - All config options
- [Patterns & Recipes](PATTERNS.md) - Enterprise patterns
- [Examples](../examples/) - Runnable code
