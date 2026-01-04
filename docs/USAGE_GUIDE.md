# Usage Guide

This guide explains **why** each feature exists and **how** to use it effectively.

> **Looking for code examples?** See the [examples/](../examples/) folder for runnable scripts.

## Table of Contents

1. [Environments (dev/staging/prod)](#environments)
2. [Policies](#policies)
3. [Risk Scoring](#risk-scoring)
4. [Querying Your Data](#querying-your-data)
5. [Real-World Use Cases](#real-world-use-cases)

---

## Environments

### What is `env`?

The `env` setting tells agent-observe which environment your agent is running in:

```python
observe.install(env="dev")      # Development
observe.install(env="staging")  # Staging/QA
observe.install(env="prod")     # Production
```

### Why does it matter?

| Environment | Default Behavior |
|-------------|------------------|
| `dev` | Uses SQLite (local file), full capture mode |
| `staging` | Uses Postgres if available, evidence_only mode |
| `prod` | Uses Postgres, metadata_only mode (enterprise-safe) |

### How it affects sink selection

```python
# In dev: auto-selects SQLite
observe.install(env="dev")  # → SQLite at .riff/observe.db

# In prod with DATABASE_URL: auto-selects Postgres
os.environ["DATABASE_URL"] = "postgresql://..."
observe.install(env="prod")  # → PostgreSQL
```

### Filtering by environment

You can query runs by environment:

```sql
-- Find all production runs with high risk
SELECT * FROM runs
WHERE env = 'prod' AND risk_score > 50
ORDER BY ts_start DESC;

-- Compare dev vs prod behavior
SELECT env, COUNT(*), AVG(latency_ms)
FROM runs
GROUP BY env;
```

### Best practice

Set via environment variable so code doesn't change:

```bash
# In your deployment
export AGENT_OBSERVE_ENV=prod

# In your code (reads from env var automatically)
observe.install()
```

---

## Policies

### What are policies?

Policies define rules for what your agent can and cannot do. They're written in YAML.

> **Note**: The policy engine in v0.1.x is intentionally lightweight - it provides basic allow/deny patterns and rate limits. See [Policy Roadmap](#policy-roadmap) for planned enhancements.

### Why use policies?

1. **Prevent dangerous operations**: Block shell commands, file deletions
2. **Enforce compliance**: Only allow approved tools
3. **Rate limiting**: Prevent infinite loops
4. **Audit trail**: Record all violations

### Creating a policy file

Create `.riff/observe.policy.yml`:

```yaml
# Tool access control
tools:
  # Only these tools are allowed (whitelist)
  allow:
    - "db.query"
    - "db.read"
    - "http.get"
    - "http.post"
    - "calculator.*"    # Wildcard: any calculator tool

  # These tools are blocked (blacklist)
  deny:
    - "shell.*"         # Block all shell commands
    - "file.delete"     # Block file deletion
    - "db.drop"         # Block database drops
    - "*.destructive"   # Block anything ending in .destructive

# Rate limits
limits:
  max_tool_calls: 100    # Max tools per run
  max_model_calls: 50    # Max LLM calls per run
  max_retries: 10        # Max retries before flagging
```

### How policies work

```python
from agent_observe import observe, tool

observe.install(
    policy_file=".riff/observe.policy.yml",
    fail_on_violation=False,  # Log but don't block
)

@tool(name="shell.execute", kind="shell")
def run_shell(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True).stdout

with observe.run("my-agent"):
    # This will be BLOCKED by policy
    run_shell("rm -rf /")  # Policy violation recorded!
```

### Policy violation behavior

| Setting | Behavior |
|---------|----------|
| `fail_on_violation=False` | Log violation, let tool run, increment `policy_violations` |
| `fail_on_violation=True` | Raise `PolicyViolationError`, stop execution |

### Checking violations

```python
# After a run, check violations
run = observe.sink.get_run(run_id)
print(f"Violations: {run['policy_violations']}")
print(f"Risk score: {run['risk_score']}")  # Violations add +40 to risk
```

Or query directly:

```sql
-- Find runs with policy violations
SELECT run_id, name, policy_violations, risk_score
FROM runs
WHERE policy_violations > 0
ORDER BY policy_violations DESC;
```

### Real-world policy examples

**Enterprise compliance policy:**
```yaml
tools:
  allow:
    - "crm.*"           # CRM tools only
    - "email.send"
    - "calendar.*"
  deny:
    - "file.*"          # No file access
    - "shell.*"         # No shell access
    - "http.*"          # No arbitrary HTTP calls
limits:
  max_tool_calls: 50
  max_model_calls: 20
```

**Development policy (permissive):**
```yaml
tools:
  deny:
    - "*.production"    # Block production tools in dev
    - "db.drop"
limits:
  max_tool_calls: 1000  # Higher limits for testing
```

### Policy Roadmap

The current policy engine is lightweight by design. Here's what's planned:

| Version | Feature | Status |
|---------|---------|--------|
| v0.1.x | Allow/deny patterns with wildcards | ✅ Shipped |
| v0.1.x | Rate limits (max_tool_calls, max_model_calls) | ✅ Shipped |
| v0.1.x | Policy violation tracking | ✅ Shipped |
| v0.2 | **Argument-level policies** - Block tools based on argument values | Planned |
| v0.2 | **Conditional policies** - Different rules per environment | Planned |
| v0.2 | **Policy inheritance** - Base + override policies | Planned |
| v0.3 | **Runtime budgets** - Token/cost limits | Planned |
| v0.3 | **Policy versioning** - Track policy changes over time | Planned |
| v0.3 | **External policy sources** - Load from URL/API | Planned |

**Example of planned v0.2 features:**
```yaml
# Planned: Argument-level policies
tools:
  - name: "http.request"
    allow:
      args:
        url: "https://api.internal.company.com/*"
    deny:
      args:
        url: "http://*"  # Block non-HTTPS

# Planned: Conditional policies
environments:
  prod:
    deny: ["shell.*", "file.delete"]
  dev:
    deny: []  # Allow everything in dev
```

Want a feature sooner? [Open an issue](https://github.com/junjieteoh/agent-observe/issues).

---

## Risk Scoring

### What is risk scoring?

Every run gets an automatic risk score from 0-100 based on behavioral signals.

### How scores are calculated

| Signal | Points | Tag Added |
|--------|--------|-----------|
| Policy violation | +40 | `POLICY_VIOLATION` |
| Tool success rate < 90% | +25 | `TOOL_FAILURE` |
| Repeated identical tool calls | +15 | `LOOP_SUSPECTED` |
| 5+ retries | +10 | `RETRY_STORM` |
| Latency exceeds budget | +10 | `LATENCY_BREACH` |

### Example scenarios

**Normal run (score: 0)**
```
- 5 tool calls, all successful
- 2 model calls
- 3 seconds total
→ Risk score: 0, Tags: []
```

**Suspicious run (score: 55)**
```
- 10 tool calls, 2 failed
- Policy violation (tried shell command)
- Repeated same search 5 times
→ Risk score: 55, Tags: [POLICY_VIOLATION, TOOL_FAILURE, LOOP_SUSPECTED]
```

### Querying by risk

```sql
-- High-risk runs in the last 24 hours
SELECT run_id, name, risk_score, eval_tags
FROM runs
WHERE risk_score > 30
  AND ts_start > NOW() - INTERVAL '24 hours'
ORDER BY risk_score DESC;

-- Runs with specific issues
SELECT * FROM runs
WHERE eval_tags @> '["LOOP_SUSPECTED"]';
```

### Using risk scores

```python
# After a run
with observe.run("my-agent") as ctx:
    do_agent_work()

# Check the result
run = observe.sink.get_run(ctx.run_id)
if run["risk_score"] > 50:
    alert_team(f"High-risk run detected: {run['run_id']}")
```

---

## Querying Your Data

### Using the Viewer UI

```bash
agent-observe view
# Open http://localhost:8765
```

The viewer shows:
- List of all runs with status, risk score, timing
- Click a run to see all spans (tool/model calls)
- Filter by name, status, risk score
- View events and artifacts

### SQL Queries (PostgreSQL)

**Recent runs overview:**
```sql
SELECT
    run_id,
    name,
    status,
    risk_score,
    tool_calls,
    model_calls,
    latency_ms,
    ts_start
FROM runs
ORDER BY ts_start DESC
LIMIT 20;
```

**Failed runs with error details:**
```sql
SELECT
    r.run_id,
    r.name,
    s.name as failed_tool,
    s.error_message
FROM runs r
JOIN spans s ON r.run_id = s.run_id
WHERE r.status = 'error'
  AND s.status = 'error'
ORDER BY r.ts_start DESC;
```

**Slowest tools:**
```sql
SELECT
    name,
    AVG(EXTRACT(EPOCH FROM (ts_end - ts_start)) * 1000) as avg_ms,
    COUNT(*) as call_count
FROM spans
WHERE kind = 'tool'
GROUP BY name
ORDER BY avg_ms DESC
LIMIT 10;
```

**Tool usage patterns:**
```sql
SELECT
    name,
    COUNT(*) as calls,
    SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) as success,
    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
FROM spans
WHERE kind = 'tool'
GROUP BY name
ORDER BY calls DESC;
```

**Runs by project/environment:**
```sql
SELECT
    project,
    env,
    COUNT(*) as runs,
    AVG(risk_score) as avg_risk,
    AVG(latency_ms) as avg_latency
FROM runs
GROUP BY project, env
ORDER BY runs DESC;
```

### Python API

```python
from agent_observe import observe

# Get recent runs
runs = observe.sink.get_runs(limit=10)
for run in runs:
    print(f"{run['name']}: {run['status']} (risk: {run['risk_score']})")

# Get details for a specific run
run = observe.sink.get_run("run-id-here")
spans = observe.sink.get_spans(run["run_id"])
events = observe.sink.get_events(run["run_id"])

# Filter runs
failed_runs = observe.sink.get_runs(status="error", limit=100)
high_risk = observe.sink.get_runs(min_risk_score=50)
```

### Exporting data

```bash
# Export to JSONL files
agent-observe export-jsonl -o ./export/

# Creates:
# ./export/runs.jsonl
# ./export/spans.jsonl
# ./export/events.jsonl
```

---

## Real-World Use Cases

### 1. Debugging a Failing Agent

**Problem**: Agent sometimes fails, need to understand why.

```python
# Enable full capture for debugging
observe.install(mode="full", env="dev")

with observe.run("debug-agent") as ctx:
    try:
        result = run_agent(query)
    except Exception as e:
        print(f"Failed! Run ID: {ctx.run_id}")
        raise

# Now query the run
run = observe.sink.get_run(ctx.run_id)
spans = observe.sink.get_spans(ctx.run_id)

for span in spans:
    print(f"{span['name']}: {span['status']}")
    if span['status'] == 'error':
        print(f"  Error: {span['error_message']}")
        print(f"  Attrs: {span['attrs']}")  # Full content in 'full' mode
```

### 2. Compliance Audit

**Problem**: Need to prove what the agent did for auditing.

```sql
-- Audit report: What did agent X do on date Y?
SELECT
    r.run_id,
    r.ts_start,
    r.ts_end,
    s.name as tool,
    s.ts_start as tool_called_at,
    s.status,
    r.policy_violations
FROM runs r
JOIN spans s ON r.run_id = s.run_id
WHERE r.name = 'customer-service-agent'
  AND r.ts_start BETWEEN '2024-01-01' AND '2024-01-02'
ORDER BY s.ts_start;
```

### 3. Performance Monitoring

**Problem**: Agent is slow, need to find bottleneck.

```sql
-- Find slowest operations
SELECT
    name,
    kind,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
        EXTRACT(EPOCH FROM (ts_end - ts_start)) * 1000
    ) as p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY
        EXTRACT(EPOCH FROM (ts_end - ts_start)) * 1000
    ) as p95_ms
FROM spans
WHERE ts_start > NOW() - INTERVAL '7 days'
GROUP BY name, kind
ORDER BY p95_ms DESC;
```

### 4. Detecting Anomalies

**Problem**: Need to catch unusual agent behavior.

```python
# Alert on high-risk runs
def run_with_monitoring(query: str):
    with observe.run("monitored-agent") as ctx:
        result = run_agent(query)

    run = observe.sink.get_run(ctx.run_id)

    if run["risk_score"] > 50:
        send_alert(f"High risk: {run['risk_score']}", run)

    if run["policy_violations"] > 0:
        send_alert(f"Policy violations: {run['policy_violations']}", run)

    if run["latency_ms"] > 30000:
        send_alert(f"Slow run: {run['latency_ms']}ms", run)

    return result
```

### 5. A/B Testing Agent Versions

**Problem**: Comparing two agent implementations.

```python
import random

# Tag runs with version
observe.install(agent_version="v2.1.0")

# Later, compare versions
```

```sql
SELECT
    agent_version,
    COUNT(*) as runs,
    AVG(risk_score) as avg_risk,
    AVG(latency_ms) as avg_latency,
    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)::float / COUNT(*) as error_rate
FROM runs
WHERE agent_version IN ('v2.0.0', 'v2.1.0')
  AND ts_start > NOW() - INTERVAL '7 days'
GROUP BY agent_version;
```

### 6. Deterministic Testing with Replay

**Problem**: Tests are flaky because they hit real APIs.

```python
# Step 1: Record golden responses
observe.install(mode="full", replay="write")

with observe.run("record-test"):
    result = my_agent("What is 2+2?")
    assert "4" in result

# Step 2: Run tests with cached responses
observe.install(replay="read")

with observe.run("replay-test"):
    # This uses cached responses - no API calls!
    result = my_agent("What is 2+2?")
    assert "4" in result  # Same result, deterministic
```

---

## Summary

| Feature | Why It Exists | How to Use It |
|---------|---------------|---------------|
| `env` | Different behavior for dev/staging/prod | `observe.install(env="prod")` |
| Policies | Control what agents can do | Create `.riff/observe.policy.yml` |
| Risk scoring | Automatically flag suspicious runs | Query `risk_score` and `eval_tags` |
| Spans | Track every tool/model call | Query `spans` table |
| Events | Track custom occurrences | `observe.emit_event()` |
| Replay | Deterministic testing | `observe.install(replay="write")` |
