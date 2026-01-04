# Capture Modes

`agent-observe` has four capture modes that control what data is stored. This document explains each mode, what gets captured, and when to use it.

## Quick Reference

| Mode | What's Stored | Use Case |
|------|---------------|----------|
| `off` | Nothing | Disable observability |
| `metadata_only` | Hashes, timings, counts | Production (default) |
| `evidence_only` | Small content + hashes | Debugging |
| `full` | Everything | Development/testing |

## Setting the Capture Mode

```python
# Via environment variable
export AGENT_OBSERVE_MODE=metadata_only

# Or in code
observe.install(mode="metadata_only")

# Or with explicit config
from agent_observe.config import Config, CaptureMode

config = Config(mode=CaptureMode.METADATA_ONLY)
observe.install(config=config)
```

---

## `off` - Disabled

No data is captured. The decorators still work (your code runs normally), but nothing is stored.

**When to use:**
- Performance-critical production paths
- When observability is handled elsewhere
- Temporarily disabling for specific runs

```python
observe.install(mode="off")

with observe.run("my-agent"):
    search_web("query")  # Runs, but nothing recorded
```

---

## `metadata_only` - Default (Enterprise-Safe)

Stores **hashes and metadata only**, not actual content.

### What gets stored:

```json
{
  "args_hash": "4758640e48886c0cfdb6eb3ec4b3f68197a2d3e063c0711511aa0736a5672b36",
  "result_hash": "be42ae699d507351e0c50c09f81e7b2b8393820ead040aff232b7d140d4e1fa8",
  "result_size": 2048,
  "tool.kind": "http",
  "tool.version": "1",
  "replay.hit": false
}
```

### What you CAN audit:

| Question | How to Answer |
|----------|---------------|
| What tools were called? | `spans` table shows all calls |
| In what order? | Sort by `ts_start` |
| Did they succeed? | Check `status` field |
| How long did each take? | `ts_end - ts_start` |
| Were same inputs used? | Same `args_hash` = same inputs |
| Any policy violations? | `runs.policy_violations` count |
| What's the risk level? | `runs.risk_score` (0-100) |

### What you CANNOT see:

- Actual input arguments
- Actual output content
- Specific error details in args/results

### Why use this mode?

1. **No PII leakage**: User data, passwords, API keys can't leak
2. **Compliance-friendly**: Many regulations require minimizing stored data
3. **Smaller storage**: Hashes are fixed 64 chars vs variable content
4. **Still auditable**: Prove WHAT happened without storing HOW

### Example audit scenario:

> "Did Agent X call the delete API on Tuesday?"

With `metadata_only`:
```sql
SELECT name, ts_start, status
FROM spans
WHERE name = 'delete_record'
  AND ts_start BETWEEN '2024-01-02' AND '2024-01-03';
```

You can prove the call happened, when, and if it succeeded - without storing what was deleted.

---

## `evidence_only` - Partial Content

Stores **small content directly**, hashes large content.

### What gets stored:

```json
{
  "args": {"query": "weather in NYC", "limit": 10},
  "args_size": 42,
  "result_hash": "be42ae699d507351e0c50c09f81e7b2b8393820ead...",
  "result_size": 15360,
  "tool.kind": "http"
}
```

### Rules:

| Content Size | What's Stored |
|--------------|---------------|
| < 1 KB | Full content |
| >= 1 KB | Hash + size |

### Why use this mode?

1. **Debugging**: See small inputs/outputs directly
2. **Limited exposure**: Large payloads (files, API responses) still hashed
3. **Balance**: More visibility than `metadata_only`, safer than `full`

### Example:

```python
@tool(name="search")
def search(query: str) -> list:  # query is small, stored directly
    return big_api_response       # response is large, hashed
```

Stored:
```json
{
  "args": {"query": "AI news"},
  "result_hash": "abc123...",
  "result_size": 50000
}
```

---

## `full` - Everything

Stores **all content** (with size caps to prevent abuse).

### What gets stored:

```json
{
  "args": {"query": "weather in NYC", "limit": 10},
  "result": {
    "temperature": 72,
    "conditions": "sunny",
    "forecast": [...]
  },
  "tool.kind": "http"
}
```

### Size caps:

| Content | Max Size |
|---------|----------|
| Arguments | 100 KB |
| Results | 1 MB |

Content exceeding caps is truncated with a marker.

### Why use this mode?

1. **Development**: Full visibility for debugging
2. **Testing**: Verify exact inputs/outputs
3. **Replay**: Need full content for cache

### When NOT to use:

- **Production**: Risk of storing PII/secrets
- **High volume**: Storage grows quickly
- **Compliance environments**: May violate data policies

---

## Comparison Example

Given this tool call:

```python
@tool(name="search_users")
def search_users(email: str) -> dict:
    return {"user_id": 123, "name": "John Doe", "email": "john@example.com"}

result = search_users("john@example.com")
```

### `metadata_only` stores:

```json
{
  "args_hash": "a1b2c3...",
  "result_hash": "d4e5f6...",
  "result_size": 67
}
```

### `evidence_only` stores:

```json
{
  "args": {"email": "john@example.com"},
  "result": {"user_id": 123, "name": "John Doe", "email": "john@example.com"}
}
```
(Both are small, so stored directly)

### `full` stores:

```json
{
  "args": {"email": "john@example.com"},
  "result": {"user_id": 123, "name": "John Doe", "email": "john@example.com"}
}
```
(Same as evidence_only for small content)

### The difference matters when:

```python
result = fetch_large_document(doc_id="12345")  # Returns 50KB document
```

| Mode | args | result |
|------|------|--------|
| `metadata_only` | Hash | Hash |
| `evidence_only` | `{"doc_id": "12345"}` | Hash (too large) |
| `full` | `{"doc_id": "12345"}` | Full 50KB document |

---

## Recommendations

| Environment | Recommended Mode | Reason |
|-------------|------------------|--------|
| Production | `metadata_only` | Enterprise-safe, compliant |
| Staging | `evidence_only` | Debug issues with some visibility |
| Development | `full` | Maximum visibility |
| CI/CD Tests | `full` | Verify exact behavior |
| Load Testing | `off` | Minimize overhead |

---

## Switching Modes at Runtime

You can change modes between runs:

```python
# Production mode
observe.install(mode="metadata_only")

with observe.run("normal-agent"):
    do_normal_work()

# Debug a specific issue
observe.install(mode="full")

with observe.run("debug-agent"):
    do_problematic_work()  # Full capture for debugging
```
