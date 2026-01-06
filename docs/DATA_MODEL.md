# Data Model

`agent-observe` captures structured data during agent execution. This document explains what gets stored and why.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        observe.run()                         │
│                           (Run)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ @tool       │  │ @model_call │  │ emit_event  │          │
│  │  (Span)     │  │   (Span)    │  │  (Event)    │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                  │
│         └────────────────┼────────────────┘                  │
│                          ▼                                   │
│                    ┌───────────┐                             │
│                    │   Sink    │                             │
│                    └─────┬─────┘                             │
└──────────────────────────┼───────────────────────────────────┘
                           ▼
              ┌────────────────────────┐
              │  SQLite / PostgreSQL   │
              │  JSONL / OpenTelemetry │
              └────────────────────────┘
```

## Runs

A **Run** represents a single agent execution from start to finish.

### When is a Run created?

Every time you use `observe.run()`:

```python
with observe.run("my-agent", task={"goal": "Research AI"}):
    # Everything inside is part of this run
    search_web("AI news")
    call_llm("Summarize findings")
```

### What's stored in a Run?

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | UUID | Unique identifier |
| `trace_id` | TEXT | For distributed tracing correlation |
| `name` | TEXT | Agent name you provided |
| `ts_start` | TIMESTAMP | When the run started |
| `ts_end` | TIMESTAMP | When the run ended |
| `task` | JSONB | Optional task metadata you provided |
| `capture_mode` | TEXT | `off`, `metadata_only`, `evidence_only`, `full` |
| `status` | TEXT | `ok`, `error`, or `blocked` |
| `risk_score` | INTEGER | Automatic score 0-100 |
| `eval_tags` | JSONB | Auto-generated tags like `POLICY_VIOLATION` |
| `tool_calls` | INTEGER | Count of tool invocations |
| `model_calls` | INTEGER | Count of LLM invocations |
| `policy_violations` | INTEGER | Count of blocked operations |
| `latency_ms` | INTEGER | Total run duration |
| `project` | TEXT | Project name (from config) |
| `env` | TEXT | Environment (dev/staging/prod) |
| `agent_version` | TEXT | Your agent's version |
| **v0.1.7 Attribution** | | |
| `user_id` | TEXT | User/account ID for attribution |
| `session_id` | TEXT | Session/conversation ID for linking runs |
| `prompt_version` | TEXT | Explicit prompt version (e.g., "v2.3") |
| `prompt_hash` | TEXT | Auto-calculated hash of system prompt |
| `model_config` | JSONB | Model configuration (model, temperature, etc.) |
| `experiment_id` | TEXT | A/B test or experiment cohort ID |
| **v0.1.7 Content** | | |
| `input_json` | TEXT | JSON-serialized input (what user asked) |
| `input_text` | TEXT | Plain text input (for display) |
| `output_json` | TEXT | JSON-serialized output (agent response) |
| `output_text` | TEXT | Plain text output (for display) |
| `metadata` | JSONB | Custom key-value metadata |

### Why track Runs?

- **Audit trail**: See every agent execution
- **Performance monitoring**: Track latency over time
- **Failure analysis**: Find runs that failed or were blocked
- **Risk assessment**: Identify high-risk runs

---

## Spans

A **Span** represents a single operation within a run - either a tool call or model call.

### When is a Span created?

Every time a decorated function is called:

```python
@tool(name="search", kind="http")
def search_web(query: str) -> list:  # Creates a span each time
    return requests.get(f"https://api.search.com?q={query}").json()

@model_call(provider="openai", model="gpt-4")
def call_llm(prompt: str) -> str:    # Creates a span each time
    return openai.chat.completions.create(...).choices[0].message.content
```

### What's stored in a Span?

| Field | Type | Description |
|-------|------|-------------|
| `span_id` | TEXT | Unique identifier (16-char hex, OpenTelemetry compatible) |
| `run_id` | UUID | Parent run |
| `parent_span_id` | TEXT | Parent span (for nested calls) |
| `kind` | TEXT | `tool` or `model` |
| `name` | TEXT | Tool/model name from decorator |
| `ts_start` | TIMESTAMP | When the operation started |
| `ts_end` | TIMESTAMP | When it completed |
| `status` | TEXT | `ok`, `error`, or `blocked` |
| `attrs` | JSONB | Operation-specific attributes (see below) |
| `error_message` | TEXT | Error details if failed |

### What's in `attrs`?

Depends on the [capture mode](CAPTURE_MODES.md):

**For tools (`metadata_only` mode):**
```json
{
  "args_hash": "4758640e48886c0cfdb6eb3ec4b3f68197a2d3e063c...",
  "result_hash": "be42ae699d507351e0c50c09f81e7b2b8393820ead...",
  "tool.kind": "http",
  "tool.version": "1",
  "replay.hit": false
}
```

**For model calls:**
```json
{
  "provider": "openai",
  "model": "gpt-4",
  "args_hash": "...",
  "result_hash": "..."
}
```

**For model calls (v0.1.7 full mode - includes LLM context):**
```json
{
  "provider": "openai",
  "model": "gpt-4",
  "args_hash": "...",
  "result_hash": "...",
  "llm_context": {
    "messages": [...],
    "system_prompt": "You are a helpful assistant...",
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000,
    "tools": [...],
    "tool_choice": "auto"
  }
}
```

### Why use 16-char hex for span_id?

OpenTelemetry uses 16-character hexadecimal span IDs (not UUIDs). We use the same format for compatibility with OTLP exporters like Jaeger, Honeycomb, and Datadog.

### Why track Spans?

- **Debugging**: See exactly what operations ran
- **Performance**: Find slow tools or model calls
- **Failure analysis**: Identify which operation failed
- **Replay**: Cache results for deterministic testing

---

## Events

An **Event** is a custom occurrence you emit during a run.

### When is an Event created?

When you explicitly emit one:

```python
with observe.run("my-agent"):
    # Custom events
    observe.emit_event("user.query", {"query": "What is AI?"})
    observe.emit_event("step.complete", {"step": 1, "result": "success"})

    # Artifacts (special type of event)
    observe.emit_artifact("report", {"content": "..."}, provenance=["search", "llm"])
```

### What's stored in an Event?

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | UUID | Unique identifier |
| `run_id` | UUID | Parent run |
| `ts` | TIMESTAMP | When the event occurred |
| `type` | TEXT | Event type (e.g., `artifact`, `user.query`) |
| `payload` | JSONB | Event data |

### Event vs Span

| Aspect | Span | Event |
|--------|------|-------|
| Created by | `@tool` / `@model_call` decorators | `emit_event()` / `emit_artifact()` |
| Has duration | Yes (start/end) | No (point in time) |
| Has status | Yes (ok/error) | No |
| Use case | Track operations | Track custom occurrences |

### Why track Events?

- **Custom checkpoints**: Mark important moments in agent flow
- **Artifacts**: Store final outputs with provenance
- **Debugging**: Log arbitrary data for troubleshooting

---

## Replay Cache

The **Replay Cache** stores tool results for deterministic testing.

### When is the cache used?

When you enable replay mode:

```python
# First run: Execute tools and cache results
observe.install(replay="write")

with observe.run("my-agent"):
    result = search_web("AI news")  # Executed and cached

# Second run: Return cached results (no execution)
observe.install(replay="read")

with observe.run("my-agent"):
    result = search_web("AI news")  # Returns cached result
```

### What's stored in Replay Cache?

| Field | Type | Description |
|-------|------|-------------|
| `key` | TEXT | Unique cache key |
| `tool_name` | TEXT | Tool that was called |
| `args_hash` | TEXT | Hash of input arguments |
| `tool_version` | TEXT | Version from `@tool(version="1")` |
| `created_ts` | TIMESTAMP | When cached |
| `status` | TEXT | `ok` or `error` |
| `result` | BYTEA | Serialized output |
| `result_hash` | TEXT | Hash of output for verification |

### How is the cache key generated?

```
key = f"{tool_name}:{args_hash}:{tool_version}"
```

Same tool + same inputs + same version = cache hit.

### Why use Replay Cache?

- **Deterministic tests**: Same inputs always produce same outputs
- **Cost savings**: Don't re-call expensive APIs during testing
- **Offline development**: Work without network dependencies
- **Debugging**: Reproduce exact behavior from production

---

## Relationships

```
┌─────────────┐
│    Run      │
│  (run_id)   │
└──────┬──────┘
       │
       ├────────────────┬────────────────┐
       │                │                │
       ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Span 1    │  │   Span 2    │  │   Event     │
│  (tool)     │  │  (model)    │  │ (artifact)  │
└──────┬──────┘  └─────────────┘  └─────────────┘
       │
       ▼
┌─────────────┐
│  Span 1.1   │
│ (nested)    │
└─────────────┘

Replay Cache is separate (global, not per-run)
```

- A **Run** has many **Spans** and **Events**
- A **Span** can have child **Spans** (nested calls)
- **Replay Cache** is global (shared across all runs)
