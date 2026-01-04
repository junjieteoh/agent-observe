# agent-observe Integration Guide

Quick guide for integrating `agent-observe` into your AI agent.

## Why Framework-Agnostic?

`agent-observe` works with **any** AI agent framework because it uses simple decorators:

- **No SDK lock-in**: Works with OpenAI, Anthropic, Google, or any LLM provider
- **No framework dependency**: Works with LangChain, LlamaIndex, CrewAI, or custom agents
- **Decorator-based**: Just wrap your existing functions - no code rewrites needed

```
Your Code                 agent-observe              Storage
─────────────────────────────────────────────────────────────
@tool                  →  Captures timing,     →   SQLite /
def my_tool():            args hash, status        PostgreSQL /
    ...                                            OTLP

@model_call            →  Captures provider,   →   Query via
def call_llm():           model, latency           viewer UI
    ...
```

## Core Pattern

```python
from agent_observe import observe, tool, model_call

# 1. Initialize once at startup
observe.install(mode="full")  # Use "full" to see all data

# 2. Wrap tools with @tool
@tool(name="search", kind="http")
def search_web(query: str) -> list:
    return [{"title": "Result", "url": "..."}]

# 3. Wrap LLM calls with @model_call
@model_call(provider="openai", model="gpt-4.1")
def call_llm(prompt: str) -> str:
    return openai.chat.completions.create(...).choices[0].message.content

# 4. Wrap agent execution with observe.run()
with observe.run("my-agent", task={"goal": "Research AI"}):
    results = search_web("AI agents")
    analysis = call_llm(f"Analyze: {results}")
    observe.emit_artifact("output", analysis)
```

## Decorator Reference

### @tool

```python
@tool(name="tool_name", kind="http|db|compute|file|generic", version="1")
def my_tool(arg: str) -> dict:
    return {"result": "..."}
```

- `name`: Tool identifier (appears in logs)
- `kind`: Category for grouping
- `version`: For replay cache versioning

### @model_call

```python
@model_call(provider="openai|anthropic|google", model="gpt-4.1")
def my_llm_call(prompt: str) -> str:
    return "response"
```

### Async Support

```python
@tool(name="async_fetch", kind="http")
async def async_fetch(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        return await client.get(url)

async with observe.arun("async-agent"):
    data = await async_fetch("https://api.example.com")
```

## Framework Examples

### OpenAI (GPT-4.1, GPT-5, o4-mini)

```python
import openai
from agent_observe import observe, tool, model_call

observe.install(mode="full")

@tool(name="get_weather", kind="http")
def get_weather(location: str) -> dict:
    return {"temp": 72, "condition": "sunny"}

@model_call(provider="openai", model="gpt-4.1")
def call_gpt(messages: list):
    return openai.chat.completions.create(
        model="gpt-4.1",  # or "gpt-5", "o4-mini"
        messages=messages
    ).choices[0]

with observe.run("openai-agent"):
    response = call_gpt([{"role": "user", "content": "What's the weather?"}])
```

### Anthropic (Claude Opus 4.5, Sonnet 4.5)

```python
import anthropic
from agent_observe import observe, model_call

observe.install(mode="full")

@model_call(provider="anthropic", model="claude-sonnet-4-5-20250929")
def call_claude(messages: list):
    return anthropic.Anthropic().messages.create(
        model="claude-sonnet-4-5-20250929",  # or "claude-opus-4-5-20251101"
        max_tokens=4096,
        messages=messages
    )

with observe.run("claude-agent"):
    response = call_claude([{"role": "user", "content": "Hello"}])
```

### Google Gemini (Gemini 3 Flash, 2.5 Pro)

```python
import google.generativeai as genai
from agent_observe import observe, model_call

observe.install(mode="full")

@model_call(provider="google", model="gemini-3-flash")
def call_gemini(prompt: str):
    model = genai.GenerativeModel("gemini-3-flash")  # or "gemini-2.5-pro"
    return model.generate_content(prompt)

with observe.run("gemini-agent"):
    response = call_gemini("Explain quantum computing")
```

### LangChain

```python
from langchain.tools import StructuredTool
from agent_observe import observe, tool

observe.install(mode="full")

@tool(name="search", kind="http")
def search(query: str) -> list:
    return [{"title": "Result"}]

# Wrap with LangChain - decorator still captures calls
lc_tool = StructuredTool.from_function(func=search, name="search")

with observe.run("langchain-agent"):
    # Your LangChain agent uses lc_tool
    pass
```

## Emitting Data

```python
with observe.run("my-agent"):
    # Custom events
    observe.emit_event("step.started", {"step": 1})

    # Artifacts (final outputs)
    observe.emit_artifact("report", {"content": "..."})
```

## Configuration

### Simple (Recommended)

```python
# Strings work - no need for enums
observe.install(mode="full")

# Or with config
from agent_observe.config import Config

config = Config(
    mode="full",           # Strings accepted!
    env="dev",
    sink_type="postgres",
    database_url="postgresql://...",
)
observe.install(config=config)
```

### Environment Variables

```bash
AGENT_OBSERVE_MODE=full              # full|metadata_only|evidence_only|off
AGENT_OBSERVE_ENV=dev                # dev|staging|prod
DATABASE_URL=postgresql://...        # Enables Postgres
```

## PostgreSQL Setup

```bash
pip install "agent-observe[postgres]"

# If "libpq not found" error:
pip install "psycopg[binary]"
```

### Manual Table Creation

If your database user can't create tables:

```sql
CREATE TABLE IF NOT EXISTS runs (
    run_id UUID PRIMARY KEY,
    trace_id TEXT,
    name TEXT NOT NULL,
    ts_start TIMESTAMPTZ NOT NULL,
    ts_end TIMESTAMPTZ,
    task JSONB,
    agent_version TEXT,
    project TEXT,
    env TEXT,
    capture_mode TEXT CHECK (capture_mode IN ('off', 'metadata_only', 'evidence_only', 'full')),
    status TEXT CHECK (status IN ('ok', 'error', 'blocked')),
    risk_score INTEGER CHECK (risk_score >= 0 AND risk_score <= 100),
    eval_tags JSONB,
    policy_violations INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    model_calls INTEGER DEFAULT 0,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    run_id UUID REFERENCES runs(run_id) ON DELETE CASCADE,
    parent_span_id TEXT,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    ts_start TIMESTAMPTZ NOT NULL,
    ts_end TIMESTAMPTZ,
    status TEXT CHECK (status IN ('ok', 'error', 'blocked')),
    attrs JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS events (
    event_id UUID PRIMARY KEY,
    run_id UUID REFERENCES runs(run_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    type TEXT NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX idx_runs_ts ON runs(ts_start DESC);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_risk ON runs(risk_score);
CREATE INDEX idx_spans_run ON spans(run_id);
CREATE INDEX idx_spans_parent ON spans(parent_span_id) WHERE parent_span_id IS NOT NULL;
CREATE INDEX idx_events_run ON events(run_id);
CREATE INDEX idx_eval_tags ON runs USING GIN (eval_tags);
```

## Viewing Results

```bash
agent-observe view
# Open http://localhost:8765
```

## Quick Checklist

1. `observe.install(mode="full")` at startup
2. `@tool(name="...", kind="...")` on all tool functions
3. `@model_call(provider="...", model="...")` on all LLM calls
4. `observe.run("agent-name")` around agent execution
5. `observe.emit_artifact()` for final outputs

## Latest Model IDs (January 2025)

| Provider | Latest Models |
|----------|---------------|
| OpenAI | `gpt-5`, `gpt-4.1`, `gpt-4.1-mini`, `o4-mini` |
| Anthropic | `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251015` |
| Google | `gemini-3-flash`, `gemini-3-pro`, `gemini-2.5-pro`, `gemini-2.0-flash` |
