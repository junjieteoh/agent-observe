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

## Configuration

**Recommended: Use Config object**

```python
from agent_observe import observe
from agent_observe.config import Config

# Create config with all settings
config = Config(
    mode="full",           # full | evidence_only | metadata_only | off (default: full as of v0.1.7)
    env="dev",             # dev | staging | prod
    database_url="postgresql://user:pass@host/db",  # Optional: enables PostgreSQL
    pg_schema="public",    # PostgreSQL schema (default: public)
)

# Initialize once at startup
observe.install(config=config)
```

**Alternative: Environment variables**

```bash
export AGENT_OBSERVE_MODE=full       # Default as of v0.1.7
export AGENT_OBSERVE_ENV=dev
export DATABASE_URL=postgresql://user:pass@host/db
export AGENT_OBSERVE_PG_SCHEMA=public  # PostgreSQL schema (default: public)
```

```python
observe.install()  # Reads from env vars automatically
```

## v0.1.7: Wide Event Traces

v0.1.7 introduces **Wide Event** trace capture - comprehensive traces that capture everything needed to understand, debug, and audit your agent runs.

### Run Attribution

Add context to your runs for better debugging and analytics:

```python
with observe.run(
    "support-agent",
    user_id="jane_doe",              # Who triggered this run?
    session_id="conversation_123",   # Part of which conversation?
    prompt_version="v2.3",           # Which prompt version?
    experiment_id="ab_test_new_rag", # A/B test cohort
    model_config={"model": "gpt-4", "temperature": 0.7},
    metadata={"customer_tier": "enterprise"},
) as run:
    # Your agent code here
    pass
```

### Capturing Input/Output

Record what your agent received and produced:

```python
with observe.run("support-agent", user_id="jane") as run:
    # Capture the original user request
    run.set_input(user_message)

    # ... agent processing ...
    response = call_llm(messages)
    order_info = lookup_order(order_id)
    final_response = format_response(response, order_info)

    # Capture the final output
    run.set_output(final_response)
```

**Auto-inference**: If you don't call `set_input()`/`set_output()`, they're automatically inferred from the first and last spans.

### Adding Custom Metadata

```python
with observe.run("agent") as run:
    run.add_metadata("customer_tier", "enterprise")
    run.add_metadata("region", "us-west-2")
    run.add_metadata("feature_flags", {"new_rag": True})
```

### Full LLM Context Capture

`@model_call` now captures complete LLM context:

```python
@model_call(provider="openai", model="gpt-4")
def call_llm(messages: list):
    return openai.chat.completions.create(
        model="gpt-4",
        messages=messages,           # Full message history captured
        temperature=0.7,             # Model config captured
        tools=[...],                 # Tool definitions captured
    )
```

The decorator automatically captures:
- System prompt
- Full message history
- Model configuration (temperature, max_tokens, etc.)
- Tool/function definitions
- Response format

### Prompt Hash (Auto-calculated)

Each run automatically calculates a hash of the system prompt used:

```python
with observe.run("agent") as run:
    call_llm([
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "Hello"},
    ])

# After run completes:
# run.prompt_hash = "abc123..." (auto-calculated from system prompt)
```

This lets you query runs by prompt version:
```sql
SELECT * FROM runs WHERE prompt_hash = 'abc123...'
```

### Session Continuity

Link runs in a conversation:

```python
# First message
with observe.run("agent", session_id="conv_123", user_id="jane") as run:
    run.set_input("What's the weather?")
    # ...
    run.set_output("It's 72°F and sunny in SF.")

# Follow-up message (same session)
with observe.run("agent", session_id="conv_123", user_id="jane") as run:
    run.set_input("What about tomorrow?")
    # ...
    run.set_output("Tomorrow will be 68°F with clouds.")
```

Query all runs in a session:
```sql
SELECT * FROM runs WHERE session_id = 'conv_123' ORDER BY ts_start
```

## Core Pattern

```python
from agent_observe import observe, tool, model_call
from agent_observe.config import Config

# 1. Initialize once at startup
config = Config(mode="full", env="dev")
observe.install(config=config)

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

**Important:** All `@tool` and `@model_call` decorated functions must be called **inside** `observe.run()` context.

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
from agent_observe.config import Config

observe.install(config=Config(mode="full", env="dev"))

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
from agent_observe.config import Config

observe.install(config=Config(mode="full", env="dev"))

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
from agent_observe.config import Config

observe.install(config=Config(mode="full", env="dev"))

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
from agent_observe.config import Config

observe.install(config=Config(mode="full", env="dev"))

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
    -- v0.1.7: Attribution fields
    user_id TEXT,
    session_id TEXT,
    prompt_version TEXT,
    prompt_hash TEXT,
    model_config JSONB,
    experiment_id TEXT,
    -- v0.1.7: Content fields (Wide Event)
    input_json TEXT,
    input_text TEXT,
    output_json TEXT,
    output_text TEXT,
    -- v0.1.7: Custom metadata
    metadata JSONB,
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
-- v0.1.7: Indexes for new fields
CREATE INDEX idx_runs_user_id ON runs(user_id);
CREATE INDEX idx_runs_session_id ON runs(session_id);
CREATE INDEX idx_runs_prompt_version ON runs(prompt_version);
CREATE INDEX idx_runs_experiment_id ON runs(experiment_id);
```

## Viewing Results

```bash
agent-observe view
# Open http://localhost:8765
```

## Lifecycle Hooks (v0.2)

Hooks let you inject custom logic at key points in execution:

```python
from agent_observe import observe, HookResult

# Block dangerous operations
@observe.hooks.before_tool
def security_gate(ctx):
    if ctx.tool_name in BLOCKED_TOOLS:
        return HookResult.block(f"Tool {ctx.tool_name} is blocked")
    return HookResult.proceed()

# Modify inputs before execution
@observe.hooks.before_tool
def sanitize_sql(ctx):
    if ctx.tool_name == "execute_query":
        cleaned = sanitize_sql_input(ctx.args[0])
        return HookResult.modify(args=(cleaned,), kwargs=ctx.kwargs)
    return HookResult.proceed()

# Track costs after model calls
@observe.hooks.after_model
def track_cost(ctx, result):
    if hasattr(result, 'usage'):
        cost = calculate_cost(result.usage)
        ctx.span.set_attribute("cost_usd", cost)
    return result

# Log at run start/end
@observe.hooks.on_run_start
def log_start(ctx):
    print(f"Starting run: {ctx.run.name}")

@observe.hooks.on_run_end
def log_end(ctx):
    print(f"Run completed in {ctx.duration_ms}ms with status: {ctx.status}")
```

### Hook Types

| Hook | When it Runs | Can Block? |
|------|--------------|------------|
| `before_tool` | Before tool execution | Yes |
| `after_tool` | After tool returns | No |
| `before_model` | Before LLM call | Yes |
| `after_model` | After LLM returns | No |
| `on_run_start` | When run begins | No |
| `on_run_end` | When run completes | No |

### HookResult Actions

| Action | Effect |
|--------|--------|
| `HookResult.proceed()` | Continue normal execution |
| `HookResult.block(reason)` | Stop execution, raise error |
| `HookResult.skip(result)` | Skip execution, return provided result |
| `HookResult.modify(args, kwargs)` | Change arguments before execution |

## PII Handling (v0.2)

Automatically redact sensitive data before storage:

```python
from agent_observe import observe, PIIConfig

observe.install(
    pii=PIIConfig(
        enabled=True,
        action="redact",  # or "hash", "tokenize", "flag"
        patterns={
            "email": True,
            "phone": True,
            "ssn": True,
            "credit_card": True,
            # Custom patterns
            "employee_id": r"EMP-\d{6}",
            "api_key": r"sk-[a-zA-Z0-9]{32}",
        },
    )
)
```

**Built-in patterns**: `email`, `phone`, `ssn`, `credit_card`, `ip_address`, `date_of_birth`, `api_key`, `aws_key`

## Circuit Breaker (v0.2)

Protect your agent from failing hooks:

```python
from agent_observe import observe, CircuitBreakerConfig, HookRegistry

# Create registry with circuit breaker
registry = HookRegistry(
    circuit_breaker=CircuitBreakerConfig(
        enabled=True,
        failure_threshold=5,    # Open after 5 failures
        window_seconds=60,      # Within 60 seconds
        recovery_seconds=300,   # Try again after 5 minutes
    )
)

# Or configure on existing hooks
observe.hooks.set_circuit_breaker(CircuitBreakerConfig(
    failure_threshold=3,
    window_seconds=30,
))
```

## Which Example Should I Run?

Start with the example that matches what you want to learn:

| I want to... | Run this example |
|--------------|------------------|
| **Get started quickly** | `python examples/basic_usage.py` |
| **See a real agent workflow** | `python examples/customer_support_agent.py` |
| **Control LLM costs** | `python examples/rag_agent_with_budget.py` |
| **Block dangerous operations** | `python examples/hooks_example.py` |
| **Protect customer PII** | `python examples/pii_handling.py` |
| **Use async/await** | `python examples/async_agent.py` |
| **Set up allow/deny rules** | `python examples/with_policy.py` |
| **Query stored traces** | `python examples/query_runs.py` |
| **Understand capture modes** | `python examples/capture_modes.py` |

### By Use Case

**Building a Customer-Facing Agent?**
```bash
# Start here - shows PII protection, audit trail, ticketing
python examples/customer_support_agent.py
```

**Building a RAG System?**
```bash
# Shows budget enforcement, caching, cost tracking
python examples/rag_agent_with_budget.py
```

**Need Security Guardrails?**
```bash
# Shows blocking, input sanitization, security hooks
python examples/hooks_example.py
```

**Need GDPR/HIPAA Compliance?**
```bash
# Shows redaction, hashing, tokenization of PII
python examples/pii_handling.py
```

**Just Want to See Traces?**
```bash
# Minimal setup, then view in UI
python examples/basic_usage.py
agent-observe view
```

### Feature Matrix

| Example | Hooks | PII | Policy | Cost Tracking | Caching | Async |
|---------|:-----:|:---:|:------:|:-------------:|:-------:|:-----:|
| `basic_usage.py` | | | | | | |
| `customer_support_agent.py` | ✓ | ✓ | | ✓ | | |
| `rag_agent_with_budget.py` | ✓ | | | ✓ | ✓ | |
| `hooks_example.py` | ✓ | | | ✓ | ✓ | |
| `pii_handling.py` | | ✓ | | | | |
| `with_policy.py` | | | ✓ | | | |
| `async_agent.py` | | | | | | ✓ |

## Quick Checklist

1. `observe.install(config=Config(mode="full", env="dev"))` at startup
2. `@tool(name="...", kind="...")` on all tool functions
3. `@model_call(provider="...", model="...")` on all LLM calls
4. `observe.run("agent-name")` around agent execution (tools must be called INSIDE this context)
5. `observe.emit_artifact()` for final outputs
6. (Optional) Add hooks for security, cost tracking, or custom logic
7. (Optional) Enable PII handling for compliance

## Latest Model IDs (January 2025)

| Provider | Latest Models |
|----------|---------------|
| OpenAI | `gpt-5`, `gpt-4.1`, `gpt-4.1-mini`, `o4-mini` |
| Anthropic | `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251015` |
| Google | `gemini-3-flash`, `gemini-3-pro`, `gemini-2.5-pro`, `gemini-2.0-flash` |
