# Design Doc: Agent Trace Capture

> **Philosophy**: One comprehensive trace per agent run, with all context attached.
> Not just "what happened" but "everything needed to understand, debug, evaluate, and audit."

---

## Problem

Enterprise teams running AI agents need to answer:

1. **"What did the agent do?"** - Full audit trail of every action
2. **"What data did it access?"** - What systems were touched, what flowed where
3. **"Why did it make that decision?"** - The complete reasoning chain
4. **"What was the full context?"** - System prompt, conversation history, retrieved docs
5. **"Can I prove it?"** - Evidence for compliance, legal, debugging
6. **"Is it getting better?"** - Foundation for evaluation and improvement

**Current state**: We capture spans with timing and status, but the **substance** is either not captured or buried. We have structured logs, but not wide events.

---

## Design Principles

### Wide Events for Agents

Inspired by the "canonical log line" / wide event pattern:

| Principle | Application to Agents |
|-----------|----------------------|
| **One event per request** | One comprehensive trace per agent run |
| **High cardinality** | user_id, session_id, run_id, prompt_version - all queryable |
| **High dimensionality** | 50+ fields capturing everything |
| **Context-rich** | Full prompts, full responses, full reasoning chain |
| **Queryable** | SQL queries, not grep |

### What We're NOT Doing

- ❌ Just structured logging (JSON with 5 fields)
- ❌ Just tracing (span name, duration, status)
- ❌ Metrics-only (token counts, latency p99)

### What We ARE Doing

- ✅ Complete trace with full context
- ✅ Every LLM call with entire prompt + response
- ✅ Every tool call with args + result
- ✅ Data flow tracking (what went where)
- ✅ Session continuity (conversation context)
- ✅ Foundation for evaluation

---

## Goals

### v0.1.7 (This Release)

| Priority | Goal |
|----------|------|
| **P0** | Full trace capture (input/output for every step) |
| **P0** | Session/conversation linking |
| **P0** | Full LLM context (system prompt, tools, config) |
| **P0** | Content search (find runs by what was said) |
| **P1** | Data flow direction (read/write, internal/egress) |
| **P1** | Config versioning (prompt_version, model_config) |

### Future (Foundation for Eval)

| Version | Goal |
|---------|------|
| v0.1.8 | Feedback collection (human ratings) |
| v0.1.9 | Evaluation hooks (automated quality checks) |
| v0.2.0 | Immutable audit trail (tamper-proof) |
| v0.2.0 | Anomaly detection |

### Non-Goals (This Release)

- Cost tracking (not core value)
- Framework integrations
- PII detection (future)

---

## User Experience

### Today (Bare Minimum)

```python
observe.install()

@observe.model_call
def call_llm(messages):
    return openai.chat.completions.create(model="gpt-4", messages=messages)

with observe.run("agent") as run:
    call_llm([{"role": "user", "content": "hello"}])
```

**What gets captured**: Span name, timing, status. Maybe args hash. Useless for debugging.

---

### v0.1.7 (Complete Picture)

Same code, rich capture:

```python
observe.install()  # Default: standard mode (full capture)

@observe.model_call
def call_llm(messages):
    return openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
    )

@observe.tool
def search_orders(customer_id: str) -> list[dict]:
    return db.query("SELECT * FROM orders WHERE customer_id = ?", customer_id)

with observe.run(
    "support-agent",
    user_id="agent_user_jane",
    session_id="conversation_123",
    prompt_version="v2.3",
) as run:
    # Capture the user's request
    run.set_input(user_message)

    response = call_llm([
        {"role": "system", "content": "You are a support agent..."},
        {"role": "user", "content": user_message}
    ])

    orders = search_orders("cust_456")

    final_response = call_llm([...])

    run.set_output(final_response)
```

**What gets captured** (Wide Event):

```yaml
Run Trace:
  # Identity
  run_id: run_abc123
  trace_id: trace_xyz
  session_id: conversation_123  # Links to other runs in conversation
  name: support-agent

  # Attribution
  user_id: agent_user_jane
  prompt_version: v2.3

  # Timing
  ts_start: 2024-01-15T10:23:45.000Z
  ts_end: 2024-01-15T10:23:48.500Z
  duration_ms: 3500

  # Status
  status: success
  error: null

  # Content (the substance)
  input: "Where is my order #12345?"
  output: "Your order #12345 is currently in transit..."

  # Data Flow
  data_accessed:
    - type: llm
      resource: openai/gpt-4
      direction: egress  # Data sent to external API
      operation: call
      details:
        tokens_in: 156
        tokens_out: 89
        model: gpt-4
        temperature: 0.7

    - type: database
      resource: orders_table
      direction: internal
      operation: read
      details:
        query: "SELECT * FROM orders WHERE customer_id = ?"
        rows_returned: 3

  # Full Reasoning Chain
  spans:
    - kind: model
      name: call_llm
      input:
        system_prompt: "You are a support agent..."  # FULL system prompt
        messages:
          - role: system
            content: "You are a support agent..."
          - role: user
            content: "Where is my order #12345?"
        model_config:
          model: gpt-4
          temperature: 0.7
          max_tokens: 1000
      output:
        content: "I'll look up order #12345 for you..."
        finish_reason: stop
      duration_ms: 450

    - kind: tool
      name: search_orders
      input:
        customer_id: "cust_456"
      output:
        rows: 3
        data:
          - order_id: "12345"
            status: "in_transit"
            ...
      data_access:
        type: database
        resource: orders_table
        direction: internal
        operation: read
      duration_ms: 120

    - kind: model
      name: call_llm
      input:
        messages: [...]  # Full conversation including previous context
      output:
        content: "Your order #12345 is currently in transit..."
      duration_ms: 380

  # Metrics (secondary, for dashboards)
  model_calls: 2
  tool_calls: 1
  policy_violations: 0
```

---

## Key Concepts

### 1. Session Continuity

Agents have multi-turn conversations. A single run is meaningless without context.

```
Session: conversation_123
├── Run 1: "What are my recent orders?"
│   └── Agent lists orders
├── Run 2: "Where is the last one?"
│   └── Agent looks up order #12345
└── Run 3: "Can I get a refund?"
    └── Agent processes refund
```

**API**:
```python
# Get full conversation
session = observe.get_session("conversation_123")
for run in session.runs:
    print(f"{run.input} → {run.output}")

# Query runs with session context
runs = observe.query_runs(
    session_id="conversation_123",
    include_session_context=True,  # Include previous runs
)
```

### 2. Full LLM Context

Not just the last message, but EVERYTHING the model saw:

```python
@dataclass
class LLMContext:
    """Complete context sent to the LLM."""

    # The full prompt
    system_prompt: str | None
    messages: list[dict]  # Complete message history

    # Model configuration
    model: str
    temperature: float | None
    max_tokens: int | None
    top_p: float | None

    # Tool use
    tools: list[dict] | None  # Function definitions
    tool_choice: str | None

    # Retrieved context (RAG)
    retrieved_documents: list[dict] | None

    # Any other context
    metadata: dict
```

**Why this matters**:
- Debug: "Why did the agent say X?" → Look at full context
- Eval: Compare responses given same context
- Security: What data was sent to the LLM?

### 3. Data Flow Tracking

Track not just "accessed database" but the full data flow:

```python
@dataclass
class DataAccess:
    """Record of data flow."""

    type: str        # database, api, file, llm, cache
    resource: str    # Table name, API endpoint, file path
    operation: str   # read, write, delete, call
    direction: str   # internal, egress, ingress

    # Volume
    details: dict
    # For database: {query, params, rows_returned, rows_affected}
    # For API: {method, endpoint, request_size, response_size, status_code}
    # For LLM: {model, tokens_in, tokens_out, prompt_hash}
    # For file: {path, mode, bytes_read, bytes_written}

    timestamp: int
    span_id: str
```

**Direction matters**:
- `internal`: Data stayed within your systems
- `egress`: Data sent to external API (security concern!)
- `ingress`: Data received from external source

**Queries this enables**:
```sql
-- All runs that sent data to external APIs
SELECT * FROM data_access WHERE direction = 'egress';

-- All runs that wrote to the orders table
SELECT * FROM data_access WHERE resource = 'orders' AND operation = 'write';
```

### 4. Content Search

Find runs by what was said, not just metadata:

```python
# Find all runs mentioning "refund"
runs = observe.search_runs("refund")

# Find runs where agent mentioned a specific order
runs = observe.search_runs("order #12345")

# Combine with filters
runs = observe.search_runs(
    query="refund",
    user_id="jane",
    status="error",
    since="2024-01-01",
)
```

**Implementation**: Full-text search on `input_json` and `output_json`:
```sql
-- SQLite FTS5
CREATE VIRTUAL TABLE runs_fts USING fts5(
    input_text,
    output_text,
    content=runs,
    content_rowid=rowid
);

-- Search
SELECT r.* FROM runs r
JOIN runs_fts ON r.rowid = runs_fts.rowid
WHERE runs_fts MATCH 'refund';
```

### 5. Config Versioning

Track which configuration produced which results:

```python
with observe.run(
    "agent",
    prompt_version="v2.3",           # Which prompt template
    model_config={                    # Model settings
        "model": "gpt-4",
        "temperature": 0.7,
    },
    experiment_id="exp_abc",          # A/B test cohort
) as run:
    ...
```

**Queries this enables**:
```sql
-- Compare success rate across prompt versions
SELECT
    prompt_version,
    COUNT(*) as total,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes
FROM runs
GROUP BY prompt_version;

-- Compare latency across model configs
SELECT
    json_extract(model_config, '$.model') as model,
    AVG(duration_ms) as avg_latency
FROM runs
GROUP BY model;
```

---

## Data Model

### RunTrace

```python
@dataclass
class RunTrace:
    """Complete trace of an agent run - the Wide Event."""

    # Identity
    run_id: str
    trace_id: str
    session_id: str | None      # Links runs in a conversation
    name: str

    # Attribution
    user_id: str | None         # Who ran the agent
    prompt_version: str | None  # Which prompt template
    model_config: dict | None   # Model settings
    experiment_id: str | None   # A/B test cohort

    # Timing
    ts_start: int
    ts_end: int | None
    duration_ms: int | None

    # Status
    status: RunStatus           # ok, error, blocked
    error: ErrorContext | None

    # Content (the substance)
    input: Any                  # Original user request
    output: Any                 # Final agent output

    # Data lineage
    data_accessed: list[DataAccess]

    # Full reasoning chain
    spans: list[SpanTrace]

    # Metrics (secondary)
    model_calls: int
    tool_calls: int
    policy_violations: int

    # Custom metadata
    metadata: dict
```

### SpanTrace

```python
@dataclass
class SpanTrace:
    """A single step in the reasoning chain."""

    span_id: str
    parent_span_id: str | None
    kind: SpanKind              # model, tool, internal
    name: str

    ts_start: int
    ts_end: int | None
    duration_ms: int | None

    status: SpanStatus
    error: ErrorContext | None

    # Content
    input: Any                  # Full input
    output: Any                 # Full output

    # For model calls
    llm_context: LLMContext | None

    # For tool calls
    data_access: DataAccess | None
```

### LLMContext

```python
@dataclass
class LLMContext:
    """Everything sent to the LLM."""

    system_prompt: str | None
    messages: list[dict]
    model: str
    temperature: float | None
    max_tokens: int | None
    tools: list[dict] | None
    tool_choice: str | None
    retrieved_documents: list[dict] | None  # RAG context
```

### DataAccess

```python
@dataclass
class DataAccess:
    """Record of data flow."""

    type: str           # database, api, file, llm, cache
    resource: str       # What was accessed
    operation: str      # read, write, delete, call
    direction: str      # internal, egress, ingress
    details: dict       # Type-specific details
    timestamp: int
    span_id: str
```

---

## Storage Schema

```sql
-- Runs table (the wide event)
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    session_id TEXT,            -- Links conversation
    name TEXT NOT NULL,

    -- Attribution
    user_id TEXT,
    prompt_version TEXT,
    model_config_json TEXT,
    experiment_id TEXT,

    -- Timing
    ts_start INTEGER NOT NULL,
    ts_end INTEGER,
    duration_ms INTEGER,

    -- Status
    status TEXT NOT NULL,
    error_json TEXT,

    -- Content (the substance)
    input_json TEXT,
    input_text TEXT,            -- Extracted text for search
    output_json TEXT,
    output_text TEXT,           -- Extracted text for search

    -- Metrics
    model_calls INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    policy_violations INTEGER DEFAULT 0,

    -- Custom
    metadata_json TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Spans table
CREATE TABLE spans (
    span_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    parent_span_id TEXT,

    kind TEXT NOT NULL,
    name TEXT NOT NULL,

    ts_start INTEGER NOT NULL,
    ts_end INTEGER,
    duration_ms INTEGER,

    status TEXT NOT NULL,
    error_json TEXT,

    -- Content
    input_json TEXT,
    output_json TEXT,

    -- For model calls
    llm_context_json TEXT,

    -- Link to data access
    data_access_id INTEGER REFERENCES data_access(id)
);

-- Data access log
CREATE TABLE data_access (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    span_id TEXT REFERENCES spans(span_id),

    type TEXT NOT NULL,
    resource TEXT NOT NULL,
    operation TEXT NOT NULL,
    direction TEXT NOT NULL,    -- internal, egress, ingress

    details_json TEXT,
    timestamp INTEGER NOT NULL
);

-- Full-text search
CREATE VIRTUAL TABLE runs_fts USING fts5(
    input_text,
    output_text,
    content=runs,
    content_rowid=rowid
);

-- Indexes
CREATE INDEX idx_runs_session ON runs(session_id);
CREATE INDEX idx_runs_user ON runs(user_id);
CREATE INDEX idx_runs_prompt_version ON runs(prompt_version);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_ts ON runs(ts_start);

CREATE INDEX idx_data_access_run ON data_access(run_id);
CREATE INDEX idx_data_access_resource ON data_access(resource);
CREATE INDEX idx_data_access_direction ON data_access(direction);
CREATE INDEX idx_data_access_type ON data_access(type);
```

---

## API Design

### Run Context

```python
@contextmanager
def run(
    name: str,
    *,
    # Attribution
    user_id: str | None = None,
    session_id: str | None = None,
    prompt_version: str | None = None,
    model_config: dict | None = None,
    experiment_id: str | None = None,

    # Existing
    mode: CaptureMode = CaptureMode.STANDARD,  # NEW default
    policy_file: str | None = None,
    fail_on_violation: bool = False,

    # Custom
    metadata: dict | None = None,
) -> Generator[RunContext, None, None]:
    ...
```

### RunContext Methods

```python
class RunContext:
    def set_input(self, input: Any) -> None:
        """Set the original user request."""

    def set_output(self, output: Any) -> None:
        """Set the final agent output."""

    def add_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata."""

    def record_data_access(
        self,
        type: str,
        resource: str,
        operation: str,
        direction: str = "internal",
        details: dict | None = None,
    ) -> None:
        """Explicitly record data access."""
```

### Query API

```python
class Observe:
    def get_trace(self, run_id: str) -> RunTrace:
        """Get complete trace for a run."""

    def get_session(self, session_id: str) -> Session:
        """Get all runs in a conversation session."""

    def query_runs(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        status: str | None = None,
        prompt_version: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[RunTrace]:
        """Query runs by filters."""

    def search_runs(
        self,
        query: str,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RunTrace]:
        """Full-text search on run content."""

    def query_data_access(
        self,
        resource: str | None = None,
        type: str | None = None,
        direction: str | None = None,
        operation: str | None = None,
    ) -> list[DataAccess]:
        """Query data access log."""
```

---

## Example Queries

### Debugging: "Why did the agent say X?"

```python
# Find the run
runs = observe.search_runs("Your order is delayed")

# Get full trace
trace = observe.get_trace(runs[0].run_id)

# See the reasoning chain
for span in trace.spans:
    if span.kind == "model":
        print("PROMPT:", span.llm_context.messages)
        print("RESPONSE:", span.output)
```

### Debugging: "What happened in this conversation?"

```python
session = observe.get_session("conversation_123")

for run in session.runs:
    print(f"User: {run.input}")
    print(f"Agent: {run.output}")
    print("---")
```

### Compliance: "What data did agent access?"

```sql
SELECT
    r.run_id,
    r.user_id,
    d.resource,
    d.operation,
    d.direction
FROM runs r
JOIN data_access d ON r.run_id = d.run_id
WHERE d.direction = 'egress'  -- Data sent externally
ORDER BY r.ts_start DESC;
```

### Evaluation: "Compare prompt versions"

```sql
SELECT
    prompt_version,
    COUNT(*) as total_runs,
    AVG(duration_ms) as avg_latency,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
FROM runs
WHERE ts_start > datetime('now', '-7 days')
GROUP BY prompt_version;
```

### Security: "Any agent access customer PII?"

```sql
SELECT DISTINCT r.run_id, r.user_id, d.resource
FROM runs r
JOIN data_access d ON r.run_id = d.run_id
WHERE d.resource IN ('customers', 'users', 'profiles')
  AND d.operation = 'read';
```

---

## Foundation for Evaluation

This design enables future evaluation capabilities:

### Human Feedback (v0.1.8)

```python
# Collect feedback
observe.record_feedback(
    run_id="run_abc",
    rating=5,              # 1-5 scale
    correct=True,          # Was the answer correct?
    helpful=True,          # Was it helpful?
    comment="Perfect answer",
)

# Query feedback
runs_with_feedback = observe.query_runs(has_feedback=True)
```

### Automated Evaluation (v0.1.9)

```python
# Register evaluators
observe.install(
    evaluators=[
        # Built-in
        LatencyEvaluator(threshold_ms=5000),
        ToolSuccessEvaluator(),

        # Custom
        CustomEvaluator(
            name="factuality",
            fn=lambda trace: check_factuality(trace.output),
        ),
    ]
)

# Evaluators run at end of each run
# Results stored in trace.evaluations
```

### Regression Detection (v0.2.0)

```python
# Compare runs across time/versions
comparison = observe.compare(
    baseline={"prompt_version": "v2.2", "since": "2024-01-01"},
    candidate={"prompt_version": "v2.3", "since": "2024-01-08"},
)

print(comparison.success_rate_diff)  # +5%
print(comparison.latency_diff)       # -100ms
print(comparison.regressions)        # [...]
```

---

## Implementation Plan

### Phase 1: Core Trace Capture

- [ ] Change default mode to `STANDARD`
- [ ] Add `input_json`, `output_json`, `input_text`, `output_text` to runs
- [ ] Add `input_json`, `output_json`, `llm_context_json` to spans
- [ ] Add `user_id`, `session_id`, `prompt_version` to runs
- [ ] Add `set_input()`, `set_output()` to RunContext
- [ ] Capture full LLM context in `@model_call`

### Phase 2: Data Access Tracking

- [ ] Create `data_access` table
- [ ] Add `direction` field
- [ ] Add `record_data_access()` to RunContext
- [ ] Auto-capture in `@tool` decorator

### Phase 3: Query API

- [ ] Implement `get_trace()`
- [ ] Implement `get_session()`
- [ ] Implement `query_runs()`
- [ ] Implement `search_runs()` with FTS
- [ ] Implement `query_data_access()`

### Phase 4: Schema Migration

- [ ] SQLite migration
- [ ] Postgres migration
- [ ] JSONL format update

---

## Migration

### From v0.1.6

**Behavior change**: Default mode changes from `metadata_only` to `standard` (full capture).

This means:
- LLM inputs/outputs are now stored by default
- Tool args/results are now stored by default
- Storage usage will increase

**If you need the old behavior**:
```python
observe.install(mode="metadata_only")
```

**If you have PII concerns**, use `evidence_only` (truncates to 64KB):
```python
observe.install(mode="evidence_only")
```

**Schema changes**: New columns added (nullable, no migration script needed).

---

## Usability Review (End-User Perspective)

### Friction Points Identified

| Problem | Impact | Solution |
|---------|--------|----------|
| `set_input/output()` is manual | Easy to forget, incomplete traces | Auto-detect from first/last span |
| `prompt_version` is magic string | No enforcement, easy to mismatch | Hash-based or decorator-based versioning |
| `session_id` requires manual plumbing | Boilerplate through app stack | Auto-detection hooks |
| `record_data_access()` too verbose | Nobody will use it | Simpler helpers + auto-detect |
| No fallback if methods not called | Silent incomplete data | Graceful defaults |

---

### Solution 1: Auto-Detect Input/Output

Instead of requiring manual calls, infer from spans:

```python
# Before: Manual (easy to forget)
with observe.run("agent") as run:
    run.set_input(user_message)
    result = call_llm(user_message)
    run.set_output(result)

# After: Automatic
with observe.run("agent") as run:
    result = call_llm(user_message)
# run.input = first model_call's input (auto-detected)
# run.output = last model_call's output (auto-detected)
```

**Implementation**:
```python
class RunContext:
    def _infer_input_output(self):
        """Called at run end if input/output not explicitly set."""
        if self.input is None and self._spans:
            # First span's input = run input
            first_span = self._spans[0]
            self.input = first_span.input

        if self.output is None and self._spans:
            # Last successful span's output = run output
            for span in reversed(self._spans):
                if span.status == SpanStatus.OK:
                    self.output = span.output
                    break
```

**Explicit still works** (for override):
```python
with observe.run("agent") as run:
    run.set_input(reformatted_input)  # Override auto-detection
    ...
```

---

### Solution 2: Smart Prompt Versioning

**Option A: Content-based hashing**
```python
# Prompt hash is calculated automatically
@observe.model_call
def call_llm(messages):
    return openai.chat.completions.create(
        model="gpt-4",
        messages=messages,  # Hash of system prompt stored automatically
    )

# Query by hash
runs = observe.query_runs(prompt_hash="abc123...")
```

**Option B: Decorator-based versioning**
```python
@observe.prompt(version="2.3", name="support-agent-prompt")
def get_system_prompt():
    return """You are a helpful support agent..."""

# Version is bound to function, tracked automatically
with observe.run("agent") as run:
    prompt = get_system_prompt()  # Version "2.3" recorded in run
```

**Option C: Config-driven (simplest)**
```python
# observe.yml
prompts:
  support-agent:
    version: "2.3"
    file: prompts/support.txt

# Usage
observe.install(config="observe.yml")
# Version auto-recorded for matching agent names
```

**Recommendation**: Start with Option A (hash-based) + explicit override:
```python
# Auto-hash by default
with observe.run("agent") as run:
    ...
# run.prompt_hash = "sha256:abc123..."

# Explicit version when needed
with observe.run("agent", prompt_version="v2.3") as run:
    ...
```

---

### Solution 3: Session Auto-Detection

Add hooks for common patterns:

```python
observe.install(
    session_detector=lambda: (
        # Try common sources
        flask.request.headers.get("X-Session-Id") or
        flask.session.get("conversation_id") or
        None
    )
)

# Now session_id is auto-detected
with observe.run("agent") as run:
    ...  # session_id populated automatically
```

**Built-in detectors** for common frameworks:
```python
# Auto-detect if framework installed
observe.install(
    session_detector="auto"  # Tries Flask, FastAPI, etc.
)
```

**Or thread-local for custom apps**:
```python
# Set once at request entry
observe.set_session("conversation_123")

# All runs in this thread use it
with observe.run("agent") as run:  # session_id = "conversation_123"
    ...
```

---

### Solution 4: Simpler Data Access API

**Verbose (current)**:
```python
run.record_data_access(
    type="database",
    resource="orders_table",
    operation="read",
    direction="internal",
    details={"query": "SELECT...", "rows": 5},
)
```

**Simple helpers**:
```python
# Database
run.db_read("orders", rows=5)
run.db_write("orders", rows_affected=1)

# API
run.api_call("stripe", "GET /customers", status=200)

# File
run.file_read("/path/to/file", bytes=1024)

# LLM (auto-detected, but can override)
run.llm_call("openai/gpt-4", tokens_in=150, tokens_out=80)
```

**Auto-detection for instrumented clients**:
```python
# If using SQLAlchemy
observe.install(
    instrument=["sqlalchemy", "httpx", "openai"]
)
# Data access recorded automatically for these clients
```

---

### Solution 5: Graceful Defaults

When methods aren't called, use sensible defaults:

| Field | Default if not set |
|-------|-------------------|
| `input` | Auto-inferred from first span |
| `output` | Auto-inferred from last span |
| `session_id` | `None` (standalone run) |
| `prompt_version` | Hash of first system prompt |
| `user_id` | From session detector or `None` |

**Never fail silently** - log debug message if auto-inferred:
```
DEBUG: Run 'agent' input auto-inferred from span 'call_llm' (no set_input() called)
```

---

### Revised User Experience

**Minimal (everything auto-detected)**:
```python
observe.install()

@observe.model_call
def call_llm(messages):
    return openai.chat.completions.create(model="gpt-4", messages=messages)

with observe.run("agent"):
    call_llm([{"role": "user", "content": "Hello"}])

# ✅ input: auto-inferred from first call
# ✅ output: auto-inferred from last call
# ✅ prompt_hash: auto-calculated from system prompt
# ✅ llm_context: auto-captured
```

**With context (when needed)**:
```python
with observe.run(
    "agent",
    user_id="jane",              # Explicit
    session_id="conv_123",       # Or use set_session()
    prompt_version="v2.3",       # Explicit override of auto-hash
) as run:
    # Only set input/output if different from span content
    run.set_input(sanitized_input)

    result = call_llm([...])

    run.set_output(formatted_output)
```

---

### Usability Checklist

Before shipping v0.1.7:

- [ ] Can user get value with **zero** manual method calls?
- [ ] Are common patterns (session, user, prompt version) auto-detectable?
- [ ] Are verbose APIs accompanied by simple helpers?
- [ ] Does forgetting a step result in **degraded** experience, not **broken** experience?
- [ ] Are debug logs helpful when auto-detection kicks in?
- [ ] Is the "just works" path documented prominently?

---

## Open Questions

1. ~~**Default mode change**: Is `standard` too risky as default?~~
   - **Decision**: Yes, default to `standard`. Users install this library *because* they want observability.
   - Users who need minimal capture: `observe.install(mode="metadata_only")`

2. **Storage size**: Full traces are larger
   - Future: Add TTL, sampling, compression

3. **Sensitive data**: Full prompts may contain PII
   - Future: Add redaction patterns, PII detection

4. **Immutability**: Current design allows modification
   - v0.2.0: Add hash chains, append-only audit log

---

## Success Criteria

After v0.1.7, users can:

1. ✅ See exactly what any agent run did, thought, and produced
2. ✅ Understand the full reasoning chain with complete LLM context
3. ✅ Track what data was accessed and where it flowed
4. ✅ Find runs by content, not just metadata
5. ✅ Link runs in a conversation session
6. ✅ Compare different prompt versions
7. ✅ Do all this with minimal code changes

**The foundation for evaluation is in place.**
