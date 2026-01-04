# Using agent-observe with AI Agents

This guide shows how to integrate `agent-observe` with various AI agent frameworks and implementations.

## Core Concept

`agent-observe` is an **observability wrapper** - it doesn't run agents, it **observes** them. You wrap your existing agent code with decorators and context managers to capture:

- Tool calls (timing, success/failure, args hash)
- Model/LLM calls (latency, tokens, provider)
- Policy violations (blocked operations)
- Risk scores (automatic evaluation)

## Quick Start

```python
from agent_observe import observe, tool, model_call

# 1. Initialize once at startup
observe.install()

# 2. Wrap your tools with @tool
@tool(name="search_web", kind="http")
def search_web(query: str) -> list[dict]:
    # Your actual implementation
    return [{"title": "Result", "url": "..."}]

# 3. Wrap your LLM calls with @model_call
@model_call(provider="openai", model="gpt-4")
def call_openai(prompt: str) -> str:
    # Your actual OpenAI call
    import openai
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 4. Wrap your agent run
with observe.run("my-agent", task={"goal": "Research AI"}):
    results = search_web("AI agents 2024")
    analysis = call_openai(f"Summarize: {results}")
    print(analysis)
```

---

## Integration Examples

### 1. OpenAI Function Calling Agent

```python
from agent_observe import observe, tool, model_call
import openai
import json

observe.install()

# Define tools
@tool(name="get_weather", kind="http")
def get_weather(location: str) -> dict:
    # Simulated - replace with real API
    return {"location": location, "temp": 72, "condition": "sunny"}

@tool(name="search_news", kind="http")
def search_news(query: str) -> list[dict]:
    # Simulated - replace with real API
    return [{"title": f"News about {query}", "url": "https://..."}]

# Wrap OpenAI calls
@model_call(provider="openai", model="gpt-4")
def call_gpt4(messages: list, tools: list = None) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
    )
    return response.choices[0]

# Tool registry
TOOLS = {
    "get_weather": get_weather,
    "search_news": search_news,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search for news articles",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }
]

def run_agent(user_query: str):
    with observe.run("openai-function-agent", task={"query": user_query}):
        messages = [{"role": "user", "content": user_query}]

        while True:
            response = call_gpt4(messages, tools=TOOL_SCHEMAS)

            if response.finish_reason == "tool_calls":
                # Execute tool calls
                for tool_call in response.message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    # Call the wrapped tool
                    result = TOOLS[fn_name](**fn_args)

                    messages.append(response.message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            else:
                # Final response
                observe.emit_artifact("final_response", response.message.content)
                return response.message.content

# Run it
result = run_agent("What's the weather in NYC and any news about AI?")
print(result)
```

### 2. Anthropic Claude Agent

```python
from agent_observe import observe, tool, model_call
import anthropic

observe.install()

client = anthropic.Anthropic()

@tool(name="calculator", kind="compute")
def calculator(expression: str) -> float:
    # Safe eval for math
    return eval(expression, {"__builtins__": {}}, {})

@tool(name="web_search", kind="http")
def web_search(query: str) -> list[str]:
    # Your search implementation
    return [f"Result for: {query}"]

@model_call(provider="anthropic", model="claude-3-opus")
def call_claude(messages: list, tools: list = None) -> dict:
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        messages=messages,
        tools=tools or [],
    )
    return response

def run_claude_agent(query: str):
    with observe.run("claude-agent", task={"query": query}):
        tools = [
            {
                "name": "calculator",
                "description": "Evaluate math expressions",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]
                }
            },
            {
                "name": "web_search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]

        messages = [{"role": "user", "content": query}]

        while True:
            response = call_claude(messages, tools)

            # Check for tool use
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if tool_use_blocks:
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in tool_use_blocks:
                    if block.name == "calculator":
                        result = calculator(block.input["expression"])
                    elif block.name == "web_search":
                        result = web_search(block.input["query"])

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })

                messages.append({"role": "user", "content": tool_results})
            else:
                # Final text response
                final = response.content[0].text
                observe.emit_artifact("response", final)
                return final

result = run_claude_agent("Calculate 15 * 7 + 23, then search for Python tutorials")
```

### 3. Google Vertex AI / Gemini Agent

```python
from agent_observe import observe, tool, model_call
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration

vertexai.init(project="your-project", location="us-central1")

observe.install()

@tool(name="lookup_product", kind="db")
def lookup_product(product_id: str) -> dict:
    # Your database lookup
    return {"id": product_id, "name": "Widget", "price": 29.99}

@tool(name="check_inventory", kind="db")
def check_inventory(product_id: str) -> dict:
    return {"product_id": product_id, "quantity": 150, "warehouse": "US-WEST"}

@model_call(provider="google", model="gemini-1.5-pro")
def call_gemini(model: GenerativeModel, prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text

def run_gemini_agent(query: str):
    with observe.run("gemini-agent", task={"query": query}):
        # Define tools for Gemini
        tools = Tool(function_declarations=[
            FunctionDeclaration(
                name="lookup_product",
                description="Look up product details by ID",
                parameters={
                    "type": "object",
                    "properties": {"product_id": {"type": "string"}},
                    "required": ["product_id"]
                }
            ),
            FunctionDeclaration(
                name="check_inventory",
                description="Check inventory for a product",
                parameters={
                    "type": "object",
                    "properties": {"product_id": {"type": "string"}},
                    "required": ["product_id"]
                }
            )
        ])

        model = GenerativeModel("gemini-1.5-pro", tools=[tools])
        chat = model.start_chat()

        response = chat.send_message(query)

        # Handle function calls
        while response.candidates[0].content.parts[0].function_call:
            fc = response.candidates[0].content.parts[0].function_call

            if fc.name == "lookup_product":
                result = lookup_product(fc.args["product_id"])
            elif fc.name == "check_inventory":
                result = check_inventory(fc.args["product_id"])

            response = chat.send_message(str(result))

        final = response.text
        observe.emit_artifact("response", final)
        return final

result = run_gemini_agent("What's the price and inventory for product SKU-12345?")
```

### 4. LangChain Agent

```python
from agent_observe import observe, tool, model_call
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain import hub

observe.install()

# Wrap LangChain tools with agent-observe
@tool(name="get_stock_price", kind="http")
def get_stock_price(symbol: str) -> dict:
    # Your implementation
    return {"symbol": symbol, "price": 150.25, "change": "+2.5%"}

@tool(name="get_company_info", kind="http")
def get_company_info(symbol: str) -> dict:
    return {"symbol": symbol, "name": "Example Corp", "sector": "Technology"}

# Wrap the LLM
@model_call(provider="openai", model="gpt-4")
def wrapped_llm_call(llm, messages):
    return llm.invoke(messages)

def run_langchain_agent(query: str):
    with observe.run("langchain-agent", task={"query": query}):
        # Create LangChain tools (they call our wrapped functions)
        lc_tools = [
            StructuredTool.from_function(
                func=get_stock_price,
                name="get_stock_price",
                description="Get current stock price"
            ),
            StructuredTool.from_function(
                func=get_company_info,
                name="get_company_info",
                description="Get company information"
            )
        ]

        llm = ChatOpenAI(model="gpt-4")
        prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_openai_tools_agent(llm, lc_tools, prompt)
        executor = AgentExecutor(agent=agent, tools=lc_tools, verbose=True)

        result = executor.invoke({"input": query})
        observe.emit_artifact("result", result["output"])
        return result["output"]

result = run_langchain_agent("What's Apple's stock price and company info?")
```

### 5. Custom ReAct Agent

```python
from agent_observe import observe, tool, model_call
import re

observe.install()

@tool(name="search", kind="http")
def search(query: str) -> str:
    return f"Search results for '{query}': Found 3 relevant articles..."

@tool(name="calculate", kind="compute")
def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool(name="lookup", kind="db")
def lookup(entity: str) -> str:
    data = {"Paris": "Capital of France, population 2.1M", "Einstein": "Physicist, E=mc²"}
    return data.get(entity, f"No info found for {entity}")

@model_call(provider="openai", model="gpt-4")
def think(prompt: str) -> str:
    import openai
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

TOOLS_DESC = """
Available tools:
- search(query): Search the web for information
- calculate(expression): Evaluate a math expression
- lookup(entity): Look up facts about an entity

Format your response as:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input>

Or if you have the final answer:
Thought: <your reasoning>
Final Answer: <your answer>
"""

def run_react_agent(question: str, max_steps: int = 5):
    with observe.run("react-agent", task={"question": question}):
        prompt = f"{TOOLS_DESC}\n\nQuestion: {question}\n"

        for step in range(max_steps):
            observe.emit_event("agent.step", {"step": step + 1})

            response = think(prompt)
            prompt += response + "\n"

            # Check for final answer
            if "Final Answer:" in response:
                answer = response.split("Final Answer:")[-1].strip()
                observe.emit_artifact("answer", answer)
                return answer

            # Parse action
            action_match = re.search(r"Action: (\w+)", response)
            input_match = re.search(r"Action Input: (.+)", response)

            if action_match and input_match:
                action = action_match.group(1)
                action_input = input_match.group(1).strip()

                # Execute tool
                if action == "search":
                    result = search(action_input)
                elif action == "calculate":
                    result = calculate(action_input)
                elif action == "lookup":
                    result = lookup(action_input)
                else:
                    result = f"Unknown tool: {action}"

                prompt += f"Observation: {result}\n"

        return "Max steps reached"

result = run_react_agent("What is the population of Paris divided by 1000?")
print(result)
```

---

## Testing Your Agent with agent-observe

### 1. Basic Test - Verify Observability

```python
import pytest
from agent_observe import observe, tool
from agent_observe.config import Config, CaptureMode, Environment, SinkType
from pathlib import Path
import tempfile

def test_agent_observability():
    """Test that agent runs are properly observed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=Path(tmpdir) / "test.db",
        )

        obs = observe.__class__()  # Fresh instance
        obs.install(config=config)

        @tool(name="test_tool")
        def my_tool(x: int) -> int:
            return x * 2

        with obs.run("test-agent", task={"input": 5}):
            result = my_tool(5)
            obs.emit_event("computed", {"result": result})

        # Verify observability
        runs = obs.sink.get_runs()
        assert len(runs) == 1
        assert runs[0]["name"] == "test-agent"
        assert runs[0]["tool_calls"] == 1
        assert runs[0]["status"] == "ok"

        spans = obs.sink.get_spans(runs[0]["run_id"])
        assert len(spans) == 1
        assert spans[0]["name"] == "test_tool"
```

### 2. Test Policy Enforcement

```python
def test_policy_blocks_dangerous_tools():
    """Test that policy violations are recorded."""
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create policy
        policy_path = Path(tmpdir) / "policy.yml"
        policy_path.write_text(yaml.dump({
            "tools": {"deny": ["dangerous.*"]}
        }))

        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=Path(tmpdir) / "test.db",
            policy_file=policy_path,
            fail_on_violation=False,
        )

        obs = observe.__class__()
        obs.install(config=config)

        @tool(name="dangerous.delete_all")
        def dangerous_tool():
            return "executed"

        with obs.run("policy-test"):
            dangerous_tool()  # Should record violation

        runs = obs.sink.get_runs()
        assert runs[0]["policy_violations"] == 1
        assert runs[0]["risk_score"] >= 40  # POLICY_VIOLATION adds 40
```

### 3. Test with Mock LLM

```python
def test_agent_with_mock_llm():
    """Test agent flow with mocked LLM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=Path(tmpdir) / "test.db",
        )

        obs = observe.__class__()
        obs.install(config=config)

        @tool(name="fetch_data")
        def fetch_data(query: str) -> dict:
            return {"results": [1, 2, 3], "query": query}

        from agent_observe import model_call

        @model_call(provider="mock", model="test-model")
        def mock_llm(prompt: str) -> str:
            # Simulate LLM response
            if "analyze" in prompt.lower():
                return "Analysis: The data shows positive trends."
            return "I don't understand."

        with obs.run("mock-agent"):
            data = fetch_data("sales Q4")
            analysis = mock_llm(f"Analyze this: {data}")
            obs.emit_artifact("report", analysis)

        runs = obs.sink.get_runs()
        assert runs[0]["tool_calls"] == 1
        assert runs[0]["model_calls"] == 1

        events = obs.sink.get_events(runs[0]["run_id"])
        artifact_events = [e for e in events if e["type"] == "artifact"]
        assert len(artifact_events) == 1
```

### 4. Test Error Handling

```python
def test_agent_handles_tool_errors():
    """Test that tool errors are properly captured."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=Path(tmpdir) / "test.db",
        )

        obs = observe.__class__()
        obs.install(config=config)

        @tool(name="flaky_tool")
        def flaky_tool():
            raise ConnectionError("Service unavailable")

        with pytest.raises(ConnectionError):
            with obs.run("error-agent"):
                flaky_tool()

        runs = obs.sink.get_runs()
        assert runs[0]["status"] == "error"

        spans = obs.sink.get_spans(runs[0]["run_id"])
        assert spans[0]["status"] == "error"
        assert "Service unavailable" in spans[0]["error_message"]
```

### 5. Integration Test with Real API (Optional)

```python
import os
import pytest

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
def test_real_openai_agent():
    """Integration test with real OpenAI API."""
    import openai

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=Path(tmpdir) / "test.db",
        )

        obs = observe.__class__()
        obs.install(config=config)

        from agent_observe import model_call

        @model_call(provider="openai", model="gpt-3.5-turbo")
        def call_openai(prompt: str) -> str:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            return response.choices[0].message.content

        with obs.run("openai-test"):
            result = call_openai("Say 'Hello' in French")
            assert "Bonjour" in result or "bonjour" in result.lower()

        runs = obs.sink.get_runs()
        assert runs[0]["model_calls"] == 1
        assert runs[0]["status"] == "ok"
```

---

## Viewing Results

After running your agent:

```bash
# Start the viewer
agent-observe view

# Open http://localhost:8765 in your browser
```

You'll see:
- List of all runs with status, risk score, tool/model call counts
- Detailed view of each run with spans (tool calls) and events
- Filter by name, status, risk score, or eval tags

---

## Configuration

### Auto-detection (Recommended)

The simplest approach - just call `observe.install()` and it reads from environment variables:

```python
# Set environment variables
# DATABASE_URL=postgresql://... (for Postgres sink)
# AGENT_OBSERVE_MODE=metadata_only

from agent_observe import observe

observe.install()  # Auto-detects sink from DATABASE_URL
```

### Explicit Configuration

When passing a `Config` object directly, you must include **all connection strings explicitly**.
They are NOT read from environment variables in this case:

```python
import os
from agent_observe import observe
from agent_observe.config import Config, CaptureMode, Environment, SinkType

# Get connection string from environment
database_url = os.environ.get("DATABASE_URL")

if database_url:
    config = Config(
        mode=CaptureMode.METADATA_ONLY,
        env=Environment.PROD,
        sink_type=SinkType.POSTGRES,
        project="my-agent",
        database_url=database_url,  # REQUIRED for Postgres!
    )
    observe.install(config=config)
else:
    # Fallback to auto-detection (will use JSONL or SQLite)
    observe.install()
```

**Common mistake:** Setting `sink_type=SinkType.POSTGRES` but forgetting to pass `database_url`. This causes a silent fallback to NullSink.

### PostgreSQL Setup

Install the Postgres dependency:

```bash
pip install "agent-observe[postgres]"

# Or manually:
pip install "psycopg>=3.1.0"
```

> **Note:** If you get "libpq not found" errors, install `psycopg[binary]` instead: `pip install "psycopg[binary]>=3.1.0"`

#### Production Best Practices

The PostgreSQL sink is designed for production use:

| Feature | Implementation |
|---------|----------------|
| **SQL Injection Safe** | All queries use parameterized `%s` placeholders |
| **Batch Inserts** | Uses `executemany` for efficient bulk writes |
| **Retry Logic** | Transient errors retry with exponential backoff (3 attempts) |
| **Connection Timeout** | 10-second timeout prevents hanging |
| **Graceful Degradation** | Works with pre-created tables (no CREATE permission needed) |
| **Efficient Checks** | Single query to verify all tables exist |

#### Manual Table Creation

If your database user doesn't have permission to create tables, run this SQL manually first:

```sql
-- Schema version tracking
CREATE TABLE IF NOT EXISTS agent_observe_schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT now()
);

-- Runs table
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
    status TEXT CHECK (status IN ('ok', 'error', 'blocked')),
    risk_score INTEGER CHECK (risk_score >= 0 AND risk_score <= 100),
    eval_tags JSONB,
    policy_violations INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    model_calls INTEGER DEFAULT 0,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_runs_ts_start ON runs(ts_start DESC);
CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_risk_score ON runs(risk_score);
CREATE INDEX IF NOT EXISTS idx_runs_project_env ON runs(project, env);
CREATE INDEX IF NOT EXISTS idx_runs_eval_tags ON runs USING GIN(eval_tags);

-- Spans table (span_id is TEXT for OpenTelemetry compatibility)
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
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

CREATE INDEX IF NOT EXISTS idx_spans_run_id ON spans(run_id);
CREATE INDEX IF NOT EXISTS idx_spans_parent ON spans(parent_span_id) WHERE parent_span_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_spans_kind_name ON spans(kind, name);
CREATE INDEX IF NOT EXISTS idx_spans_ts_start ON spans(ts_start);

-- Events table
CREATE TABLE IF NOT EXISTS events (
    event_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    type TEXT NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_events_run_id_type ON events(run_id, type);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

-- Replay cache table (optional, for tool replay feature)
CREATE TABLE IF NOT EXISTS replay_cache (
    key TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    tool_version TEXT,
    created_ts TIMESTAMPTZ DEFAULT now(),
    status TEXT CHECK (status IN ('ok', 'error')),
    result BYTEA,
    result_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_replay_tool_args ON replay_cache(tool_name, args_hash);
CREATE INDEX IF NOT EXISTS idx_replay_created ON replay_cache(created_ts);

-- Insert schema version
INSERT INTO agent_observe_schema_version (version) VALUES (1);
```

#### Schema Design Notes

| Design Choice | Rationale |
|---------------|-----------|
| `run_id UUID` | Globally unique across distributed systems |
| `span_id TEXT` | OpenTelemetry uses 16-char hex IDs, not UUIDs |
| `TIMESTAMPTZ` | Timezone-aware, avoids ambiguity |
| `JSONB` for attrs/tags | Flexible schema for varying tool attributes |
| `ON DELETE CASCADE` | Automatic cleanup when runs are deleted |
| `GIN index on eval_tags` | Fast JSONB containment queries (`@>`) |
| `Partial index on parent_span_id` | Only indexes non-null values (efficient) |
| `idx_replay_created` | Enables efficient TTL-based cleanup |
| `risk_score CHECK 0-100` | Validates range at database level |

---

## Best Practices

1. **Wrap all external calls** - Any tool that calls an API, database, or file system should use `@tool`

2. **Wrap all LLM calls** - Use `@model_call` to track latency and cost

3. **Use meaningful names** - `@tool(name="db.query.users")` is better than `@tool(name="query")`

4. **Emit artifacts for outputs** - Use `observe.emit_artifact()` to track final outputs

5. **Set up policies** - Create a policy file to block dangerous operations

6. **Test with mocks first** - Validate observability before running real API tests

7. **Prefer auto-detection** - Use `observe.install()` without config when possible, it handles env vars automatically
