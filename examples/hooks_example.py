"""
Lifecycle Hooks Example

Demonstrates how to use hooks to intercept and modify agent execution.
Hooks can be used for:
- Security checks (block dangerous operations)
- Input sanitization (modify arguments before execution)
- Cost tracking (record token usage after model calls)
- Logging and auditing (track all operations)
- Caching (skip execution and return cached results)
"""

from agent_observe import observe, tool, model_call, HookResult
from agent_observe.config import Config, CaptureMode, Environment

# Initialize with full capture mode
observe.install(config=Config(mode=CaptureMode.FULL, env=Environment.DEV))


# =============================================================================
# EXAMPLE 1: Security Hook - Block dangerous operations
# =============================================================================

BLOCKED_TOOLS = {"delete_all", "drop_table", "execute_shell"}

@observe.hooks.before_tool
def security_gate(ctx):
    """Block execution of dangerous tools."""
    if ctx.tool_name in BLOCKED_TOOLS:
        return HookResult.block(f"Tool '{ctx.tool_name}' is blocked for security reasons")

    # Check for SQL injection patterns
    args_str = str(ctx.args)
    if "DROP TABLE" in args_str.upper() or "DELETE FROM" in args_str.upper():
        return HookResult.block("SQL injection detected in tool arguments")

    return HookResult.proceed()


# =============================================================================
# EXAMPLE 2: Input Sanitization Hook - Modify arguments before execution
# =============================================================================

def sanitize_query(query: str) -> str:
    """Remove potentially dangerous characters from queries."""
    # This is a simple example - use proper sanitization in production
    return query.replace(";", "").replace("--", "")

@observe.hooks.before_tool
def sanitize_database_queries(ctx):
    """Sanitize SQL queries before execution."""
    if ctx.tool_name == "query_database":
        original_query = ctx.args[0] if ctx.args else ctx.kwargs.get("query", "")
        sanitized_query = sanitize_query(original_query)

        if original_query != sanitized_query:
            print(f"Sanitized query: '{original_query}' -> '{sanitized_query}'")
            return HookResult.modify(args=(sanitized_query,), kwargs=ctx.kwargs)

    return HookResult.proceed()


# =============================================================================
# EXAMPLE 3: Cost Tracking Hook - Record token usage after model calls
# =============================================================================

# Pricing (example - adjust for actual model pricing)
PRICING = {
    "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
    "gpt-3.5-turbo": {"input": 0.001 / 1000, "output": 0.002 / 1000},
    "claude-3-opus": {"input": 0.015 / 1000, "output": 0.075 / 1000},
}

@observe.hooks.after_model
def track_model_cost(ctx, result):
    """Track token usage and cost after each model call."""
    # Extract token counts from result (structure varies by provider)
    try:
        if hasattr(result, 'usage'):
            input_tokens = result.usage.prompt_tokens
            output_tokens = result.usage.completion_tokens
        elif isinstance(result, dict) and 'usage' in result:
            input_tokens = result['usage'].get('prompt_tokens', 0)
            output_tokens = result['usage'].get('completion_tokens', 0)
        else:
            return result  # Can't track usage

        # Calculate cost
        model = ctx.model or "gpt-4"
        pricing = PRICING.get(model, PRICING["gpt-4"])
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])

        # Record in span attributes
        ctx.span.set_attribute("tokens_input", input_tokens)
        ctx.span.set_attribute("tokens_output", output_tokens)
        ctx.span.set_attribute("cost_usd", round(cost, 6))

        print(f"Model call cost: ${cost:.6f} ({input_tokens} in, {output_tokens} out)")
    except Exception as e:
        print(f"Could not track cost: {e}")

    return result


# =============================================================================
# EXAMPLE 4: Caching Hook - Skip execution for repeated calls
# =============================================================================

_search_cache: dict[str, list] = {}

@observe.hooks.before_tool
def cache_search_results(ctx):
    """Return cached results for repeated search queries."""
    if ctx.tool_name == "search_web":
        query = ctx.args[0] if ctx.args else ctx.kwargs.get("query", "")

        if query in _search_cache:
            print(f"Cache hit for query: '{query}'")
            return HookResult.skip(result=_search_cache[query])

    return HookResult.proceed()

@observe.hooks.after_tool
def store_search_results(ctx, result):
    """Store search results in cache after execution."""
    if ctx.tool_name == "search_web":
        query = ctx.args[0] if ctx.args else ctx.kwargs.get("query", "")
        _search_cache[query] = result
        print(f"Cached results for query: '{query}'")

    return result


# =============================================================================
# EXAMPLE 5: Logging Hooks - Track run lifecycle
# =============================================================================

@observe.hooks.on_run_start
def log_run_start(ctx):
    """Log when a run starts."""
    print(f"\n{'='*60}")
    print(f"Starting run: {ctx.run.name}")
    print(f"Run ID: {ctx.run.run_id}")
    print(f"User: {ctx.run.user_id or 'anonymous'}")
    print(f"{'='*60}\n")

@observe.hooks.on_run_end
def log_run_end(ctx):
    """Log when a run ends with summary."""
    print(f"\n{'='*60}")
    print(f"Run completed: {ctx.run.name}")
    print(f"Status: {ctx.status}")
    print(f"Duration: {ctx.duration_ms:.2f}ms")
    print(f"Tool calls: {ctx.run.tool_calls}")
    print(f"Model calls: {ctx.run.model_calls}")
    print(f"{'='*60}\n")


# =============================================================================
# Define tools and model calls
# =============================================================================

@tool(name="search_web", kind="http")
def search_web(query: str) -> list:
    """Search the web for information."""
    return [
        {"title": f"Result 1 for {query}", "url": "https://example.com/1"},
        {"title": f"Result 2 for {query}", "url": "https://example.com/2"},
    ]

@tool(name="query_database", kind="db")
def query_database(query: str) -> list:
    """Execute a database query."""
    return [{"id": 1, "name": "Example", "query": query}]

@tool(name="delete_all", kind="db")
def delete_all() -> str:
    """Dangerous tool that should be blocked."""
    return "Deleted everything!"

@model_call(provider="openai", model="gpt-4")
def call_llm(prompt: str) -> dict:
    """Simulate an LLM call."""
    # Simulated response with usage
    return {
        "content": f"Response to: {prompt}",
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 100,
        }
    }


# =============================================================================
# Run the example
# =============================================================================

if __name__ == "__main__":
    from agent_observe import PolicyViolationError

    with observe.run("hooks-demo", user_id="demo-user") as run:
        run.set_input("Demonstrate hook functionality")

        # Normal tool call (will be cached on second call)
        print("1. First search (will be cached):")
        results = search_web("AI agents")

        print("\n2. Second search (cache hit):")
        results = search_web("AI agents")

        # Query with sanitization
        print("\n3. Query with potential injection (will be sanitized):")
        db_results = query_database("SELECT * FROM users; DROP TABLE users--")

        # Model call with cost tracking
        print("\n4. Model call (cost will be tracked):")
        response = call_llm("Explain AI agents")

        # Blocked tool call
        print("\n5. Attempting blocked tool:")
        try:
            delete_all()
        except PolicyViolationError as e:
            print(f"Blocked as expected: {e}")

        run.set_output("Demo completed successfully")

    # Flush and show summary
    observe.sink.flush()
    print("\nAll traces saved to database.")
