# agent-observe Patterns & Recipes

Real-world patterns for using `agent-observe` in production AI agent systems.

## Table of Contents

1. [Enterprise Security Patterns](#enterprise-security-patterns)
2. [Cost Management Patterns](#cost-management-patterns)
3. [Compliance & Audit Patterns](#compliance--audit-patterns)
4. [Performance Patterns](#performance-patterns)
5. [Testing Patterns](#testing-patterns)
6. [Multi-Tenant Patterns](#multi-tenant-patterns)
7. [Integration Patterns](#integration-patterns)

---

## Enterprise Security Patterns

### Pattern 1: Defense in Depth

Layer multiple security hooks for comprehensive protection:

```python
from agent_observe import observe, HookResult, PolicyViolationError

# Layer 1: Tool allowlist
ALLOWED_TOOLS = {"search", "calculate", "format", "summarize"}

@observe.hooks.before_tool(priority=100)  # Runs first (highest priority)
def allowlist_check(ctx):
    if ctx.tool_name not in ALLOWED_TOOLS:
        return HookResult.block(f"Tool '{ctx.tool_name}' not in allowlist")
    return HookResult.proceed()

# Layer 2: Rate limiting per user
from collections import defaultdict
import time

user_calls: dict[str, list[float]] = defaultdict(list)
MAX_CALLS_PER_MINUTE = 60

@observe.hooks.before_tool(priority=90)
def rate_limit(ctx):
    user_id = ctx.run.user_id or "anonymous"
    now = time.time()

    # Clean old entries
    user_calls[user_id] = [t for t in user_calls[user_id] if now - t < 60]

    if len(user_calls[user_id]) >= MAX_CALLS_PER_MINUTE:
        return HookResult.block(f"Rate limit exceeded for user {user_id}")

    user_calls[user_id].append(now)
    return HookResult.proceed()

# Layer 3: Content filtering
BLOCKED_CONTENT = ["DROP TABLE", "DELETE FROM", "rm -rf", "sudo"]

@observe.hooks.before_tool(priority=80)
def content_filter(ctx):
    args_str = str(ctx.args) + str(ctx.kwargs)
    for blocked in BLOCKED_CONTENT:
        if blocked.upper() in args_str.upper():
            return HookResult.block(f"Blocked content detected: {blocked}")
    return HookResult.proceed()
```

### Pattern 2: Contextual Access Control

Grant different permissions based on context:

```python
from agent_observe import observe, HookResult

# Define permission levels
PERMISSION_LEVELS = {
    "admin": {"read", "write", "delete", "admin"},
    "user": {"read", "write"},
    "viewer": {"read"},
}

TOOL_PERMISSIONS = {
    "read_data": "read",
    "write_data": "write",
    "delete_data": "delete",
    "manage_users": "admin",
}

def get_user_role(user_id: str) -> str:
    """Look up user role from your auth system."""
    # In production, call your auth service
    return user_roles.get(user_id, "viewer")

@observe.hooks.before_tool
def check_permissions(ctx):
    user_id = ctx.run.user_id
    if not user_id:
        return HookResult.block("Authentication required")

    role = get_user_role(user_id)
    required_permission = TOOL_PERMISSIONS.get(ctx.tool_name, "read")
    user_permissions = PERMISSION_LEVELS.get(role, set())

    if required_permission not in user_permissions:
        observe.emit_event("security.access_denied", {
            "user_id": user_id,
            "role": role,
            "tool": ctx.tool_name,
            "required": required_permission,
        })
        return HookResult.block(f"Permission denied: {required_permission} required")

    return HookResult.proceed()
```

---

## Cost Management Patterns

### Pattern 3: Budget Enforcement

Enforce per-user or per-session cost limits:

```python
from agent_observe import observe, HookResult
from collections import defaultdict

# Track costs per session
session_costs: dict[str, float] = defaultdict(float)
SESSION_BUDGET = 1.00  # $1 per session

# Pricing per 1K tokens
MODEL_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
}

@observe.hooks.before_model
def check_budget(ctx):
    session_id = ctx.run.session_id or ctx.run.run_id
    current_cost = session_costs[session_id]

    if current_cost >= SESSION_BUDGET:
        return HookResult.block(
            f"Session budget exhausted (${current_cost:.2f}/${SESSION_BUDGET:.2f})"
        )
    return HookResult.proceed()

@observe.hooks.after_model
def track_cost(ctx, result):
    session_id = ctx.run.session_id or ctx.run.run_id

    try:
        usage = result.usage if hasattr(result, 'usage') else result.get('usage', {})
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)

        pricing = MODEL_PRICING.get(ctx.model, MODEL_PRICING["gpt-4"])
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000

        session_costs[session_id] += cost

        ctx.span.set_attribute("cost_usd", cost)
        ctx.span.set_attribute("session_cost_total", session_costs[session_id])

    except Exception:
        pass

    return result
```

### Pattern 4: Smart Model Routing

Route to cheaper models when appropriate:

```python
from agent_observe import observe, HookResult

def estimate_complexity(messages: list) -> str:
    """Estimate task complexity based on input."""
    total_chars = sum(len(str(m)) for m in messages)

    # Simple heuristics - replace with your logic
    if total_chars < 500:
        return "simple"
    elif total_chars < 2000:
        return "medium"
    else:
        return "complex"

MODEL_BY_COMPLEXITY = {
    "simple": "gpt-3.5-turbo",
    "medium": "gpt-4-turbo",
    "complex": "gpt-4",
}

@observe.hooks.before_model
def route_model(ctx):
    # Skip if model is explicitly specified
    if ctx.kwargs.get("force_model"):
        return HookResult.proceed()

    messages = ctx.args[0] if ctx.args else ctx.kwargs.get("messages", [])
    complexity = estimate_complexity(messages)
    recommended_model = MODEL_BY_COMPLEXITY[complexity]

    # Modify the model in kwargs
    new_kwargs = {**ctx.kwargs, "model": recommended_model}

    observe.emit_event("model.routed", {
        "original_model": ctx.model,
        "routed_model": recommended_model,
        "complexity": complexity,
    })

    return HookResult.modify(args=ctx.args, kwargs=new_kwargs)
```

---

## Compliance & Audit Patterns

### Pattern 5: Complete Audit Trail

Log all operations with compliance-ready format:

```python
from agent_observe import observe, PIIConfig
from datetime import datetime, timezone
import json

# Enable PII handling for compliance
observe.install(
    pii=PIIConfig(
        enabled=True,
        action="hash",  # Hash PII for referential integrity without exposure
        patterns={"email": True, "phone": True, "ssn": True},
    )
)

@observe.hooks.before_tool
def audit_before(ctx):
    observe.emit_event("audit.tool_invoked", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": ctx.tool_name,
        "user_id": ctx.run.user_id,
        "session_id": ctx.run.session_id,
        "run_id": str(ctx.run.run_id),
        "args_hash": hash(str(ctx.args)),  # Hash for referential integrity
    })
    return HookResult.proceed()

@observe.hooks.after_tool
def audit_after(ctx, result):
    observe.emit_event("audit.tool_completed", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": ctx.tool_name,
        "user_id": ctx.run.user_id,
        "run_id": str(ctx.run.run_id),
        "success": ctx.span.status == "ok",
        "duration_ms": ctx.span.latency_ms,
    })
    return result

@observe.hooks.on_run_end
def audit_run(ctx):
    observe.emit_event("audit.run_completed", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": str(ctx.run.run_id),
        "user_id": ctx.run.user_id,
        "session_id": ctx.run.session_id,
        "status": ctx.status,
        "duration_ms": ctx.duration_ms,
        "tool_calls": ctx.run.tool_calls,
        "model_calls": ctx.run.model_calls,
        "policy_violations": ctx.run.policy_violations,
    })
```

### Pattern 6: Data Residency Compliance

Ensure data stays in the right region:

```python
from agent_observe import observe, HookResult

# Map users to their data residency requirements
USER_REGIONS = {
    "eu_user": "eu",
    "us_user": "us",
    "apac_user": "apac",
}

# Map tools to their data processing regions
TOOL_REGIONS = {
    "eu_database": "eu",
    "us_database": "us",
    "global_cache": "global",  # Available everywhere
}

@observe.hooks.before_tool
def check_data_residency(ctx):
    user_id = ctx.run.user_id
    user_region = USER_REGIONS.get(user_id)
    tool_region = TOOL_REGIONS.get(ctx.tool_name, "global")

    if tool_region == "global":
        return HookResult.proceed()

    if user_region and tool_region != user_region:
        observe.emit_event("compliance.residency_violation", {
            "user_id": user_id,
            "user_region": user_region,
            "tool": ctx.tool_name,
            "tool_region": tool_region,
        })
        return HookResult.block(
            f"Data residency violation: User data must stay in {user_region}"
        )

    return HookResult.proceed()
```

---

## Performance Patterns

### Pattern 7: Intelligent Caching

Cache expensive operations with TTL and invalidation:

```python
from agent_observe import observe, HookResult
import time
import hashlib
from typing import Any

class Cache:
    def __init__(self, default_ttl: int = 300):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> tuple[bool, Any]:
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return True, value
            del self._cache[key]
        return False, None

    def set(self, key: str, value: Any, ttl: int | None = None):
        expiry = time.time() + (ttl or self._default_ttl)
        self._cache[key] = (value, expiry)

    def invalidate(self, pattern: str):
        keys_to_delete = [k for k in self._cache if pattern in k]
        for k in keys_to_delete:
            del self._cache[k]

cache = Cache(default_ttl=300)

# Tools that should be cached
CACHEABLE_TOOLS = {"search", "fetch_data", "get_user_profile"}

def make_cache_key(tool_name: str, args: tuple, kwargs: dict) -> str:
    content = f"{tool_name}:{args}:{sorted(kwargs.items())}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]

@observe.hooks.before_tool
def check_cache(ctx):
    if ctx.tool_name not in CACHEABLE_TOOLS:
        return HookResult.proceed()

    cache_key = make_cache_key(ctx.tool_name, ctx.args, ctx.kwargs)
    hit, value = cache.get(cache_key)

    if hit:
        ctx.span.set_attribute("cache_hit", True)
        observe.emit_event("performance.cache_hit", {
            "tool": ctx.tool_name,
            "cache_key": cache_key,
        })
        return HookResult.skip(result=value)

    ctx.span.set_attribute("cache_hit", False)
    return HookResult.proceed()

@observe.hooks.after_tool
def populate_cache(ctx, result):
    if ctx.tool_name not in CACHEABLE_TOOLS:
        return result

    cache_key = make_cache_key(ctx.tool_name, ctx.args, ctx.kwargs)
    cache.set(cache_key, result)

    return result
```

### Pattern 8: Circuit Breaker for External Services

Protect against cascading failures:

```python
from agent_observe import observe, HookResult, CircuitBreakerConfig

# Configure per-tool circuit breakers
observe.hooks.set_circuit_breaker(CircuitBreakerConfig(
    enabled=True,
    failure_threshold=5,
    window_seconds=60,
    recovery_seconds=300,
))

# Additional fallback logic
FALLBACK_RESPONSES = {
    "external_api": {"status": "unavailable", "cached": True},
    "database_query": [],
}

@observe.hooks.before_tool
def check_service_health(ctx):
    # Check if we have a fallback for this tool
    if ctx.tool_name in FALLBACK_RESPONSES:
        # You could check service health here
        # If unhealthy, return fallback immediately
        if not is_service_healthy(ctx.tool_name):
            observe.emit_event("resilience.fallback_used", {
                "tool": ctx.tool_name,
                "reason": "service_unhealthy",
            })
            return HookResult.skip(result=FALLBACK_RESPONSES[ctx.tool_name])

    return HookResult.proceed()

def is_service_healthy(service_name: str) -> bool:
    """Check if a service is healthy (implement your logic)."""
    # Could check circuit breaker state, ping endpoint, etc.
    return True  # Placeholder
```

---

## Testing Patterns

### Pattern 9: Recording and Replay

Record agent sessions for deterministic replay testing:

```python
from agent_observe import observe, HookResult
import json

class SessionRecorder:
    def __init__(self):
        self.recording = False
        self.replaying = False
        self.calls: list[dict] = []
        self.replay_index = 0

    def start_recording(self):
        self.recording = True
        self.calls = []

    def stop_recording(self) -> list[dict]:
        self.recording = False
        return self.calls

    def start_replay(self, calls: list[dict]):
        self.replaying = True
        self.calls = calls
        self.replay_index = 0

    def stop_replay(self):
        self.replaying = False
        self.replay_index = 0

recorder = SessionRecorder()

@observe.hooks.after_tool
def record_result(ctx, result):
    if recorder.recording:
        recorder.calls.append({
            "tool": ctx.tool_name,
            "args": ctx.args,
            "kwargs": ctx.kwargs,
            "result": result,
        })
    return result

@observe.hooks.before_tool
def replay_result(ctx):
    if recorder.replaying and recorder.replay_index < len(recorder.calls):
        expected = recorder.calls[recorder.replay_index]

        if expected["tool"] == ctx.tool_name:
            recorder.replay_index += 1
            return HookResult.skip(result=expected["result"])

    return HookResult.proceed()

# Usage in tests:
def test_agent_behavior():
    # Record a session
    recorder.start_recording()
    with observe.run("test-agent"):
        agent.process("What's the weather?")
    recorded = recorder.stop_recording()

    # Replay for deterministic testing
    recorder.start_replay(recorded)
    with observe.run("test-agent-replay"):
        result = agent.process("What's the weather?")
    recorder.stop_replay()

    assert result == expected_result
```

### Pattern 10: Test Assertions with Hooks

Use hooks to validate behavior during tests:

```python
from agent_observe.hooks import (
    RecordingHookRegistry,
    assert_hook_blocks,
    assert_hook_proceeds,
    mock_tool_context,
)

def test_security_hook_blocks_dangerous_tools():
    """Test that security hook blocks dangerous tools."""
    registry = RecordingHookRegistry()

    @registry.before_tool
    def security_check(ctx):
        if ctx.tool_name == "delete_all":
            return HookResult.block("Dangerous tool blocked")
        return HookResult.proceed()

    # Test blocking
    ctx = mock_tool_context(tool_name="delete_all")
    assert_hook_blocks(
        registry.run_before_hooks("before_tool", ctx, (), {}),
        expected_reason="Dangerous tool blocked"
    )

    # Test allowing
    ctx = mock_tool_context(tool_name="search")
    assert_hook_proceeds(
        registry.run_before_hooks("before_tool", ctx, (), {})
    )
```

---

## Multi-Tenant Patterns

### Pattern 11: Tenant Isolation

Ensure complete isolation between tenants:

```python
from agent_observe import observe, HookResult
import contextvars

# Current tenant context
current_tenant = contextvars.ContextVar("current_tenant", default=None)

# Tenant-specific configurations
TENANT_CONFIGS = {
    "tenant_a": {
        "allowed_tools": {"search", "calculate"},
        "max_tokens": 4000,
        "pii_enabled": True,
    },
    "tenant_b": {
        "allowed_tools": {"search", "calculate", "database_query"},
        "max_tokens": 8000,
        "pii_enabled": False,
    },
}

@observe.hooks.on_run_start
def set_tenant_context(ctx):
    # Extract tenant from user_id or metadata
    tenant_id = ctx.run.metadata.get("tenant_id") or extract_tenant(ctx.run.user_id)
    current_tenant.set(tenant_id)

@observe.hooks.before_tool
def enforce_tenant_policy(ctx):
    tenant_id = current_tenant.get()
    if not tenant_id:
        return HookResult.block("Tenant context required")

    config = TENANT_CONFIGS.get(tenant_id)
    if not config:
        return HookResult.block(f"Unknown tenant: {tenant_id}")

    if ctx.tool_name not in config["allowed_tools"]:
        return HookResult.block(
            f"Tool '{ctx.tool_name}' not allowed for tenant {tenant_id}"
        )

    return HookResult.proceed()

def extract_tenant(user_id: str | None) -> str | None:
    """Extract tenant from user ID (e.g., user@tenant.com -> tenant)."""
    if user_id and "@" in user_id:
        return user_id.split("@")[1].split(".")[0]
    return None
```

---

## Integration Patterns

### Pattern 12: Slack/Teams Notifications

Send alerts to communication platforms:

```python
from agent_observe import observe
import httpx

SLACK_WEBHOOK = "https://hooks.slack.com/services/..."

async def send_slack_alert(message: str, severity: str = "warning"):
    """Send alert to Slack."""
    color = {"warning": "#ff9800", "error": "#f44336", "info": "#2196f3"}[severity]

    async with httpx.AsyncClient() as client:
        await client.post(SLACK_WEBHOOK, json={
            "attachments": [{
                "color": color,
                "text": message,
                "footer": "agent-observe",
            }]
        })

@observe.hooks.on_run_end
def alert_on_errors(ctx):
    if ctx.status == "error":
        # Run async in background (don't block)
        import asyncio
        asyncio.create_task(send_slack_alert(
            f"Agent run failed: {ctx.run.name}\n"
            f"User: {ctx.run.user_id}\n"
            f"Error: {ctx.error_message}",
            severity="error"
        ))

@observe.hooks.after_tool
def alert_on_high_latency(ctx, result):
    if ctx.span.latency_ms > 5000:  # > 5 seconds
        asyncio.create_task(send_slack_alert(
            f"High latency detected: {ctx.tool_name} took {ctx.span.latency_ms}ms",
            severity="warning"
        ))
    return result
```

### Pattern 13: Prometheus Metrics

Export metrics for monitoring:

```python
from agent_observe import observe
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
tool_calls_total = Counter(
    'agent_tool_calls_total',
    'Total tool calls',
    ['tool_name', 'status']
)
tool_latency = Histogram(
    'agent_tool_latency_seconds',
    'Tool call latency',
    ['tool_name']
)
active_runs = Gauge(
    'agent_active_runs',
    'Currently active runs'
)

@observe.hooks.on_run_start
def track_run_start(ctx):
    active_runs.inc()

@observe.hooks.on_run_end
def track_run_end(ctx):
    active_runs.dec()

@observe.hooks.after_tool
def track_tool_metrics(ctx, result):
    tool_calls_total.labels(
        tool_name=ctx.tool_name,
        status=ctx.span.status
    ).inc()

    tool_latency.labels(
        tool_name=ctx.tool_name
    ).observe(ctx.span.latency_ms / 1000)

    return result
```

---

## Next Steps

- See [examples/](../examples/) for runnable code
- See [AGENTS.md](../AGENTS.md) for integration with specific frameworks
- See [CONFIGURATION.md](CONFIGURATION.md) for all config options
