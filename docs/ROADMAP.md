# Roadmap

Planned features for future releases.

---

## v0.2.0 - Per-Run Config Overrides

**Status:** Planned

### Problem

Currently, `observe.install()` sets global config that applies to all runs. Users can't override settings for specific runs without restarting.

```python
observe.install(mode="metadata_only")  # Global only

with observe.run("agent"):  # Can't override mode here
    ...
```

### Solution

Allow per-run overrides:

```python
observe.install(mode="metadata_only")  # Default

with observe.run("agent-a"):  # Uses default
    ...

with observe.run("agent-b", mode="full"):  # Override for this run
    ...

with observe.run("agent-c", fail_on_violation=True, latency_budget_ms=5000):
    ...
```

### Overridable Settings

| Setting | Per-run? | Rationale |
|---------|----------|-----------|
| `mode` | ✅ Yes | Debug specific runs without restart |
| `policy_file` | ✅ Yes | Different rules per agent type |
| `fail_on_violation` | ✅ Yes | Strict mode for critical runs |
| `latency_budget_ms` | ✅ Yes | Different SLAs per agent |
| `sink_type` | ❌ No | Can't switch database mid-process |
| `database_url` | ❌ No | Can't switch database mid-process |

### Use Cases

1. **Debugging a specific agent**: Run with `mode="full"` to capture everything
2. **Strict mode for production**: Override `fail_on_violation=True` for critical paths
3. **Different SLAs**: Customer-facing agents get tight latency budgets
4. **A/B testing policies**: Test new policy file on subset of runs

---

## v0.3.0 - Advanced Policy Engine

**Status:** Planned

### Planned Features

| Feature | Description |
|---------|-------------|
| **Argument-level policies** | Block tools based on argument values |
| **Conditional policies** | Different rules per environment |
| **Policy inheritance** | Base + override policies |
| **Runtime budgets** | Token/cost limits per run |
| **Policy versioning** | Track policy changes over time |

### Example: Argument-Level Policies

```yaml
tools:
  - name: "http.request"
    allow:
      args:
        url: "https://api.internal.company.com/*"
    deny:
      args:
        url: "http://*"  # Block non-HTTPS
```

### Example: Conditional Policies

```yaml
environments:
  prod:
    deny: ["shell.*", "file.delete"]
  dev:
    deny: []  # Allow everything in dev
```

---

## v0.4.0 - Enterprise Features

**Status:** Planned

| Feature | Description |
|---------|-------------|
| **External policy sources** | Load policies from URL/API |
| **Webhook notifications** | Alert on high-risk runs |
| **Multi-tenant isolation** | Project-level data separation |
| **RBAC for viewer** | Role-based access to run data |

---

## Contributing

Want a feature sooner? [Open an issue](https://github.com/junjieteoh/agent-observe/issues) or submit a PR.
