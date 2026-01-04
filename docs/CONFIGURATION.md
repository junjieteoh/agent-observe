# Configuration

`agent-observe` can be configured via environment variables or explicit code configuration.

## Quick Start

The simplest approach - zero configuration:

```python
from agent_observe import observe

observe.install()  # Auto-detects everything from environment
```

This automatically:
- Selects the right sink based on available connections
- Uses enterprise-safe defaults (`metadata_only` capture)
- Reads all settings from environment variables

---

## Environment Variables

### Core Settings

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `AGENT_OBSERVE_MODE` | `off`, `metadata_only`, `evidence_only`, `full` | `metadata_only` | Capture mode |
| `AGENT_OBSERVE_ENV` | `dev`, `staging`, `prod` | `prod` | Environment |
| `AGENT_OBSERVE_PROJECT` | string | `""` | Project name for grouping |
| `AGENT_OBSERVE_AGENT_VERSION` | string | `""` | Your agent's version |

### Sink Selection

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `AGENT_OBSERVE_SINK` | `auto`, `sqlite`, `jsonl`, `postgres`, `otlp` | `auto` | Sink type |
| `DATABASE_URL` | PostgreSQL URL | - | Enables Postgres sink |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | URL | - | Enables OTLP sink |
| `AGENT_OBSERVE_SQLITE_PATH` | path | `.riff/observe.db` | SQLite file location |

### Auto Sink Selection (when `AGENT_OBSERVE_SINK=auto`)

| Condition | Sink Selected |
|-----------|---------------|
| `DATABASE_URL` is set | PostgreSQL |
| `OTEL_EXPORTER_OTLP_ENDPOINT` is set | OTLP |
| `AGENT_OBSERVE_ENV=dev` | SQLite |
| Default | JSONL |

### Policy Settings

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `AGENT_OBSERVE_POLICY_FILE` | path | `.riff/observe.policy.yml` | Policy file |
| `AGENT_OBSERVE_FAIL_ON_VIOLATION` | `0`, `1` | `0` | Raise exception on violations |

### Replay Settings

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `AGENT_OBSERVE_REPLAY` | `off`, `write`, `read` | `off` | Replay mode |

### Performance Settings

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `AGENT_OBSERVE_LATENCY_BUDGET_MS` | integer | `20000` | Max run duration before warning |

---

## Explicit Configuration

For full control, use a `Config` object:

```python
import os
from agent_observe import observe
from agent_observe.config import (
    Config,
    CaptureMode,
    Environment,
    SinkType,
    ReplayMode,
)
from pathlib import Path

config = Config(
    # Capture settings
    mode=CaptureMode.METADATA_ONLY,
    env=Environment.PROD,
    project="my-agent",
    agent_version="1.0.0",

    # Sink settings
    sink_type=SinkType.POSTGRES,
    database_url=os.environ.get("DATABASE_URL"),  # Required for Postgres!

    # Policy settings
    policy_file=Path(".riff/observe.policy.yml"),
    fail_on_violation=False,

    # Replay settings
    replay=ReplayMode.OFF,

    # Performance settings
    latency_budget_ms=30000,
)

observe.install(config=config)
```

### Important: Explicit Config Doesn't Read Environment

When you pass a `Config` object, environment variables are **NOT** read automatically. You must include all required fields:

```python
# WRONG - database_url will be None even if DATABASE_URL is set
config = Config(
    sink_type=SinkType.POSTGRES,
    # Missing database_url!
)

# CORRECT - explicitly pass the URL
config = Config(
    sink_type=SinkType.POSTGRES,
    database_url=os.environ.get("DATABASE_URL"),
)
```

---

## Config Class Reference

```python
@dataclass
class Config:
    # Capture mode
    mode: CaptureMode = CaptureMode.METADATA_ONLY

    # Environment
    env: Environment = Environment.PROD

    # Project metadata
    project: str = ""
    agent_version: str = ""

    # Sink selection
    sink_type: SinkType = SinkType.AUTO

    # Sink-specific settings
    database_url: str | None = None        # For Postgres
    sqlite_path: Path | None = None         # For SQLite
    otlp_endpoint: str | None = None        # For OTLP

    # Policy
    policy_file: Path | None = None
    fail_on_violation: bool = False

    # Replay
    replay: ReplayMode = ReplayMode.OFF

    # Performance
    latency_budget_ms: int = 20000
```

### Enums

```python
class CaptureMode(Enum):
    OFF = "off"
    METADATA_ONLY = "metadata_only"
    EVIDENCE_ONLY = "evidence_only"
    FULL = "full"

class Environment(Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

class SinkType(Enum):
    AUTO = "auto"
    SQLITE = "sqlite"
    JSONL = "jsonl"
    POSTGRES = "postgres"
    OTLP = "otlp"

class ReplayMode(Enum):
    OFF = "off"
    WRITE = "write"
    READ = "read"
```

---

## Sink-Specific Configuration

### SQLite

```bash
AGENT_OBSERVE_SINK=sqlite
AGENT_OBSERVE_SQLITE_PATH=./my-database.db
```

Or in code:
```python
config = Config(
    sink_type=SinkType.SQLITE,
    sqlite_path=Path("./my-database.db"),
)
```

### PostgreSQL

```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

Or in code:
```python
config = Config(
    sink_type=SinkType.POSTGRES,
    database_url="postgresql://user:pass@host:5432/dbname",
)
```

See [PostgreSQL Setup](../AGENTS.md#postgresql-setup) for table creation and best practices.

### OTLP (OpenTelemetry)

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

Or in code:
```python
config = Config(
    sink_type=SinkType.OTLP,
    otlp_endpoint="http://localhost:4317",
)
```

Compatible with: Jaeger, Honeycomb, Datadog, Grafana Tempo, etc.

### JSONL (Fallback)

```bash
AGENT_OBSERVE_SINK=jsonl
```

Writes to `.riff/observe.jsonl` by default. Good for simple debugging or when no database is available.

---

## Policy Configuration

Create `.riff/observe.policy.yml`:

```yaml
# Tool patterns
tools:
  allow:
    - "db.*"           # Allow all db.* tools
    - "http.get"       # Allow http.get specifically
  deny:
    - "shell.*"        # Block all shell commands
    - "*.destructive"  # Block anything ending in .destructive

# Limits
limits:
  max_tool_calls: 100   # Max tools per run
  max_retries: 10       # Max retries before flagging
  max_model_calls: 50   # Max LLM calls per run
```

### Policy Behavior

| Setting | Behavior |
|---------|----------|
| `fail_on_violation=False` | Log violation, continue execution |
| `fail_on_violation=True` | Raise `PolicyViolationError` |

---

## Example Configurations

### Development

```bash
AGENT_OBSERVE_MODE=full
AGENT_OBSERVE_ENV=dev
AGENT_OBSERVE_SINK=sqlite
```

### Staging

```bash
AGENT_OBSERVE_MODE=evidence_only
AGENT_OBSERVE_ENV=staging
DATABASE_URL=postgresql://...
AGENT_OBSERVE_FAIL_ON_VIOLATION=0
```

### Production

```bash
AGENT_OBSERVE_MODE=metadata_only
AGENT_OBSERVE_ENV=prod
DATABASE_URL=postgresql://...
AGENT_OBSERVE_FAIL_ON_VIOLATION=1
AGENT_OBSERVE_POLICY_FILE=/etc/myapp/policy.yml
```

### Testing with Replay

```bash
# Record golden responses
AGENT_OBSERVE_MODE=full
AGENT_OBSERVE_REPLAY=write

# Run tests against cached responses
AGENT_OBSERVE_REPLAY=read
```
