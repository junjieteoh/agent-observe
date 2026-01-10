"""
Configuration management for agent-observe.

Loads configuration from environment variables with sensible defaults.
Implements zero-config behavior with automatic sink selection.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_observe.pii import PIIConfig

logger = logging.getLogger(__name__)


class CaptureMode(Enum):
    """Controls what data is captured and stored."""

    OFF = "off"
    METADATA_ONLY = "metadata_only"
    EVIDENCE_ONLY = "evidence_only"
    FULL = "full"


class Environment(Enum):
    """Deployment environment."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class SinkType(Enum):
    """Available sink types."""

    AUTO = "auto"
    SQLITE = "sqlite"
    JSONL = "jsonl"
    OTLP = "otlp"
    POSTGRES = "postgres"
    MULTI = "multi"


class ReplayMode(Enum):
    """Tool replay behavior."""

    OFF = "off"
    WRITE = "write"
    READ = "read"


def _to_capture_mode(value: CaptureMode | str) -> CaptureMode:
    """Convert string or enum to CaptureMode."""
    if isinstance(value, CaptureMode):
        return value
    if isinstance(value, str):
        try:
            return CaptureMode(value.lower())
        except ValueError:
            logger.warning(f"Invalid capture mode '{value}', using full")
            return CaptureMode.FULL
    return CaptureMode.FULL


def _to_environment(value: Environment | str) -> Environment:
    """Convert string or enum to Environment."""
    if isinstance(value, Environment):
        return value
    if isinstance(value, str):
        try:
            return Environment(value.lower())
        except ValueError:
            logger.warning(f"Invalid environment '{value}', using prod")
            return Environment.PROD
    return Environment.PROD


def _to_sink_type(value: SinkType | str) -> SinkType:
    """Convert string or enum to SinkType."""
    if isinstance(value, SinkType):
        return value
    if isinstance(value, str):
        try:
            return SinkType(value.lower())
        except ValueError:
            logger.warning(f"Invalid sink type '{value}', using auto")
            return SinkType.AUTO
    return SinkType.AUTO


def _to_replay_mode(value: ReplayMode | str) -> ReplayMode:
    """Convert string or enum to ReplayMode."""
    if isinstance(value, ReplayMode):
        return value
    if isinstance(value, str):
        try:
            return ReplayMode(value.lower())
        except ValueError:
            logger.warning(f"Invalid replay mode '{value}', using off")
            return ReplayMode.OFF
    return ReplayMode.OFF


@dataclass
class Config:
    """Configuration for agent-observe. Accepts strings or enums for mode/env/sink."""

    # Core settings
    # v0.1.7: Default changed from METADATA_ONLY to FULL for comprehensive traces
    mode: CaptureMode = CaptureMode.FULL
    env: Environment = Environment.PROD
    project: str = ""
    agent_version: str = ""

    # Sink settings
    sink_type: SinkType = SinkType.AUTO
    sqlite_path: Path = field(default_factory=lambda: Path(".riff/observe.db"))
    jsonl_dir: Path = field(default_factory=lambda: Path(".riff/traces/"))
    database_url: str | None = None
    otlp_endpoint: str | None = None
    pg_schema: str = "public"  # PostgreSQL schema name

    # Policy settings
    policy_file: Path | None = None
    fail_on_violation: bool = False

    # Replay settings
    replay_mode: ReplayMode = ReplayMode.OFF

    # Performance settings
    latency_budget_ms: int = 20000
    trace_sample_rate: float = 0.1

    # Size caps
    max_event_payload_bytes: int = 16 * 1024  # 16KB
    max_artifact_bytes: int = 64 * 1024  # 64KB

    # PII configuration (optional, disabled by default)
    pii: PIIConfig | dict | None = None

    def __post_init__(self) -> None:
        """Convert string values to enums."""
        # Use object.__setattr__ since we want to normalize values
        object.__setattr__(self, "mode", _to_capture_mode(self.mode))
        object.__setattr__(self, "env", _to_environment(self.env))
        object.__setattr__(self, "sink_type", _to_sink_type(self.sink_type))
        object.__setattr__(self, "replay_mode", _to_replay_mode(self.replay_mode))

    def resolve_sink_type(self) -> SinkType:
        """
        Resolve the actual sink type based on auto-detection logic.

        Priority:
        1. DATABASE_URL set → Postgres
        2. env == dev → SQLite
        3. OTLP endpoint set → OTLP
        4. Fallback → JSONL
        """
        if self.sink_type != SinkType.AUTO:
            return self.sink_type

        if self.database_url:
            return SinkType.POSTGRES
        if self.env == Environment.DEV:
            return SinkType.SQLITE
        if self.otlp_endpoint:
            return SinkType.OTLP
        return SinkType.JSONL


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Parse boolean from environment variable."""
    val = os.environ.get(key, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _get_env_float(key: str, default: float) -> float:
    """Parse float from environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning(f"Invalid float value for {key}: {val}, using default {default}")
        return default


def _get_env_int(key: str, default: int) -> int:
    """Parse integer from environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning(f"Invalid int value for {key}: {val}, using default {default}")
        return default


def _infer_project() -> str:
    """Infer project name from package or directory."""
    # Try common project indicators
    cwd = Path.cwd()

    # Check pyproject.toml
    pyproject = cwd / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text()
            for line in content.splitlines():
                if line.strip().startswith("name"):
                    # Simple parsing: name = "project-name"
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        name = parts[1].strip().strip('"').strip("'")
                        if name:
                            return name
        except Exception:
            pass

    # Fall back to directory name
    return cwd.name


def _infer_agent_version() -> str:
    """Infer agent version from environment."""
    # Check common CI/CD env vars
    for var in ("GIT_SHA", "GIT_COMMIT", "GITHUB_SHA", "CI_COMMIT_SHA"):
        val = os.environ.get(var)
        if val:
            return val[:8]  # Short SHA

    # Check VERSION env
    version = os.environ.get("VERSION") or os.environ.get("APP_VERSION")
    if version:
        return version

    return "dev"


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Environment Variables:
        AGENT_OBSERVE_MODE: off|metadata_only|evidence_only|full (default: metadata_only)
        AGENT_OBSERVE_ENV: dev|staging|prod (default: prod)
        AGENT_OBSERVE_PROJECT: Project name (default: inferred)
        AGENT_OBSERVE_AGENT_VERSION: Agent version (default: inferred from GIT_SHA or "dev")
        AGENT_OBSERVE_SINK: auto|sqlite|jsonl|otlp|postgres|multi (default: auto)
        AGENT_OBSERVE_SQLITE_PATH: SQLite database path (default: .riff/observe.db)
        AGENT_OBSERVE_JSONL_DIR: JSONL output directory (default: .riff/traces/)
        DATABASE_URL: Postgres connection string (enables postgres sink)
        AGENT_OBSERVE_PG_SCHEMA: PostgreSQL schema name (default: public)
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (enables otlp sink)
        AGENT_OBSERVE_POLICY_FILE: Path to policy YAML file
        AGENT_OBSERVE_FAIL_ON_VIOLATION: 0|1 (default: 0)
        AGENT_OBSERVE_REPLAY: off|write|read (default: off)
        AGENT_OBSERVE_LATENCY_BUDGET_MS: Latency budget in ms (default: 20000)
        AGENT_OBSERVE_TRACE_SAMPLE_RATE: Trace sampling rate 0.0-1.0 (default: 0.1)
    """
    # Parse mode (v0.1.7: default changed from metadata_only to full)
    mode_str = os.environ.get("AGENT_OBSERVE_MODE", "full").lower()
    try:
        mode = CaptureMode(mode_str)
    except ValueError:
        logger.warning(f"Invalid AGENT_OBSERVE_MODE: {mode_str}, using full")
        mode = CaptureMode.FULL

    # Parse environment
    env_str = os.environ.get("AGENT_OBSERVE_ENV", "prod").lower()
    try:
        env = Environment(env_str)
    except ValueError:
        logger.warning(f"Invalid AGENT_OBSERVE_ENV: {env_str}, using prod")
        env = Environment.PROD

    # Parse sink type
    sink_str = os.environ.get("AGENT_OBSERVE_SINK", "auto").lower()
    try:
        sink_type = SinkType(sink_str)
    except ValueError:
        logger.warning(f"Invalid AGENT_OBSERVE_SINK: {sink_str}, using auto")
        sink_type = SinkType.AUTO

    # Parse replay mode
    replay_str = os.environ.get("AGENT_OBSERVE_REPLAY", "off").lower()
    try:
        replay_mode = ReplayMode(replay_str)
    except ValueError:
        logger.warning(f"Invalid AGENT_OBSERVE_REPLAY: {replay_str}, using off")
        replay_mode = ReplayMode.OFF

    # Parse paths
    sqlite_path = Path(os.environ.get("AGENT_OBSERVE_SQLITE_PATH", ".riff/observe.db"))
    jsonl_dir = Path(os.environ.get("AGENT_OBSERVE_JSONL_DIR", ".riff/traces/"))

    policy_file_str = os.environ.get("AGENT_OBSERVE_POLICY_FILE")
    policy_file = Path(policy_file_str) if policy_file_str else None

    # Get connection strings
    database_url = os.environ.get("DATABASE_URL")
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    pg_schema = os.environ.get("AGENT_OBSERVE_PG_SCHEMA", "public")

    # Get project and version
    project = os.environ.get("AGENT_OBSERVE_PROJECT") or _infer_project()
    agent_version = os.environ.get("AGENT_OBSERVE_AGENT_VERSION") or _infer_agent_version()

    return Config(
        mode=mode,
        env=env,
        project=project,
        agent_version=agent_version,
        sink_type=sink_type,
        sqlite_path=sqlite_path,
        jsonl_dir=jsonl_dir,
        database_url=database_url,
        otlp_endpoint=otlp_endpoint,
        pg_schema=pg_schema,
        policy_file=policy_file,
        fail_on_violation=_get_env_bool("AGENT_OBSERVE_FAIL_ON_VIOLATION", False),
        replay_mode=replay_mode,
        latency_budget_ms=_get_env_int("AGENT_OBSERVE_LATENCY_BUDGET_MS", 20000),
        trace_sample_rate=_get_env_float("AGENT_OBSERVE_TRACE_SAMPLE_RATE", 0.1),
    )
