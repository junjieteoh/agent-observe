"""
Pytest fixtures for riff-observe tests.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from agent_observe.config import (
    CaptureMode,
    Config,
    Environment,
    ReplayMode,
    SinkType,
)
from agent_observe.observe import Observe
from agent_observe.sinks.sqlite_sink import SQLiteSink


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db(temp_dir: Path) -> Path:
    """Create a temporary SQLite database path."""
    return temp_dir / "test.db"


@pytest.fixture
def sqlite_sink(temp_db: Path) -> Generator[SQLiteSink, None, None]:
    """Create and initialize a SQLite sink for testing."""
    sink = SQLiteSink(path=temp_db, async_writes=False)
    sink.initialize()
    yield sink
    sink.close()


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create a test configuration."""
    return Config(
        mode=CaptureMode.METADATA_ONLY,
        env=Environment.DEV,
        project="test-project",
        agent_version="test-1.0",
        sink_type=SinkType.SQLITE,
        sqlite_path=temp_dir / "observe.db",
        jsonl_dir=temp_dir / "traces",
        fail_on_violation=False,
        replay_mode=ReplayMode.OFF,
    )


@pytest.fixture
def observe_instance(test_config: Config) -> Generator[Observe, None, None]:
    """Create and install an observe instance for testing."""
    obs = Observe()
    obs.install(config=test_config)
    yield obs
    obs._cleanup()


@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Clean environment variables before and after tests."""
    # Save current env
    saved_env = {}
    env_vars = [
        "AGENT_OBSERVE_MODE",
        "AGENT_OBSERVE_ENV",
        "AGENT_OBSERVE_PROJECT",
        "AGENT_OBSERVE_SINK",
        "AGENT_OBSERVE_SQLITE_PATH",
        "AGENT_OBSERVE_JSONL_DIR",
        "AGENT_OBSERVE_POLICY_FILE",
        "AGENT_OBSERVE_FAIL_ON_VIOLATION",
        "AGENT_OBSERVE_REPLAY",
        "DATABASE_URL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ]

    for var in env_vars:
        if var in os.environ:
            saved_env[var] = os.environ[var]
            del os.environ[var]

    yield

    # Restore env
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]
    for var, val in saved_env.items():
        os.environ[var] = val
