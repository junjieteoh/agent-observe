"""Tests for configuration module."""

from __future__ import annotations

import os
from pathlib import Path

from agent_observe.config import (
    CaptureMode,
    Config,
    Environment,
    ReplayMode,
    SinkType,
    load_config,
)


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Config()

        # v0.1.7: Default changed from METADATA_ONLY to FULL
        assert config.mode == CaptureMode.FULL
        assert config.env == Environment.PROD
        assert config.sink_type == SinkType.AUTO
        assert config.fail_on_violation is False
        assert config.replay_mode == ReplayMode.OFF

    def test_resolve_sink_type_explicit(self) -> None:
        """Test explicit sink type override."""
        config = Config(sink_type=SinkType.SQLITE)
        assert config.resolve_sink_type() == SinkType.SQLITE

        config = Config(sink_type=SinkType.JSONL)
        assert config.resolve_sink_type() == SinkType.JSONL

    def test_resolve_sink_type_auto_with_database_url(self) -> None:
        """Test auto sink selection with DATABASE_URL."""
        config = Config(
            sink_type=SinkType.AUTO,
            database_url="postgresql://localhost/test",
        )
        assert config.resolve_sink_type() == SinkType.POSTGRES

    def test_resolve_sink_type_auto_dev(self) -> None:
        """Test auto sink selection in dev environment."""
        config = Config(
            sink_type=SinkType.AUTO,
            env=Environment.DEV,
        )
        assert config.resolve_sink_type() == SinkType.SQLITE

    def test_resolve_sink_type_auto_with_otlp(self) -> None:
        """Test auto sink selection with OTLP endpoint."""
        config = Config(
            sink_type=SinkType.AUTO,
            env=Environment.PROD,
            otlp_endpoint="http://localhost:4317",
        )
        assert config.resolve_sink_type() == SinkType.OTLP

    def test_resolve_sink_type_auto_fallback(self) -> None:
        """Test auto sink selection fallback to JSONL."""
        config = Config(
            sink_type=SinkType.AUTO,
            env=Environment.PROD,
        )
        assert config.resolve_sink_type() == SinkType.JSONL

    def test_config_accepts_strings(self) -> None:
        """Test that Config accepts string values and converts to enums."""
        config = Config(
            mode="full",
            env="dev",
            sink_type="postgres",
            replay_mode="write",
        )

        assert config.mode == CaptureMode.FULL
        assert config.env == Environment.DEV
        assert config.sink_type == SinkType.POSTGRES
        assert config.replay_mode == ReplayMode.WRITE

    def test_config_accepts_mixed_strings_and_enums(self) -> None:
        """Test that Config accepts mix of strings and enums."""
        config = Config(
            mode=CaptureMode.EVIDENCE_ONLY,
            env="staging",
            sink_type=SinkType.SQLITE,
        )

        assert config.mode == CaptureMode.EVIDENCE_ONLY
        assert config.env == Environment.STAGING
        assert config.sink_type == SinkType.SQLITE

    def test_config_invalid_string_uses_default(self) -> None:
        """Test that invalid string values fall back to defaults."""
        config = Config(
            mode="invalid_mode",
            env="invalid_env",
        )

        # v0.1.7: Default changed from METADATA_ONLY to FULL
        assert config.mode == CaptureMode.FULL
        assert config.env == Environment.PROD


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_defaults(self) -> None:
        """Test loading config with defaults."""
        config = load_config()

        # v0.1.7: Default changed from METADATA_ONLY to FULL
        assert config.mode == CaptureMode.FULL
        assert config.env == Environment.PROD
        assert config.sink_type == SinkType.AUTO

    def test_load_config_from_env(self) -> None:
        """Test loading config from environment variables."""
        os.environ["AGENT_OBSERVE_MODE"] = "full"
        os.environ["AGENT_OBSERVE_ENV"] = "dev"
        os.environ["AGENT_OBSERVE_SINK"] = "sqlite"
        os.environ["AGENT_OBSERVE_FAIL_ON_VIOLATION"] = "1"
        os.environ["AGENT_OBSERVE_REPLAY"] = "write"

        config = load_config()

        assert config.mode == CaptureMode.FULL
        assert config.env == Environment.DEV
        assert config.sink_type == SinkType.SQLITE
        assert config.fail_on_violation is True
        assert config.replay_mode == ReplayMode.WRITE

    def test_load_config_invalid_values(self) -> None:
        """Test loading config with invalid values uses defaults."""
        os.environ["AGENT_OBSERVE_MODE"] = "invalid_mode"
        os.environ["AGENT_OBSERVE_ENV"] = "invalid_env"

        config = load_config()

        # Should fall back to defaults
        # v0.1.7: Default changed from METADATA_ONLY to FULL
        assert config.mode == CaptureMode.FULL
        assert config.env == Environment.PROD

    def test_load_config_database_url(self) -> None:
        """Test loading config with DATABASE_URL."""
        os.environ["DATABASE_URL"] = "postgresql://localhost/test"

        config = load_config()

        assert config.database_url == "postgresql://localhost/test"
        assert config.resolve_sink_type() == SinkType.POSTGRES

    def test_load_config_sqlite_path(self) -> None:
        """Test loading config with custom SQLite path."""
        os.environ["AGENT_OBSERVE_SQLITE_PATH"] = "/custom/path/observe.db"

        config = load_config()

        assert config.sqlite_path == Path("/custom/path/observe.db")

    def test_load_config_latency_budget(self) -> None:
        """Test loading latency budget from env."""
        os.environ["AGENT_OBSERVE_LATENCY_BUDGET_MS"] = "30000"

        config = load_config()

        assert config.latency_budget_ms == 30000
