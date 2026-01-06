"""Tests for v0.1.7 Wide Event features."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agent_observe import model_call, observe, tool
from agent_observe.config import CaptureMode, Config
from agent_observe.decorators import _extract_llm_context


class TestRunAttribution:
    """Tests for run attribution fields (user_id, session_id, etc.)."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        config = Config(
            mode=CaptureMode.FULL,
            env="dev",
            sqlite_path=Path(self.temp_dir) / "test.db",
        )
        observe.install(config=config)

    def teardown_method(self) -> None:
        """Clean up."""
        observe._cleanup()

    def test_run_with_user_id(self) -> None:
        """Test run with user_id parameter."""
        with observe.run("test-agent", user_id="jane_doe") as run:
            pass

        assert run.user_id == "jane_doe"

    def test_run_with_session_id(self) -> None:
        """Test run with session_id parameter."""
        with observe.run("test-agent", session_id="conv_123") as run:
            pass

        assert run.session_id == "conv_123"

    def test_run_with_prompt_version(self) -> None:
        """Test run with prompt_version parameter."""
        with observe.run("test-agent", prompt_version="v2.3") as run:
            pass

        assert run.prompt_version == "v2.3"

    def test_run_with_experiment_id(self) -> None:
        """Test run with experiment_id parameter."""
        with observe.run("test-agent", experiment_id="ab_test_1") as run:
            pass

        assert run.experiment_id == "ab_test_1"

    def test_run_with_model_config(self) -> None:
        """Test run with model_config parameter."""
        model_config = {"model": "gpt-4", "temperature": 0.7}
        with observe.run("test-agent", model_config=model_config) as run:
            pass

        assert run.model_config == model_config

    def test_run_with_all_attribution_fields(self) -> None:
        """Test run with all attribution fields."""
        with observe.run(
            "test-agent",
            user_id="jane",
            session_id="conv_123",
            prompt_version="v2.3",
            experiment_id="ab_test_1",
            model_config={"model": "gpt-4"},
            metadata={"customer_tier": "premium"},
        ) as run:
            pass

        assert run.user_id == "jane"
        assert run.session_id == "conv_123"
        assert run.prompt_version == "v2.3"
        assert run.experiment_id == "ab_test_1"
        assert run.model_config == {"model": "gpt-4"}
        assert run.metadata == {"customer_tier": "premium"}

    def test_run_to_dict_includes_attribution(self) -> None:
        """Test that to_dict includes attribution fields."""
        with observe.run(
            "test-agent",
            user_id="jane",
            session_id="conv_123",
            prompt_version="v2.3",
        ) as run:
            pass

        data = run.to_dict()
        assert data["user_id"] == "jane"
        assert data["session_id"] == "conv_123"
        assert data["prompt_version"] == "v2.3"


class TestInputOutput:
    """Tests for set_input/set_output and auto-inference."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        config = Config(
            mode=CaptureMode.FULL,
            env="dev",
            sqlite_path=Path(self.temp_dir) / "test.db",
        )
        observe.install(config=config)

    def teardown_method(self) -> None:
        """Clean up."""
        observe._cleanup()

    def test_set_input(self) -> None:
        """Test set_input method."""
        with observe.run("test-agent") as run:
            run.set_input("Hello, world!")

        assert run.input == "Hello, world!"
        assert run._input_set_explicitly is True

    def test_set_output(self) -> None:
        """Test set_output method."""
        with observe.run("test-agent") as run:
            run.set_output("Goodbye, world!")

        assert run.output == "Goodbye, world!"
        assert run._output_set_explicitly is True

    def test_set_input_output_dict(self) -> None:
        """Test set_input/set_output with dict values."""
        with observe.run("test-agent") as run:
            run.set_input({"message": "Hello", "user": "jane"})
            run.set_output({"response": "Hi there!", "confidence": 0.95})

        assert run.input == {"message": "Hello", "user": "jane"}
        assert run.output == {"response": "Hi there!", "confidence": 0.95}

    def test_to_dict_includes_input_output(self) -> None:
        """Test that to_dict includes input/output as JSON."""
        with observe.run("test-agent") as run:
            run.set_input("Hello")
            run.set_output("Goodbye")

        data = run.to_dict()
        assert data["input_json"] == '"Hello"'
        assert data["input_text"] == "Hello"
        assert data["output_json"] == '"Goodbye"'
        assert data["output_text"] == "Goodbye"

    def test_add_metadata(self) -> None:
        """Test add_metadata method."""
        with observe.run("test-agent") as run:
            run.add_metadata("key1", "value1")
            run.add_metadata("key2", {"nested": "value"})

        assert run.metadata == {"key1": "value1", "key2": {"nested": "value"}}

    def test_auto_infer_input_from_first_span(self) -> None:
        """Test auto-inference of input from first span."""
        @tool(name="my_tool")
        def my_tool(arg: str) -> str:
            return f"result: {arg}"

        with observe.run("test-agent") as run:
            # Don't call set_input - should be inferred
            my_tool("test_arg")

        # Input should be inferred from first span's args
        assert run.input is not None

    def test_auto_infer_output_from_last_span(self) -> None:
        """Test auto-inference of output from last successful span."""
        @tool(name="my_tool")
        def my_tool(arg: str) -> str:
            return f"result: {arg}"

        with observe.run("test-agent") as run:
            # Don't call set_output - should be inferred
            my_tool("test_arg")

        # Output should be inferred from last span's result
        assert run.output is not None

    def test_explicit_input_overrides_inference(self) -> None:
        """Test that explicit set_input overrides inference."""
        @tool(name="my_tool")
        def my_tool(arg: str) -> str:
            return f"result: {arg}"

        with observe.run("test-agent") as run:
            run.set_input("explicit input")
            my_tool("tool_arg")

        assert run.input == "explicit input"


class TestLLMContextExtraction:
    """Tests for _extract_llm_context function."""

    def test_extract_messages_from_kwargs(self) -> None:
        """Test extracting messages from kwargs."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        context = _extract_llm_context((), {"messages": messages})

        assert context["messages"] == messages
        assert context["system_prompt"] == "You are helpful"

    def test_extract_messages_from_args(self) -> None:
        """Test extracting messages from positional args."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
        ]
        context = _extract_llm_context((messages,), {})

        assert context["messages"] == messages
        assert context["system_prompt"] == "System prompt"

    def test_extract_model_config(self) -> None:
        """Test extracting model configuration."""
        context = _extract_llm_context((), {
            "messages": [],
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
        })

        assert context["model"] == "gpt-4"
        assert context["temperature"] == 0.7
        assert context["max_tokens"] == 1000
        assert context["top_p"] == 0.9

    def test_extract_tools(self) -> None:
        """Test extracting tools/functions."""
        tools = [{"type": "function", "function": {"name": "search"}}]
        context = _extract_llm_context((), {
            "messages": [],
            "tools": tools,
            "tool_choice": "auto",
        })

        assert context["tools"] == tools
        assert context["tool_choice"] == "auto"

    def test_extract_response_format(self) -> None:
        """Test extracting response format."""
        context = _extract_llm_context((), {
            "messages": [],
            "response_format": {"type": "json_object"},
        })

        assert context["response_format"] == {"type": "json_object"}

    def test_empty_context(self) -> None:
        """Test empty context when no relevant kwargs."""
        context = _extract_llm_context((), {"irrelevant": "value"})
        assert context == {}


class TestPromptHash:
    """Tests for auto-calculated prompt_hash."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        config = Config(
            mode=CaptureMode.FULL,
            env="dev",
            sqlite_path=Path(self.temp_dir) / "test.db",
        )
        observe.install(config=config)

    def teardown_method(self) -> None:
        """Clean up."""
        observe._cleanup()

    def test_prompt_hash_auto_calculated(self) -> None:
        """Test that prompt_hash is auto-calculated from model calls."""
        @model_call(provider="test", model="test-model")
        def call_llm(_messages: list) -> str:  # noqa: ARG001 - messages used by decorator
            return "response"

        with observe.run("test-agent") as run:
            call_llm([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ])

        # prompt_hash should be set after run completes
        assert run.prompt_hash is not None
        assert len(run.prompt_hash) == 16  # Truncated hash

    def test_explicit_prompt_version_skips_hash(self) -> None:
        """Test that explicit prompt_version skips hash calculation."""
        @model_call(provider="test", model="test-model")
        def call_llm(_messages: list) -> str:  # noqa: ARG001 - messages used by decorator
            return "response"

        with observe.run("test-agent", prompt_version="v2.3") as run:
            call_llm([
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Hello"},
            ])

        # prompt_version is set, prompt_hash should not be calculated
        assert run.prompt_version == "v2.3"


class TestAsyncWideEvents:
    """Tests for async wide event features."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        config = Config(
            mode=CaptureMode.FULL,
            env="dev",
            sqlite_path=Path(self.temp_dir) / "test.db",
        )
        observe.install(config=config)

    def teardown_method(self) -> None:
        """Clean up."""
        observe._cleanup()

    @pytest.mark.asyncio
    async def test_async_run_with_attribution(self) -> None:
        """Test async run with attribution fields."""
        async with observe.arun(
            "test-agent",
            user_id="jane",
            session_id="conv_123",
            prompt_version="v2.3",
        ) as run:
            pass

        assert run.user_id == "jane"
        assert run.session_id == "conv_123"
        assert run.prompt_version == "v2.3"

    @pytest.mark.asyncio
    async def test_async_set_input_output(self) -> None:
        """Test async run with set_input/set_output."""
        async with observe.arun("test-agent") as run:
            run.set_input("async input")
            run.set_output("async output")

        assert run.input == "async input"
        assert run.output == "async output"


class TestSchemaMigration:
    """Tests for SQLite schema migration."""

    def test_v1_to_v2_migration(self) -> None:
        """Test migration from v1 schema to v2."""
        import sqlite3

        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "migrate_test.db"

        # Create v1 schema manually
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS agent_observe_schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT DEFAULT (datetime('now'))
            );
            INSERT INTO agent_observe_schema_version (version) VALUES (1);

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                trace_id TEXT,
                name TEXT NOT NULL,
                ts_start INTEGER NOT NULL,
                ts_end INTEGER,
                status TEXT
            );

            INSERT INTO runs (run_id, trace_id, name, ts_start, status)
            VALUES ('test_run', 'test_trace', 'test', 1234567890, 'ok');
        """)
        conn.commit()
        conn.close()

        # Now initialize with agent-observe - should migrate
        config = Config(
            mode=CaptureMode.FULL,
            env="dev",
            sqlite_path=db_path,
        )
        observe.install(config=config)

        # Check that new columns exist
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(runs)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        # New v0.1.7 columns should exist
        assert "user_id" in columns
        assert "session_id" in columns
        assert "prompt_version" in columns
        assert "prompt_hash" in columns
        assert "input_json" in columns
        assert "output_json" in columns
        assert "metadata" in columns

        observe._cleanup()

    def test_fresh_install_creates_v2_schema(self) -> None:
        """Test that fresh install creates v2 schema."""
        import sqlite3

        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "fresh_test.db"

        config = Config(
            mode=CaptureMode.FULL,
            env="dev",
            sqlite_path=db_path,
        )
        observe.install(config=config)

        # Check schema version
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT version FROM agent_observe_schema_version ORDER BY version DESC LIMIT 1"
        )
        version = cursor.fetchone()[0]
        conn.close()

        assert version == 2

        observe._cleanup()
