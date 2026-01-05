"""End-to-end tests for riff-observe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_observe.config import CaptureMode, Config, Environment, SinkType
from agent_observe.decorators import model_call, tool
from agent_observe.observe import Observe


class TestEndToEnd:
    """End-to-end tests for the complete flow."""

    def test_basic_run(self, temp_dir: Path) -> None:
        """Test a basic run with observe.run()."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "e2e.db",
        )

        obs = Observe()
        obs.install(config=config)

        with obs.run("test-agent", task={"input": "hello"}):
            # Simulate some work
            pass

        # Flush async writes before querying
        obs.sink.flush()

        # Verify run was recorded
        runs = obs.sink.get_runs()
        assert len(runs) == 1
        assert runs[0]["name"] == "test-agent"
        assert runs[0]["status"] == "ok"

        obs._cleanup()

    def test_run_with_tool_calls(self, temp_dir: Path) -> None:
        """Test run with @tool decorated functions."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "e2e.db",
        )

        obs = Observe()
        obs.install(config=config)

        @tool(name="add_numbers", kind="compute")
        def add(a: int, b: int) -> int:
            return a + b

        @tool(name="multiply_numbers", kind="compute")
        def multiply(a: int, b: int) -> int:
            return a * b

        with obs.run("math-agent"):
            result1 = add(2, 3)
            result2 = multiply(result1, 4)

        assert result1 == 5
        assert result2 == 20

        # Flush async writes before querying
        obs.sink.flush()

        # Verify run and spans
        runs = obs.sink.get_runs()
        assert len(runs) == 1
        assert runs[0]["tool_calls"] == 2

        spans = obs.sink.get_spans(runs[0]["run_id"])
        assert len(spans) == 2
        assert any(s["name"] == "add_numbers" for s in spans)
        assert any(s["name"] == "multiply_numbers" for s in spans)

        obs._cleanup()

    def test_run_with_model_calls(self, temp_dir: Path) -> None:
        """Test run with @model_call decorated functions."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "e2e.db",
        )

        obs = Observe()
        obs.install(config=config)

        @model_call(provider="test", model="mock-model")
        def fake_llm(prompt: str) -> str:
            return f"Response to: {prompt}"

        with obs.run("llm-agent"):
            response = fake_llm("Hello world")

        assert response == "Response to: Hello world"

        # Flush async writes before querying
        obs.sink.flush()

        runs = obs.sink.get_runs()
        assert runs[0]["model_calls"] == 1

        obs._cleanup()

    def test_run_with_error(self, temp_dir: Path) -> None:
        """Test run that raises an error."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "e2e.db",
        )

        obs = Observe()
        obs.install(config=config)

        @tool
        def failing_tool() -> None:
            raise ValueError("Tool failed!")

        with pytest.raises(ValueError, match="Tool failed"), obs.run("failing-agent"):
            failing_tool()

        # Flush async writes before querying
        obs.sink.flush()

        runs = obs.sink.get_runs()
        assert runs[0]["status"] == "error"

        # Check error event was emitted
        events = obs.sink.get_events(runs[0]["run_id"])
        error_events = [e for e in events if e["type"] == "run.error"]
        assert len(error_events) == 1

        obs._cleanup()

    def test_tool_policy_violation(self, temp_dir: Path) -> None:
        """Test that policy violations are recorded."""
        import yaml

        # Create policy file
        policy_path = temp_dir / "policy.yml"
        with open(policy_path, "w") as f:
            yaml.dump(
                {"tools": {"deny": ["blocked.*"]}},
                f,
            )

        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "e2e.db",
            policy_file=policy_path,
            fail_on_violation=False,  # Don't raise, just record
        )

        obs = Observe()
        obs.install(config=config)

        @tool(name="blocked.dangerous")
        def dangerous_tool() -> str:
            return "executed anyway"

        with obs.run("policy-test"):
            result = dangerous_tool()

        # Tool still executes (fail_on_violation=False)
        assert result == "executed anyway"

        # Flush async writes before querying
        obs.sink.flush()

        # But violation was recorded
        runs = obs.sink.get_runs()
        assert runs[0]["policy_violations"] == 1
        assert runs[0]["risk_score"] >= 40  # POLICY_VIOLATION adds 40

        events = obs.sink.get_events(runs[0]["run_id"])
        violation_events = [e for e in events if e["type"] == "policy.violation"]
        assert len(violation_events) == 1

        obs._cleanup()

    def test_emit_event(self, temp_dir: Path) -> None:
        """Test emitting custom events."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "e2e.db",
        )

        obs = Observe()
        obs.install(config=config)

        with obs.run("event-test"):
            obs.emit_event("user.action", {"action": "clicked", "target": "button"})
            obs.emit_event("custom.metric", {"value": 42})

        # Flush async writes before querying
        obs.sink.flush()

        runs = obs.sink.get_runs()
        events = obs.sink.get_events(runs[0]["run_id"])

        custom_events = [
            e for e in events if e["type"] in ("user.action", "custom.metric")
        ]
        assert len(custom_events) == 2

        obs._cleanup()

    def test_emit_artifact(self, temp_dir: Path) -> None:
        """Test emitting artifacts."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "e2e.db",
        )

        obs = Observe()
        obs.install(config=config)

        with obs.run("artifact-test"):
            obs.emit_artifact("report", {"summary": "All tests passed"})

        # Flush async writes before querying
        obs.sink.flush()

        runs = obs.sink.get_runs()
        events = obs.sink.get_events(runs[0]["run_id"])

        artifact_events = [e for e in events if e["type"] == "artifact"]
        assert len(artifact_events) == 1

        # In metadata_only mode, content is not stored
        payload = artifact_events[0]["payload"]
        assert "content_hash" in payload
        assert "content_size" in payload
        assert "content" not in payload

        obs._cleanup()

    def test_risk_score_calculation(self, temp_dir: Path) -> None:
        """Test that risk score is calculated correctly."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "e2e.db",
            latency_budget_ms=100,  # Very short budget
        )

        obs = Observe()
        obs.install(config=config)

        @tool
        def slow_tool() -> str:
            import time

            time.sleep(0.2)  # 200ms, exceeds budget
            return "done"

        with obs.run("slow-agent"):
            slow_tool()

        # Flush async writes before querying
        obs.sink.flush()

        runs = obs.sink.get_runs()

        # Should have LATENCY_BREACH tag
        assert "LATENCY_BREACH" in (runs[0]["eval_tags"] or [])
        assert runs[0]["risk_score"] >= 10

        obs._cleanup()

    def test_nested_runs_isolated(self, temp_dir: Path) -> None:
        """Test that nested runs are isolated."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "e2e.db",
        )

        obs = Observe()
        obs.install(config=config)

        @tool
        def outer_tool() -> str:
            return "outer"

        @tool
        def inner_tool() -> str:
            return "inner"

        with obs.run("outer-run"):
            outer_tool()
            with obs.run("inner-run"):
                inner_tool()
            outer_tool()

        # Flush async writes before querying
        obs.sink.flush()

        runs = obs.sink.get_runs()
        assert len(runs) == 2

        outer_run = next(r for r in runs if r["name"] == "outer-run")
        inner_run = next(r for r in runs if r["name"] == "inner-run")

        assert outer_run["tool_calls"] == 2
        assert inner_run["tool_calls"] == 1

        obs._cleanup()

    def test_jsonl_sink(self, temp_dir: Path) -> None:
        """Test end-to-end with JSONL sink."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.PROD,
            sink_type=SinkType.JSONL,
            jsonl_dir=temp_dir / "traces",
        )

        obs = Observe()
        obs.install(config=config)

        @tool
        def simple_tool() -> str:
            return "result"

        with obs.run("jsonl-test"):
            simple_tool()

        # Force flush
        obs.sink.flush()

        # Verify files exist
        runs_dir = temp_dir / "traces" / "runs"
        assert runs_dir.exists()
        assert len(list(runs_dir.glob("*.jsonl"))) > 0

        obs._cleanup()

    @pytest.mark.asyncio
    async def test_async_run(self, temp_dir: Path) -> None:
        """Test async run with arun()."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "async_e2e.db",
        )

        obs = Observe()
        obs.install(config=config)

        async with obs.arun("async-agent", task={"input": "hello"}):
            # Simulate some async work
            import asyncio
            await asyncio.sleep(0.01)

        # Flush async writes before querying
        obs.sink.flush()

        # Verify run was recorded
        runs = obs.sink.get_runs()
        assert len(runs) == 1
        assert runs[0]["name"] == "async-agent"
        assert runs[0]["status"] == "ok"

        obs._cleanup()

    @pytest.mark.asyncio
    async def test_async_tools(self, temp_dir: Path) -> None:
        """Test async tool functions."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "async_tools.db",
        )

        obs = Observe()
        obs.install(config=config)

        @tool(name="async_fetch", kind="http")
        async def async_fetch(url: str) -> dict:
            import asyncio
            await asyncio.sleep(0.01)
            return {"url": url, "status": "ok"}

        @tool(name="async_process", kind="compute")
        async def async_process(data: dict) -> str:
            import asyncio
            await asyncio.sleep(0.01)
            return f"Processed: {data}"

        async with obs.arun("async-tools-agent"):
            result1 = await async_fetch("https://example.com")
            result2 = await async_process(result1)

        assert result1 == {"url": "https://example.com", "status": "ok"}
        assert "Processed" in result2

        # Flush async writes before querying
        obs.sink.flush()

        runs = obs.sink.get_runs()
        assert len(runs) == 1
        assert runs[0]["tool_calls"] == 2

        spans = obs.sink.get_spans(runs[0]["run_id"])
        assert len(spans) == 2
        assert any(s["name"] == "async_fetch" for s in spans)
        assert any(s["name"] == "async_process" for s in spans)

        obs._cleanup()

    @pytest.mark.asyncio
    async def test_async_model_call(self, temp_dir: Path) -> None:
        """Test async model call functions."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "async_model.db",
        )

        obs = Observe()
        obs.install(config=config)

        @model_call(provider="openai", model="gpt-4")
        async def async_llm(prompt: str) -> dict:
            import asyncio
            await asyncio.sleep(0.01)
            return {
                "response": f"Response to: {prompt}",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }

        async with obs.arun("async-model-agent"):
            result = await async_llm("Hello world")

        assert "Response to:" in result["response"]

        # Flush async writes before querying
        obs.sink.flush()

        runs = obs.sink.get_runs()
        assert runs[0]["model_calls"] == 1

        spans = obs.sink.get_spans(runs[0]["run_id"])
        assert len(spans) == 1
        # Token tracking should work
        assert spans[0]["attrs"].get("tokens.total") == 30

        obs._cleanup()

    @pytest.mark.asyncio
    async def test_nested_async_spans(self, temp_dir: Path) -> None:
        """Test that nested async tool calls track parent spans correctly."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "nested_async.db",
        )

        obs = Observe()
        obs.install(config=config)

        @tool(name="inner_tool")
        async def inner_tool() -> str:
            import asyncio
            await asyncio.sleep(0.01)
            return "inner result"

        @tool(name="outer_tool")
        async def outer_tool() -> str:
            result = await inner_tool()
            return f"outer: {result}"

        async with obs.arun("nested-async-agent"):
            await outer_tool()

        # Flush async writes before querying
        obs.sink.flush()

        runs = obs.sink.get_runs()
        spans = obs.sink.get_spans(runs[0]["run_id"])

        assert len(spans) == 2

        # Find inner and outer spans
        inner_span = next(s for s in spans if s["name"] == "inner_tool")
        outer_span = next(s for s in spans if s["name"] == "outer_tool")

        # Inner span should have outer span as parent
        assert inner_span["parent_span_id"] == outer_span["span_id"]

        obs._cleanup()

    def test_error_context_in_full_mode(self, temp_dir: Path) -> None:
        """Test that error context includes traceback in full mode."""
        config = Config(
            mode=CaptureMode.FULL,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "error_context.db",
        )

        obs = Observe()
        obs.install(config=config)

        @tool
        def error_tool(x: int) -> int:
            if x < 0:
                raise ValueError("Negative input not allowed")
            return x * 2

        with pytest.raises(ValueError, match="Negative input"), obs.run("error-agent"):
            error_tool(-1)

        obs.sink.flush()

        runs = obs.sink.get_runs()
        spans = obs.sink.get_spans(runs[0]["run_id"])
        assert len(spans) == 1

        # Error context should be present in full mode
        error_context = spans[0]["attrs"].get("error_context", {})
        assert error_context.get("type") == "ValueError"
        assert "Negative input" in error_context.get("message", "")
        # Traceback should be present in full mode
        assert "traceback" in error_context
        # Input should be stored in full mode
        assert "input" in error_context

        obs._cleanup()

    def test_error_context_in_metadata_mode(self, temp_dir: Path) -> None:
        """Test that error context only has basic info in metadata mode."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "error_metadata.db",
        )

        obs = Observe()
        obs.install(config=config)

        @tool
        def error_tool() -> int:
            raise RuntimeError("Something went wrong")

        with pytest.raises(RuntimeError), obs.run("error-agent"):
            error_tool()

        obs.sink.flush()

        runs = obs.sink.get_runs()
        spans = obs.sink.get_spans(runs[0]["run_id"])
        assert len(spans) == 1

        # Error context should have basic info but no traceback/input
        error_context = spans[0]["attrs"].get("error_context", {})
        assert error_context.get("type") == "RuntimeError"
        assert "traceback" not in error_context
        assert "input" not in error_context

        obs._cleanup()

    def test_streaming_model_call(self, temp_dir: Path) -> None:
        """Test that streaming LLM responses are wrapped and metrics recorded."""
        from agent_observe.decorators import SyncStreamWrapper

        config = Config(
            mode=CaptureMode.FULL,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "streaming.db",
        )

        obs = Observe()
        obs.install(config=config)

        def mock_stream():
            """Simulate a streaming response."""
            yield {"delta": {"text": "Hello"}}
            yield {"delta": {"text": " World"}}
            yield {"delta": {"text": "!"}}

        @model_call(provider="test", model="streaming-model")
        def streaming_llm() -> Any:
            return mock_stream()

        with obs.run("streaming-agent"):
            stream = streaming_llm()
            # The result should be a SyncStreamWrapper
            assert isinstance(stream, SyncStreamWrapper)

            # Consume the stream
            chunks = list(stream)
            assert len(chunks) == 3

        obs.sink.flush()

        runs = obs.sink.get_runs()
        spans = obs.sink.get_spans(runs[0]["run_id"])
        assert len(spans) == 1

        # Check streaming metrics were recorded
        attrs = spans[0]["attrs"]
        assert attrs.get("streaming") is True
        assert attrs.get("chunk_count") == 3
        assert "ts_first_token" in attrs
        assert "ts_last_token" in attrs
        # Output should be accumulated in full mode
        assert attrs.get("output") == "Hello World!"

        obs._cleanup()

    def test_streaming_model_call_evidence_mode(self, temp_dir: Path) -> None:
        """Test streaming in evidence_only mode with size limit."""
        from agent_observe.decorators import SyncStreamWrapper

        config = Config(
            mode=CaptureMode.EVIDENCE_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "streaming_evidence.db",
        )

        obs = Observe()
        obs.install(config=config)

        def mock_stream():
            yield {"delta": {"text": "Short response"}}

        @model_call(provider="test", model="streaming-model")
        def streaming_llm() -> Any:
            return mock_stream()

        with obs.run("streaming-agent"):
            stream = streaming_llm()
            list(stream)  # Consume

        obs.sink.flush()

        runs = obs.sink.get_runs()
        spans = obs.sink.get_spans(runs[0]["run_id"])

        # Short output should be stored in evidence mode
        attrs = spans[0]["attrs"]
        assert attrs.get("output") == "Short response"

        obs._cleanup()

    def test_streaming_model_call_error(self, temp_dir: Path) -> None:
        """Test that streaming errors are properly finalized."""
        from agent_observe.decorators import SyncStreamWrapper

        config = Config(
            mode=CaptureMode.FULL,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "streaming_error.db",
        )

        obs = Observe()
        obs.install(config=config)

        def error_stream():
            yield {"delta": {"text": "Starting..."}}
            raise RuntimeError("Stream connection lost")

        @model_call(provider="test", model="streaming-model")
        def streaming_llm() -> Any:
            return error_stream()

        with pytest.raises(RuntimeError, match="connection lost"), obs.run("streaming-error"):
            stream = streaming_llm()
            list(stream)  # Consume until error

        obs.sink.flush()

        runs = obs.sink.get_runs()
        spans = obs.sink.get_spans(runs[0]["run_id"])
        assert len(spans) == 1

        # Span should be finalized with error status
        attrs = spans[0]["attrs"]
        assert attrs.get("streaming") is True
        assert attrs.get("chunk_count") == 1  # Got one chunk before error
        assert spans[0]["status"] == "error"

        obs._cleanup()

    def test_per_run_mode_override(self, temp_dir: Path) -> None:
        """Test per-run capture mode override."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,  # Global: metadata only
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "per_run_mode.db",
        )

        obs = Observe()
        obs.install(config=config)

        @tool(name="test_tool")
        def test_tool(arg: str) -> str:
            return f"result: {arg}"

        # Run 1: Use global config (metadata_only)
        with obs.run("run-metadata"):
            test_tool("hello")

        # Run 2: Override to full mode
        with obs.run("run-full", mode="full"):
            test_tool("world")

        obs.sink.flush()

        runs = obs.sink.get_runs()
        assert len(runs) == 2

        # Find runs by name
        run_metadata = next(r for r in runs if r["name"] == "run-metadata")
        run_full = next(r for r in runs if r["name"] == "run-full")

        # Check capture modes recorded correctly
        assert run_metadata["capture_mode"] == "metadata_only"
        assert run_full["capture_mode"] == "full"

        # Check spans - metadata_only shouldn't have args/result, full should
        spans_metadata = obs.sink.get_spans(run_metadata["run_id"])
        spans_full = obs.sink.get_spans(run_full["run_id"])

        # In metadata_only mode, args and result should not be stored
        assert "args" not in spans_metadata[0]["attrs"]
        assert "result" not in spans_metadata[0]["attrs"]

        # In full mode, args and result should be stored
        assert "args" in spans_full[0]["attrs"]
        assert "result" in spans_full[0]["attrs"]

        obs._cleanup()

    def test_per_run_policy_override(self, temp_dir: Path) -> None:
        """Test per-run policy file override."""
        import yaml

        # Create two different policy files
        strict_policy_path = temp_dir / "strict_policy.yml"
        with open(strict_policy_path, "w") as f:
            yaml.dump({"tools": {"deny": ["dangerous.*"]}}, f)

        lenient_policy_path = temp_dir / "lenient_policy.yml"
        with open(lenient_policy_path, "w") as f:
            yaml.dump({"tools": {"allow": ["*"]}}, f)

        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "per_run_policy.db",
            policy_file=strict_policy_path,  # Global: strict policy
            fail_on_violation=False,
        )

        obs = Observe()
        obs.install(config=config)

        @tool(name="dangerous.action")
        def dangerous_action() -> str:
            return "executed"

        # Run 1: Use global strict policy (should record violation)
        with obs.run("run-strict"):
            dangerous_action()

        # Run 2: Override with lenient policy (no violation)
        with obs.run("run-lenient", policy_file=lenient_policy_path):
            dangerous_action()

        obs.sink.flush()

        runs = obs.sink.get_runs()
        run_strict = next(r for r in runs if r["name"] == "run-strict")
        run_lenient = next(r for r in runs if r["name"] == "run-lenient")

        # Strict run should have policy violation
        assert run_strict["policy_violations"] == 1

        # Lenient run should have no policy violations
        assert run_lenient["policy_violations"] == 0

        obs._cleanup()

    def test_per_run_fail_on_violation_override(self, temp_dir: Path) -> None:
        """Test per-run fail_on_violation override."""
        import yaml

        from agent_observe.policy import PolicyViolationError

        policy_path = temp_dir / "policy.yml"
        with open(policy_path, "w") as f:
            yaml.dump({"tools": {"deny": ["blocked.*"]}}, f)

        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "per_run_fail.db",
            policy_file=policy_path,
            fail_on_violation=False,  # Global: don't fail
        )

        obs = Observe()
        obs.install(config=config)

        @tool(name="blocked.tool")
        def blocked_tool() -> str:
            return "executed"

        # Run 1: Use global setting (don't fail)
        with obs.run("run-no-fail"):
            result = blocked_tool()
            assert result == "executed"  # Should execute

        # Run 2: Override to fail on violation
        with pytest.raises(PolicyViolationError):
            with obs.run("run-fail", fail_on_violation=True):
                blocked_tool()  # Should raise

        obs._cleanup()

    def test_per_run_latency_budget_override(self, temp_dir: Path) -> None:
        """Test per-run latency budget override."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "per_run_latency.db",
            latency_budget_ms=100000,  # Global: 100 seconds (won't breach)
        )

        obs = Observe()
        obs.install(config=config)

        # Run 1: Use global latency budget (100s, won't breach)
        with obs.run("run-high-budget"):
            pass

        # Run 2: Override with tiny budget (1ms, will breach)
        with obs.run("run-low-budget", latency_budget_ms=1):
            import time

            time.sleep(0.01)  # Sleep 10ms to ensure breach

        obs.sink.flush()

        runs = obs.sink.get_runs()
        run_high = next(r for r in runs if r["name"] == "run-high-budget")
        run_low = next(r for r in runs if r["name"] == "run-low-budget")

        # High budget run should have no risk from latency
        # Low budget run should have risk from latency breach
        # LATENCY_BREACH adds 10 to risk score
        assert run_low["risk_score"] >= 10  # Latency breach adds risk
        assert run_high["risk_score"] < run_low["risk_score"]

        obs._cleanup()

    def test_health_check(self, temp_dir: Path) -> None:
        """Test observe.health() returns proper status and metrics."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "health_check.db",
        )

        obs = Observe()

        # Health before install
        assert obs.health() == {"status": "not_installed"}

        obs.install(config=config)

        # Health after install (initial state)
        health = obs.health()
        assert health["status"] == "healthy"
        assert health["queue_depth"] == 0
        assert health["writes_total"] == 0
        assert health["writes_failed"] == 0
        assert health["retries_total"] == 0
        assert "avg_write_latency_ms" in health
        assert "queue_high_watermark" in health
        assert "last_write_ts" in health

        # After some writes
        with obs.run("test-agent"):
            pass

        obs.sink.flush()

        health = obs.health()
        assert health["status"] == "healthy"
        assert health["writes_total"] >= 1  # At least the run was written

        obs._cleanup()

    def test_sink_metrics(self, temp_dir: Path) -> None:
        """Test that sink metrics are tracked correctly."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "sink_metrics.db",
        )

        obs = Observe()
        obs.install(config=config)

        # Initial metrics
        assert obs.sink.queue_depth == 0
        assert obs.sink.metrics.writes_total == 0

        @tool(name="test_tool")
        def test_tool() -> str:
            return "done"

        with obs.run("metrics-agent"):
            test_tool()
            test_tool()

        obs.sink.flush()

        # After writes, metrics should be updated
        metrics = obs.sink.metrics
        assert metrics.writes_total >= 1  # At least run + spans
        assert metrics.write_latency_ms_count >= 1
        assert metrics.last_write_ts > 0

        obs._cleanup()

    def test_span_memory_limit(self, temp_dir: Path) -> None:
        """Test that spans memory is managed when limit is reached."""
        config = Config(
            mode=CaptureMode.METADATA_ONLY,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=temp_dir / "span_limit.db",
        )

        obs = Observe()
        obs.install(config=config)

        @tool(name="quick_tool")
        def quick_tool(i: int) -> int:
            return i

        # Test that default limit is reasonable
        with obs.run("default-limit") as run:
            assert run._max_spans_in_memory == 10000  # Default

        obs.sink.flush()
        obs._cleanup()

    def test_span_flush_tracking(self, temp_dir: Path) -> None:
        """Test that span count tracking works correctly."""
        from agent_observe.context import RunContext, SpanContext, SpanKind, now_ms

        # Test that flush is only triggered when _observe is set
        # (it needs a sink to write to)
        run = RunContext(
            run_id="test-run-id",
            trace_id="test-trace-id",
            name="test",
            ts_start=now_ms(),
        )
        run._max_spans_in_memory = 3

        # Add spans without _observe set - they accumulate (no flush)
        for i in range(7):
            span = SpanContext(
                span_id=f"span-{i}",
                run_id=run.run_id,
                parent_span_id=None,
                kind=SpanKind.TOOL,
                name=f"tool-{i}",
                ts_start=now_ms(),
            )
            run.add_span(span)

        # Without _observe, spans just accumulate (flush is a no-op)
        # The threshold triggers _flush_spans calls, but they return early
        assert run.tool_calls == 7
        assert len(run._spans) == 7  # All still in memory (no sink to flush to)
