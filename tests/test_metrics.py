"""Tests for metrics and eval engine."""

from __future__ import annotations

from agent_observe.context import RunContext, SpanContext, SpanKind, SpanStatus
from agent_observe.metrics import (
    RunMetrics,
    compute_loop_suspect_count,
    compute_risk_score,
    evaluate_run,
)


class TestRunMetrics:
    """Tests for RunMetrics dataclass."""

    def test_tool_success_rate_all_success(self) -> None:
        """Test success rate with all successful tools."""
        metrics = RunMetrics(tool_success_count=10, tool_error_count=0)
        assert metrics.tool_success_rate == 1.0

    def test_tool_success_rate_mixed(self) -> None:
        """Test success rate with mixed results."""
        metrics = RunMetrics(tool_success_count=8, tool_error_count=2)
        assert metrics.tool_success_rate == 0.8

    def test_tool_success_rate_no_tools(self) -> None:
        """Test success rate with no tool calls."""
        metrics = RunMetrics(tool_success_count=0, tool_error_count=0)
        assert metrics.tool_success_rate == 1.0  # Default to 1.0 if no tools


class TestLoopDetection:
    """Tests for loop detection."""

    def test_no_loops(self) -> None:
        """Test with no repeated calls."""
        hashes = ["tool1:abc", "tool2:def", "tool3:ghi"]
        assert compute_loop_suspect_count(hashes) == 0

    def test_single_loop(self) -> None:
        """Test with one loop (3 repetitions)."""
        hashes = ["tool1:abc", "tool1:abc", "tool1:abc", "tool2:def"]
        assert compute_loop_suspect_count(hashes) == 1

    def test_multiple_loops(self) -> None:
        """Test with multiple distinct loops."""
        hashes = [
            "tool1:abc",
            "tool1:abc",
            "tool1:abc",  # Loop 1
            "tool2:def",
            "tool2:def",
            "tool2:def",  # Loop 2
        ]
        assert compute_loop_suspect_count(hashes) == 2

    def test_below_threshold(self) -> None:
        """Test with repetitions below threshold."""
        hashes = ["tool1:abc", "tool1:abc"]  # Only 2 repetitions
        assert compute_loop_suspect_count(hashes) == 0


class TestRiskScore:
    """Tests for risk score calculation."""

    def test_zero_risk(self) -> None:
        """Test run with no issues has zero risk."""
        metrics = RunMetrics(
            tool_success_count=10,
            tool_error_count=0,
            policy_violation_count=0,
            loop_suspect_count=0,
            retry_count=0,
            latency_ms_total=5000,
        )

        score, tags = compute_risk_score(metrics, latency_budget_ms=20000)

        assert score == 0
        assert tags == []

    def test_policy_violation_adds_risk(self) -> None:
        """Test that policy violations add risk."""
        metrics = RunMetrics(
            tool_success_count=10,
            policy_violation_count=1,
        )

        score, tags = compute_risk_score(metrics)

        assert score == 40
        assert "POLICY_VIOLATION" in tags

    def test_tool_failure_adds_risk(self) -> None:
        """Test that low success rate adds risk."""
        metrics = RunMetrics(
            tool_success_count=8,
            tool_error_count=3,  # 72.7% success rate
        )

        score, tags = compute_risk_score(metrics)

        assert score == 25
        assert "TOOL_FAILURE" in tags

    def test_loop_detected_adds_risk(self) -> None:
        """Test that suspected loops add risk."""
        metrics = RunMetrics(
            tool_success_count=10,
            loop_suspect_count=3,
        )

        score, tags = compute_risk_score(metrics)

        assert score == 15
        assert "LOOP_SUSPECTED" in tags

    def test_retry_storm_adds_risk(self) -> None:
        """Test that retry storms add risk."""
        metrics = RunMetrics(
            tool_success_count=10,
            retry_count=5,
        )

        score, tags = compute_risk_score(metrics)

        assert score == 10
        assert "RETRY_STORM" in tags

    def test_latency_breach_adds_risk(self) -> None:
        """Test that latency breach adds risk."""
        metrics = RunMetrics(
            tool_success_count=10,
            latency_ms_total=25000,
        )

        score, tags = compute_risk_score(metrics, latency_budget_ms=20000)

        assert score == 10
        assert "LATENCY_BREACH" in tags

    def test_multiple_issues_stack(self) -> None:
        """Test that multiple issues stack up."""
        metrics = RunMetrics(
            tool_success_count=7,
            tool_error_count=3,  # Low success rate
            policy_violation_count=1,  # Policy violation
            loop_suspect_count=3,  # Loop detected
            retry_count=6,  # Retry storm
            latency_ms_total=25000,  # Latency breach
        )

        score, tags = compute_risk_score(metrics, latency_budget_ms=20000)

        # 40 + 25 + 15 + 10 + 10 = 100
        assert score == 100
        assert len(tags) == 5

    def test_score_clamped_to_100(self) -> None:
        """Test that score is clamped to 100."""
        metrics = RunMetrics(
            tool_success_count=0,
            tool_error_count=10,
            policy_violation_count=5,
            loop_suspect_count=10,
            retry_count=20,
            latency_ms_total=100000,
        )

        score, _ = compute_risk_score(metrics)

        assert score == 100


class TestEvaluateRun:
    """Tests for full run evaluation."""

    def test_evaluate_empty_run(self) -> None:
        """Test evaluating run with no activity."""
        run = RunContext(
            run_id="test-id",
            trace_id="trace-id",
            name="test-run",
            ts_start=1000000,
        )
        run.ts_end = 1001000  # 1 second duration

        result = evaluate_run(run)

        assert result.risk_score == 0
        assert result.eval_tags == []
        assert result.metrics["tool_call_count"] == 0

    def test_evaluate_run_with_spans(self) -> None:
        """Test evaluating run with tool spans."""
        run = RunContext(
            run_id="test-id",
            trace_id="trace-id",
            name="test-run",
            ts_start=1000000,
        )

        # Add successful tool spans
        for i in range(5):
            span = SpanContext(
                span_id=f"span-{i}",
                run_id="test-id",
                parent_span_id=None,
                kind=SpanKind.TOOL,
                name=f"tool-{i}",
                ts_start=1000000 + i * 100,
                ts_end=1000000 + i * 100 + 50,
                status=SpanStatus.OK,
            )
            run.add_span(span)

        run.ts_end = 1001000

        result = evaluate_run(run)

        assert result.risk_score == 0
        assert result.metrics["tool_call_count"] == 5
        assert result.metrics["tool_success_rate"] == 1.0

    def test_evaluate_run_with_failures(self) -> None:
        """Test evaluating run with tool failures."""
        run = RunContext(
            run_id="test-id",
            trace_id="trace-id",
            name="test-run",
            ts_start=1000000,
        )

        # Add mixed tool spans
        for i in range(10):
            span = SpanContext(
                span_id=f"span-{i}",
                run_id="test-id",
                parent_span_id=None,
                kind=SpanKind.TOOL,
                name=f"tool-{i}",
                ts_start=1000000 + i * 100,
                ts_end=1000000 + i * 100 + 50,
                status=SpanStatus.ERROR if i < 2 else SpanStatus.OK,  # 2 failures
            )
            run.add_span(span)

        run.ts_end = 1002000

        result = evaluate_run(run)

        assert result.metrics["tool_success_rate"] == 0.8
        assert "TOOL_FAILURE" in result.eval_tags
