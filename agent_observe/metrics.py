"""
Metrics and eval engine for agent-observe.

Provides label-free evaluation based on behavioral signals:
- Risk scoring (0-100)
- Automatic eval tags
- Run aggregation
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_observe.context import RunContext


# Risk score weights
RISK_WEIGHTS = {
    "POLICY_VIOLATION": 40,
    "TOOL_FAILURE": 25,
    "LOOP_SUSPECTED": 15,
    "RETRY_STORM": 10,
    "LATENCY_BREACH": 10,
}


@dataclass
class EvalResult:
    """Result of run evaluation."""

    risk_score: int
    eval_tags: list[str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "eval_tags": self.eval_tags,
            "metrics": self.metrics,
        }


@dataclass
class RunMetrics:
    """Aggregated metrics for a run."""

    tool_call_count: int = 0
    model_call_count: int = 0
    tool_success_count: int = 0
    tool_error_count: int = 0
    retry_count: int = 0
    loop_suspect_count: int = 0
    policy_violation_count: int = 0
    latency_ms_total: int = 0

    @property
    def tool_success_rate(self) -> float:
        """Calculate tool success rate."""
        total = self.tool_success_count + self.tool_error_count
        if total == 0:
            return 1.0
        return self.tool_success_count / total


def compute_loop_suspect_count(tool_call_hashes: list[str]) -> int:
    """
    Count suspected loops based on repeated tool+args combinations.

    A loop is suspected when the same tool+args_hash appears 3+ times.

    Args:
        tool_call_hashes: List of tool call hashes (tool_name:args_hash).

    Returns:
        Number of distinct tool+args combinations that appear 3+ times.
    """
    counts = Counter(tool_call_hashes)
    return sum(1 for count in counts.values() if count >= 3)


def compute_run_metrics(run: RunContext) -> RunMetrics:
    """
    Compute metrics from a completed run.

    Args:
        run: Completed RunContext.

    Returns:
        Aggregated RunMetrics.
    """
    from agent_observe.context import SpanKind, SpanStatus

    tool_success_count = 0
    tool_error_count = 0

    for span in run.spans:
        if span.kind == SpanKind.TOOL:
            if span.status == SpanStatus.OK:
                tool_success_count += 1
            else:
                tool_error_count += 1

    loop_suspect_count = compute_loop_suspect_count(run.tool_call_hashes)
    latency_ms = run.duration_ms or 0

    return RunMetrics(
        tool_call_count=run.tool_calls,
        model_call_count=run.model_calls,
        tool_success_count=tool_success_count,
        tool_error_count=tool_error_count,
        retry_count=run.retry_count,
        loop_suspect_count=loop_suspect_count,
        policy_violation_count=run.policy_violations,
        latency_ms_total=latency_ms,
    )


def compute_risk_score(
    metrics: RunMetrics, latency_budget_ms: int = 20000
) -> tuple[int, list[str]]:
    """
    Compute risk score and eval tags from metrics.

    Risk Score Calculation:
    - +40: Any policy violations -> POLICY_VIOLATION
    - +25: Tool success rate < 0.9 -> TOOL_FAILURE
    - +15: Loop suspect count >= 3 -> LOOP_SUSPECTED
    - +10: Retry count >= 5 -> RETRY_STORM
    - +10: Latency > budget -> LATENCY_BREACH

    Score is clamped to 0-100.

    Args:
        metrics: RunMetrics to evaluate.
        latency_budget_ms: Latency budget in milliseconds.

    Returns:
        Tuple of (risk_score, eval_tags).
    """
    score = 0
    tags: list[str] = []

    # Policy violations
    if metrics.policy_violation_count > 0:
        score += RISK_WEIGHTS["POLICY_VIOLATION"]
        tags.append("POLICY_VIOLATION")

    # Tool failures
    if metrics.tool_success_rate < 0.9:
        score += RISK_WEIGHTS["TOOL_FAILURE"]
        tags.append("TOOL_FAILURE")

    # Loop detection
    if metrics.loop_suspect_count >= 3:
        score += RISK_WEIGHTS["LOOP_SUSPECTED"]
        tags.append("LOOP_SUSPECTED")

    # Retry storm
    if metrics.retry_count >= 5:
        score += RISK_WEIGHTS["RETRY_STORM"]
        tags.append("RETRY_STORM")

    # Latency breach
    if metrics.latency_ms_total > latency_budget_ms:
        score += RISK_WEIGHTS["LATENCY_BREACH"]
        tags.append("LATENCY_BREACH")

    # Clamp to 0-100
    score = max(0, min(100, score))

    return score, tags


def evaluate_run(run: RunContext, latency_budget_ms: int = 20000) -> EvalResult:
    """
    Evaluate a completed run and compute risk score.

    Args:
        run: Completed RunContext.
        latency_budget_ms: Latency budget in milliseconds.

    Returns:
        EvalResult with risk score, tags, and metrics.
    """
    metrics = compute_run_metrics(run)
    risk_score, eval_tags = compute_risk_score(metrics, latency_budget_ms)

    return EvalResult(
        risk_score=risk_score,
        eval_tags=eval_tags,
        metrics={
            "tool_call_count": metrics.tool_call_count,
            "model_call_count": metrics.model_call_count,
            "tool_success_rate": round(metrics.tool_success_rate, 3),
            "retry_count": metrics.retry_count,
            "loop_suspect_count": metrics.loop_suspect_count,
            "policy_violation_count": metrics.policy_violation_count,
            "latency_ms_total": metrics.latency_ms_total,
        },
    )
