"""
Core runtime for agent-observe.

Provides the main API:
- observe.install() - Initialize observability
- observe.run() - Create run context
- observe.emit_event() - Emit custom events
- observe.emit_artifact() - Emit artifacts
"""

from __future__ import annotations

import atexit
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Callable, TypeVar

from agent_observe.config import CaptureMode, Config, load_config
from agent_observe.pii import PIIConfig
from agent_observe.context import (
    RunContext,
    RunStatus,
    generate_trace_id,
    generate_uuid,
    get_current_run,
    now_ms,
    reset_current_run,
    set_current_run,
)
from agent_observe.hashing import hash_content
from agent_observe.hooks.context import RunEndContext, RunStartContext
from agent_observe.hooks.registry import HookRegistry
from agent_observe.metrics import evaluate_run
from agent_observe.policy import PolicyEngine, load_policy
from agent_observe.replay import ReplayCache
from agent_observe.sinks.base import Sink, create_sink

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Patterns that may indicate secrets in error messages
SECRET_PATTERNS = [
    r"(?i)(api[_-]?key|apikey)[=:\s]+['\"]?[\w-]+",
    r"(?i)(secret|password|token|bearer)[=:\s]+['\"]?[\w-]+",
    r"(?i)(authorization)[=:\s]+['\"]?[\w-]+",
    r"(?i)(aws[_-]?access|aws[_-]?secret)[=:\s]+[\w-]+",
    r"postgresql://[^@]+:[^@]+@",  # DB URLs with passwords
    r"mysql://[^@]+:[^@]+@",
]


def sanitize_error_message(error: Exception, max_length: int = 500) -> str:
    """
    Sanitize an error message to remove potential secrets.

    Args:
        error: The exception to sanitize.
        max_length: Maximum length of the error message.

    Returns:
        Sanitized error message safe for logging/storage.
    """
    import re

    message = str(error)

    # SECURITY: Redact FIRST, then truncate to avoid exposing secrets after truncation point
    for pattern in SECRET_PATTERNS:
        message = re.sub(pattern, "[REDACTED]", message)

    # Truncate after redaction
    if len(message) > max_length:
        message = message[:max_length] + "...[truncated]"

    return message


class Observe:
    """
    Main observability runtime.

    Usage:
        from agent_observe import observe

        observe.install()

        with observe.run("my-agent"):
            # agent code
            pass
    """

    def __init__(self) -> None:
        self._installed = False
        self._config: Config | None = None
        self._sink: Sink | None = None
        self._policy_engine: PolicyEngine | None = None
        self._replay_cache: ReplayCache | None = None
        self._hooks: HookRegistry | None = None

    @property
    def config(self) -> Config:
        """Get current configuration (raises if not installed)."""
        if self._config is None:
            raise RuntimeError("observe.install() has not been called")
        return self._config

    @property
    def sink(self) -> Sink:
        """Get current sink (raises if not installed)."""
        if self._sink is None:
            raise RuntimeError("observe.install() has not been called")
        return self._sink

    @property
    def policy_engine(self) -> PolicyEngine:
        """Get policy engine (raises if not installed)."""
        if self._policy_engine is None:
            raise RuntimeError("observe.install() has not been called")
        return self._policy_engine

    @property
    def replay_cache(self) -> ReplayCache:
        """Get replay cache (raises if not installed)."""
        if self._replay_cache is None:
            raise RuntimeError("observe.install() has not been called")
        return self._replay_cache

    @property
    def hooks(self) -> HookRegistry:
        """
        Get hook registry for registering lifecycle hooks.

        Usage:
            @observe.hooks.before_tool
            def my_hook(ctx):
                pass

            @observe.hooks.on_run_end
            def log_completion(ctx):
                print(f"Run completed: {ctx.status}")
        """
        if self._hooks is None:
            # Create a default registry even if not installed
            # This allows hooks to be registered before install()
            self._hooks = HookRegistry()
        return self._hooks

    @property
    def is_installed(self) -> bool:
        """Check if observability is installed."""
        return self._installed

    @property
    def is_enabled(self) -> bool:
        """Check if observability is enabled (not OFF mode)."""
        return self._installed and self._config is not None and self._config.mode != CaptureMode.OFF

    def health(self) -> dict[str, Any]:
        """
        Get health status of the observability system.

        Returns:
            dict with status, queue_depth, metrics, and other health info.

        Example:
            >>> obs.health()
            {
                'status': 'healthy',
                'queue_depth': 42,
                'writes_total': 1000,
                'writes_failed': 2,
                'avg_write_latency_ms': 5.3,
                ...
            }
        """
        if not self._installed:
            return {"status": "not_installed"}

        if self._sink is None:
            return {"status": "no_sink"}

        metrics = self._sink.metrics
        queue_depth = self._sink.queue_depth

        # Determine health status
        status = "healthy"
        if queue_depth > 5000:
            status = "degraded"
        if metrics.writes_total > 0 and metrics.writes_failed > metrics.writes_total * 0.1:
            status = "unhealthy"

        return {
            "status": status,
            "queue_depth": queue_depth,
            "writes_total": metrics.writes_total,
            "writes_failed": metrics.writes_failed,
            "retries_total": metrics.retries_total,
            "avg_write_latency_ms": (
                metrics.write_latency_ms_sum / metrics.write_latency_ms_count
                if metrics.write_latency_ms_count > 0
                else 0.0
            ),
            "queue_high_watermark": metrics.queue_high_watermark,
            "last_write_ts": metrics.last_write_ts,
        }

    def install(
        self,
        config: Config | None = None,
        *,
        mode: str | None = None,
        sink_type: str | None = None,
        pii: PIIConfig | dict | None = None,
    ) -> None:
        """
        Initialize observability.

        This should be called once at application startup.
        Configuration is loaded from environment variables by default.

        Args:
            config: Optional explicit configuration (overrides env vars).
            mode: Override capture mode (off/metadata_only/evidence_only/full).
            sink_type: Override sink type (auto/sqlite/jsonl/postgres/otlp).
            pii: PII configuration for pre-storage redaction/hashing.
        """
        if self._installed:
            logger.warning("observe.install() called multiple times")
            return

        # Load configuration
        if config is not None:
            self._config = config
        else:
            self._config = load_config()

        # Apply overrides
        if mode is not None:
            from agent_observe.config import CaptureMode as CM

            self._config = Config(
                **{**self._config.__dict__, "mode": CM(mode.lower())}
            )
        if sink_type is not None:
            from agent_observe.config import SinkType as ST

            self._config = Config(
                **{**self._config.__dict__, "sink_type": ST(sink_type.lower())}
            )

        # Apply PII configuration override
        if pii is not None:
            self._config = Config(
                **{**self._config.__dict__, "pii": pii}
            )

        # Initialize components
        try:
            self._sink = create_sink(self._config)
            self._sink.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize sink: {e}")
            from agent_observe.sinks.base import NullSink

            self._sink = NullSink(async_writes=False)

        # Load policy
        policy = load_policy(self._config.policy_file)
        self._policy_engine = PolicyEngine(
            policy=policy,
            fail_on_violation=self._config.fail_on_violation,
        )

        # Initialize replay cache
        self._replay_cache = ReplayCache(
            sink=self._sink,
            mode=self._config.replay_mode,
            capture_mode=self._config.mode,
        )

        # Initialize or update hook registry with current environment
        if self._hooks is None:
            self._hooks = HookRegistry(current_env=self._config.env.value)
        else:
            # Update existing registry with environment
            self._hooks._current_env = self._config.env.value

        self._installed = True

        # Register cleanup
        atexit.register(self._cleanup)

        logger.info(
            f"agent-observe installed: mode={self._config.mode.value}, "
            f"sink={self._config.resolve_sink_type().value}, "
            f"env={self._config.env.value}"
        )

    def _cleanup(self) -> None:
        """Cleanup on shutdown and reset state for re-installation."""
        if self._sink is not None:
            try:
                self._sink.flush()
                self._sink.close()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

        # Reset state so install() can be called again (useful for testing)
        self._installed = False
        self._config = None
        self._sink = None
        self._policy_engine = None
        self._replay_cache = None

    @contextmanager
    def run(
        self,
        name: str,
        task: dict[str, Any] | None = None,
        agent_version: str | None = None,
        *,
        # v0.1.7: Attribution and context
        user_id: str | None = None,
        session_id: str | None = None,
        prompt_version: str | None = None,
        model_config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        # Existing overrides
        mode: str | CaptureMode | None = None,
        policy_file: Path | str | None = None,
        fail_on_violation: bool | None = None,
        latency_budget_ms: int | None = None,
    ) -> Iterator[RunContext]:
        """
        Create a run context for an agent execution.

        Args:
            name: Name of the run (e.g., "order-processor").
            task: Optional task metadata.
            agent_version: Override agent version (default from config).
            user_id: User/account ID for attribution.
            session_id: Session/conversation ID for linking related runs.
            prompt_version: Explicit prompt version (e.g., "v2.3").
            model_config: Model configuration (model name, temperature, etc.).
            experiment_id: A/B test or experiment cohort ID.
            metadata: Custom metadata dictionary.
            mode: Override capture mode for this run (e.g., "full" for debugging).
            policy_file: Override policy file for this run.
            fail_on_violation: Override fail_on_violation for this run.
            latency_budget_ms: Override latency budget for this run.

        Yields:
            RunContext for the run.

        Usage:
            with observe.run("my-agent") as run:
                # agent code
                pass

            # With full context:
            with observe.run(
                "support-agent",
                user_id="jane",
                session_id="conv_123",
                prompt_version="v2.3",
            ) as run:
                run.set_input(user_message)
                # agent code
                run.set_output(response)
        """
        if not self.is_enabled:
            # Return a dummy context if not enabled
            dummy = RunContext(
                run_id=generate_uuid(),
                trace_id=generate_trace_id(),
                name=name,
                ts_start=now_ms(),
            )
            yield dummy
            return

        # Create run context with v0.1.7 fields
        run_ctx = RunContext(
            run_id=generate_uuid(),
            trace_id=generate_trace_id(),
            name=name,
            ts_start=now_ms(),
            task=task,
            agent_version=agent_version or self.config.agent_version,
            project=self.config.project,
            env=self.config.env.value,
            # v0.1.7: Attribution and context
            user_id=user_id,
            session_id=session_id,
            prompt_version=prompt_version,
            model_config=model_config,
            experiment_id=experiment_id,
            metadata=metadata or {},
            _observe=self,
        )

        # Apply per-run overrides
        if mode is not None:
            if isinstance(mode, str):
                # Convert string to CaptureMode enum
                try:
                    run_ctx._mode_override = CaptureMode(mode)
                except ValueError:
                    logger.warning(f"Invalid mode '{mode}', using global config")
            else:
                run_ctx._mode_override = mode

        if policy_file is not None:
            # Load a per-run policy
            policy_path = Path(policy_file) if isinstance(policy_file, str) else policy_file
            per_run_policy = load_policy(policy_path)
            run_ctx._policy_engine_override = PolicyEngine(per_run_policy)

        if fail_on_violation is not None:
            run_ctx._fail_on_violation_override = fail_on_violation

        if latency_budget_ms is not None:
            run_ctx._latency_budget_ms_override = latency_budget_ms

        # Set as current run
        token = set_current_run(run_ctx)

        # Call on_run_start hooks
        if self._hooks is not None:
            start_ctx = RunStartContext(
                run=run_ctx,
                observe=self,
                timestamp_ms=now_ms(),
            )
            self._hooks.run_lifecycle_hooks("on_run_start", start_ctx)

        run_error: BaseException | None = None
        try:
            yield run_ctx
            run_ctx.end(RunStatus.OK)
        except BaseException as e:
            run_error = e
            # Catch BaseException to handle KeyboardInterrupt, SystemExit too
            run_ctx.end(RunStatus.ERROR)
            # Emit error event with sanitized message (only for Exception, not SystemExit)
            if isinstance(e, Exception):
                self._emit_event_internal(
                    run_ctx,
                    "run.error",
                    {
                        "error": sanitize_error_message(e),
                        "error_type": type(e).__name__,
                    },
                )
            raise
        finally:
            # v0.1.7: Auto-infer input/output and prompt_hash if not explicitly set
            run_ctx._infer_input_output()
            run_ctx._infer_prompt_hash()

            # Compute metrics and eval (use per-run latency budget if set)
            eval_result = evaluate_run(run_ctx, run_ctx.get_latency_budget_ms())

            # Call on_run_end hooks
            if self._hooks is not None:
                end_ctx = RunEndContext(
                    run=run_ctx,
                    observe=self,
                    status="error" if run_error else "ok",
                    error=run_error if isinstance(run_error, Exception) else None,
                    duration_ms=run_ctx.duration_ms or 0,
                    timestamp_ms=now_ms(),
                    tool_calls=run_ctx.tool_calls,
                    model_calls=run_ctx.model_calls,
                    policy_violations=run_ctx.policy_violations,
                    spans=run_ctx.spans,
                )
                self._hooks.run_lifecycle_hooks("on_run_end", end_ctx)

            # Build run data for storage
            run_data = run_ctx.to_dict()
            run_data["risk_score"] = eval_result.risk_score
            run_data["eval_tags"] = eval_result.eval_tags
            run_data["latency_ms"] = run_ctx.duration_ms
            run_data["capture_mode"] = run_ctx.get_capture_mode().value

            # Write run to sink first (so FK constraints are satisfied)
            self.sink.write_run(run_data)

            # Write spans (including any not yet flushed during streaming)
            for span in run_ctx.spans:
                self.sink.write_span(span)

            # Write events
            for event in run_ctx.events:
                self.sink.write_event(event)

            # Emit eval event
            self._emit_event_internal(
                run_ctx,
                "eval",
                eval_result.to_dict(),
            )

            # Reset context
            if token is not None:
                reset_current_run(token)

    @asynccontextmanager
    async def arun(
        self,
        name: str,
        task: dict[str, Any] | None = None,
        agent_version: str | None = None,
        *,
        # v0.1.7: Attribution and context
        user_id: str | None = None,
        session_id: str | None = None,
        prompt_version: str | None = None,
        model_config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        # Existing overrides
        mode: str | CaptureMode | None = None,
        policy_file: Path | str | None = None,
        fail_on_violation: bool | None = None,
        latency_budget_ms: int | None = None,
    ) -> AsyncIterator[RunContext]:
        """
        Create an async run context for an agent execution.

        Args:
            name: Name of the run (e.g., "order-processor").
            task: Optional task metadata.
            agent_version: Override agent version (default from config).
            user_id: User/account ID for attribution.
            session_id: Session/conversation ID for linking related runs.
            prompt_version: Explicit prompt version (e.g., "v2.3").
            model_config: Model configuration (model name, temperature, etc.).
            experiment_id: A/B test or experiment cohort ID.
            metadata: Custom metadata dictionary.
            mode: Override capture mode for this run (e.g., "full" for debugging).
            policy_file: Override policy file for this run.
            fail_on_violation: Override fail_on_violation for this run.
            latency_budget_ms: Override latency budget for this run.

        Yields:
            RunContext for the run.

        Usage:
            async with observe.arun("my-agent") as run:
                # async agent code
                await some_async_tool()
        """
        if not self.is_enabled:
            # Return a dummy context if not enabled
            dummy = RunContext(
                run_id=generate_uuid(),
                trace_id=generate_trace_id(),
                name=name,
                ts_start=now_ms(),
            )
            yield dummy
            return

        # Create run context with v0.1.7 fields
        run_ctx = RunContext(
            run_id=generate_uuid(),
            trace_id=generate_trace_id(),
            name=name,
            ts_start=now_ms(),
            task=task,
            agent_version=agent_version or self.config.agent_version,
            project=self.config.project,
            env=self.config.env.value,
            # v0.1.7: Attribution and context
            user_id=user_id,
            session_id=session_id,
            prompt_version=prompt_version,
            model_config=model_config,
            experiment_id=experiment_id,
            metadata=metadata or {},
            _observe=self,
        )

        # Apply per-run overrides
        if mode is not None:
            if isinstance(mode, str):
                try:
                    run_ctx._mode_override = CaptureMode(mode)
                except ValueError:
                    logger.warning(f"Invalid mode '{mode}', using global config")
            else:
                run_ctx._mode_override = mode

        if policy_file is not None:
            policy_path = Path(policy_file) if isinstance(policy_file, str) else policy_file
            per_run_policy = load_policy(policy_path)
            run_ctx._policy_engine_override = PolicyEngine(per_run_policy)

        if fail_on_violation is not None:
            run_ctx._fail_on_violation_override = fail_on_violation

        if latency_budget_ms is not None:
            run_ctx._latency_budget_ms_override = latency_budget_ms

        # Set as current run
        token = set_current_run(run_ctx)

        # Call on_run_start hooks (async)
        if self._hooks is not None:
            start_ctx = RunStartContext(
                run=run_ctx,
                observe=self,
                timestamp_ms=now_ms(),
            )
            await self._hooks.run_lifecycle_hooks_async("on_run_start", start_ctx)

        run_error: BaseException | None = None
        try:
            yield run_ctx
            run_ctx.end(RunStatus.OK)
        except BaseException as e:
            run_error = e
            # Catch BaseException to handle KeyboardInterrupt, SystemExit too
            run_ctx.end(RunStatus.ERROR)
            # Emit error event with sanitized message (only for Exception, not SystemExit)
            if isinstance(e, Exception):
                self._emit_event_internal(
                    run_ctx,
                    "run.error",
                    {
                        "error": sanitize_error_message(e),
                        "error_type": type(e).__name__,
                    },
                )
            raise
        finally:
            # v0.1.7: Auto-infer input/output and prompt_hash if not explicitly set
            run_ctx._infer_input_output()
            run_ctx._infer_prompt_hash()

            # Compute metrics and eval (use per-run latency budget if set)
            eval_result = evaluate_run(run_ctx, run_ctx.get_latency_budget_ms())

            # Call on_run_end hooks (async)
            if self._hooks is not None:
                end_ctx = RunEndContext(
                    run=run_ctx,
                    observe=self,
                    status="error" if run_error else "ok",
                    error=run_error if isinstance(run_error, Exception) else None,
                    duration_ms=run_ctx.duration_ms or 0,
                    timestamp_ms=now_ms(),
                    tool_calls=run_ctx.tool_calls,
                    model_calls=run_ctx.model_calls,
                    policy_violations=run_ctx.policy_violations,
                    spans=run_ctx.spans,
                )
                await self._hooks.run_lifecycle_hooks_async("on_run_end", end_ctx)

            # Build run data for storage
            run_data = run_ctx.to_dict()
            run_data["risk_score"] = eval_result.risk_score
            run_data["eval_tags"] = eval_result.eval_tags
            run_data["latency_ms"] = run_ctx.duration_ms
            run_data["capture_mode"] = run_ctx.get_capture_mode().value

            # Write run to sink first (so FK constraints are satisfied)
            self.sink.write_run(run_data)

            # Write spans
            for span in run_ctx.spans:
                self.sink.write_span(span)

            # Write events
            for event in run_ctx.events:
                self.sink.write_event(event)

            # Emit eval event
            self._emit_event_internal(
                run_ctx,
                "eval",
                eval_result.to_dict(),
            )

            # Reset context
            if token is not None:
                reset_current_run(token)

    def run_fn(
        self,
        name: str,
        task: dict[str, Any] | None,
        fn: Callable[[], T],
        agent_version: str | None = None,
    ) -> T:
        """
        Execute a function within a run context.

        Convenience wrapper for observe.run().

        Args:
            name: Name of the run.
            task: Optional task metadata.
            fn: Function to execute.
            agent_version: Override agent version.

        Returns:
            Result of fn().
        """
        with self.run(name, task, agent_version):
            return fn()

    def emit_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Emit a custom event.

        Args:
            event_type: Type of event (e.g., "user.feedback").
            payload: Event payload (max 16KB).
        """
        run = get_current_run()
        if run is None:
            logger.warning("emit_event called outside of run context")
            return

        self._emit_event_internal(run, event_type, payload)

    def _emit_event_internal(
        self,
        run: RunContext,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Internal method to emit events."""
        import json

        # Cap payload size
        payload_bytes = json.dumps(payload, default=str).encode("utf-8")
        if len(payload_bytes) > self.config.max_event_payload_bytes:
            logger.warning(
                f"Event payload exceeds limit ({len(payload_bytes)} > "
                f"{self.config.max_event_payload_bytes}), truncating"
            )
            payload = {"_truncated": True, "size": len(payload_bytes)}

        event = {
            "event_id": generate_uuid(),
            "run_id": run.run_id,
            "ts": now_ms(),
            "type": event_type,
            "payload": payload,
        }

        run.add_event(event)

    def emit_artifact(
        self,
        artifact_type: str,
        content: Any,
        provenance: list[str] | None = None,
    ) -> None:
        """
        Emit an artifact.

        In metadata_only mode, only the hash and size are stored.

        Args:
            artifact_type: Type of artifact (e.g., "report", "analysis").
            content: Artifact content.
            provenance: Optional list of source identifiers.
        """
        run = get_current_run()
        if run is None:
            logger.warning("emit_artifact called outside of run context")
            return

        content_hash, content_size = hash_content(content)

        # Build artifact event
        artifact_data: dict[str, Any] = {
            "artifact_type": artifact_type,
            "content_hash": content_hash,
            "content_size": content_size,
            "provenance": provenance,
        }

        # In evidence_only/full mode, include actual content (with cap)
        if self.config.mode in (CaptureMode.EVIDENCE_ONLY, CaptureMode.FULL):
            import json

            if isinstance(content, (str, bytes)):
                content_to_store = content
            else:
                content_to_store = json.dumps(content, default=str)

            # Cap content size
            if isinstance(content_to_store, str):
                content_bytes = content_to_store.encode("utf-8")
            else:
                content_bytes = content_to_store

            if len(content_bytes) <= self.config.max_artifact_bytes:
                artifact_data["content"] = (
                    content_to_store
                    if isinstance(content_to_store, str)
                    else content_to_store.decode("utf-8", errors="replace")
                )
            else:
                artifact_data["_content_truncated"] = True

        self._emit_event_internal(run, "artifact", artifact_data)


# Global singleton
observe = Observe()
