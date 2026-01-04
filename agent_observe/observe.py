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
from typing import Any, Callable, TypeVar

from agent_observe.config import CaptureMode, Config, load_config
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

    # Truncate first
    if len(message) > max_length:
        message = message[:max_length] + "...[truncated]"

    # Redact potential secrets
    for pattern in SECRET_PATTERNS:
        message = re.sub(pattern, "[REDACTED]", message)

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
    def is_installed(self) -> bool:
        """Check if observability is installed."""
        return self._installed

    @property
    def is_enabled(self) -> bool:
        """Check if observability is enabled (not OFF mode)."""
        return self._installed and self._config is not None and self._config.mode != CaptureMode.OFF

    def install(
        self,
        config: Config | None = None,
        *,
        mode: str | None = None,
        sink_type: str | None = None,
    ) -> None:
        """
        Initialize observability.

        This should be called once at application startup.
        Configuration is loaded from environment variables by default.

        Args:
            config: Optional explicit configuration (overrides env vars).
            mode: Override capture mode (off/metadata_only/evidence_only/full).
            sink_type: Override sink type (auto/sqlite/jsonl/postgres/otlp).
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

        self._installed = True

        # Register cleanup
        atexit.register(self._cleanup)

        logger.info(
            f"agent-observe installed: mode={self._config.mode.value}, "
            f"sink={self._config.resolve_sink_type().value}, "
            f"env={self._config.env.value}"
        )

    def _cleanup(self) -> None:
        """Cleanup on shutdown."""
        if self._sink is not None:
            try:
                self._sink.flush()
                self._sink.close()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

    @contextmanager
    def run(
        self,
        name: str,
        task: dict[str, Any] | None = None,
        agent_version: str | None = None,
    ) -> Iterator[RunContext]:
        """
        Create a run context for an agent execution.

        Args:
            name: Name of the run (e.g., "order-processor").
            task: Optional task metadata.
            agent_version: Override agent version (default from config).

        Yields:
            RunContext for the run.

        Usage:
            with observe.run("my-agent") as run:
                # agent code
                pass
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

        # Create run context
        run_ctx = RunContext(
            run_id=generate_uuid(),
            trace_id=generate_trace_id(),
            name=name,
            ts_start=now_ms(),
            task=task,
            agent_version=agent_version or self.config.agent_version,
            project=self.config.project,
            env=self.config.env.value,
            _observe=self,
        )

        # Set as current run
        token = set_current_run(run_ctx)

        try:
            yield run_ctx
            run_ctx.end(RunStatus.OK)
        except Exception as e:
            run_ctx.end(RunStatus.ERROR)
            # Emit error event with sanitized message
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
            # Compute metrics and eval
            eval_result = evaluate_run(run_ctx, self.config.latency_budget_ms)

            # Build run data for storage
            run_data = run_ctx.to_dict()
            run_data["risk_score"] = eval_result.risk_score
            run_data["eval_tags"] = eval_result.eval_tags
            run_data["latency_ms"] = run_ctx.duration_ms

            # Write run to sink
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

    @asynccontextmanager
    async def arun(
        self,
        name: str,
        task: dict[str, Any] | None = None,
        agent_version: str | None = None,
    ) -> AsyncIterator[RunContext]:
        """
        Create an async run context for an agent execution.

        Args:
            name: Name of the run (e.g., "order-processor").
            task: Optional task metadata.
            agent_version: Override agent version (default from config).

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

        # Create run context
        run_ctx = RunContext(
            run_id=generate_uuid(),
            trace_id=generate_trace_id(),
            name=name,
            ts_start=now_ms(),
            task=task,
            agent_version=agent_version or self.config.agent_version,
            project=self.config.project,
            env=self.config.env.value,
            _observe=self,
        )

        # Set as current run
        token = set_current_run(run_ctx)

        try:
            yield run_ctx
            run_ctx.end(RunStatus.OK)
        except Exception as e:
            run_ctx.end(RunStatus.ERROR)
            # Emit error event with sanitized message
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
            # Compute metrics and eval
            eval_result = evaluate_run(run_ctx, self.config.latency_budget_ms)

            # Build run data for storage
            run_data = run_ctx.to_dict()
            run_data["risk_score"] = eval_result.risk_score
            run_data["eval_tags"] = eval_result.eval_tags
            run_data["latency_ms"] = run_ctx.duration_ms

            # Write run to sink
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
