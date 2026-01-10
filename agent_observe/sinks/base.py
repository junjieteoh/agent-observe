"""
Base sink interface and factory for agent-observe.

Sinks are non-blocking by default, using a background thread to flush data.
This ensures that observability never blocks agent execution.
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_observe.config import Config
    from agent_observe.context import RunContext, SpanContext
    from agent_observe.pii import PIIHandler

logger = logging.getLogger(__name__)


class WriteType(Enum):
    """Types of writes to the sink."""

    RUN = "run"
    SPAN = "span"
    EVENT = "event"
    REPLAY_CACHE = "replay_cache"


@dataclass
class WriteRequest:
    """A request to write data to the sink."""

    write_type: WriteType
    data: dict[str, Any]


@dataclass
class SinkMetrics:
    """Internal metrics for the sink.

    Tracks write operations, latency, and queue status for observability
    of the observability system itself.
    """

    writes_total: int = 0
    writes_failed: int = 0
    write_latency_ms_sum: float = 0.0
    write_latency_ms_count: int = 0
    queue_high_watermark: int = 0
    last_write_ts: float = 0.0
    retries_total: int = 0


class Sink(ABC):
    """
    Abstract base class for telemetry sinks.

    Sinks handle the persistence of runs, spans, and events.
    All implementations should be thread-safe and non-blocking.
    """

    def __init__(
        self,
        async_writes: bool = True,
        queue_size: int = 10000,
        flush_interval: float = 1.0,
        max_retries: int = 3,
        pii_handler: PIIHandler | None = None,
    ):
        """
        Initialize the sink.

        Args:
            async_writes: If True, writes are queued and flushed in background.
            queue_size: Maximum queue size for async writes.
            flush_interval: Seconds between background flushes.
            max_retries: Maximum retry attempts for failed writes.
            pii_handler: Optional PII handler for pre-storage redaction/hashing.
        """
        self.async_writes = async_writes
        self._queue: queue.Queue[WriteRequest | None] = queue.Queue(maxsize=queue_size)
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._shutdown = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._initialized = False
        self._metrics = SinkMetrics()
        self._pii_handler = pii_handler

    @property
    def pii_handler(self) -> PIIHandler | None:
        """Get the PII handler if configured."""
        return self._pii_handler

    @pii_handler.setter
    def pii_handler(self, handler: PIIHandler | None) -> None:
        """Set the PII handler."""
        self._pii_handler = handler

    def _process_pii(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process data through the PII handler if configured.

        Args:
            data: Data dictionary to process.

        Returns:
            Processed data with PII handled (redacted/hashed/etc).
        """
        if self._pii_handler is None:
            return data
        return self._pii_handler.process(data)

    @property
    def queue_depth(self) -> int:
        """Current number of items waiting to be written."""
        return self._queue.qsize()

    @property
    def metrics(self) -> SinkMetrics:
        """Get sink metrics for monitoring."""
        return self._metrics

    def initialize(self) -> None:
        """
        Initialize the sink (create tables, connections, etc.).

        Called once during observe.install().
        """
        if self._initialized:
            return

        self._do_initialize()
        self._initialized = True

        if self.async_writes:
            self._start_writer_thread()
            atexit.register(self.close)

    @abstractmethod
    def _do_initialize(self) -> None:
        """Subclass-specific initialization."""
        pass

    def _start_writer_thread(self) -> None:
        """Start the background writer thread."""
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="agent-observe-sink-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        """Background thread that processes the write queue."""
        batch: list[WriteRequest] = []
        batch_size = 100
        consecutive_errors = 0
        max_consecutive_errors = 10

        while not self._shutdown.is_set():
            try:
                # Wait for items with timeout
                try:
                    item = self._queue.get(timeout=self._flush_interval)
                except queue.Empty:
                    # Timeout - flush any pending batch and mark as done
                    if batch:
                        self._safe_flush_batch(batch)
                        for _ in range(len(batch)):
                            self._queue.task_done()
                        batch = []
                    continue

                if item is None:  # Shutdown signal
                    self._queue.task_done()
                    break

                batch.append(item)

                # Drain queue up to batch_size
                while len(batch) < batch_size:
                    try:
                        item = self._queue.get_nowait()
                        if item is None:  # Shutdown signal
                            self._queue.task_done()
                            # Flush what we have so far
                            if batch:
                                self._safe_flush_batch(batch)
                                for _ in range(len(batch)):
                                    self._queue.task_done()
                                batch = []
                            return  # Exit the loop entirely
                        batch.append(item)
                    except queue.Empty:
                        break

                # Flush batch and mark all items as done AFTER successful write
                if batch:
                    success = self._safe_flush_batch(batch)
                    # Mark items as done regardless of success to prevent deadlock
                    for _ in range(len(batch)):
                        self._queue.task_done()

                    if success:
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error(
                                f"Sink writer exceeded {max_consecutive_errors} "
                                "consecutive errors, disabling async writes"
                            )
                            break
                    batch = []

            except Exception as e:
                logger.error(f"Unexpected error in sink writer thread: {e}")
                # Mark any items in batch as done to prevent deadlock
                for _ in range(len(batch)):
                    try:
                        self._queue.task_done()
                    except ValueError:
                        break  # task_done called too many times
                consecutive_errors += 1
                batch = []

                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Sink writer thread shutting down due to repeated errors")
                    break

        # Final flush of any remaining items
        if batch:
            self._safe_flush_batch(batch)
            for _ in range(len(batch)):
                self._queue.task_done()

    def _safe_flush_batch(self, batch: list[WriteRequest]) -> bool:
        """Flush a batch with retry and metrics tracking. Returns True on success."""
        start_time = time.time()

        for attempt in range(self._max_retries):
            try:
                self._flush_batch(batch)

                # Track success metrics
                elapsed_ms = (time.time() - start_time) * 1000
                self._metrics.writes_total += len(batch)
                self._metrics.write_latency_ms_sum += elapsed_ms
                self._metrics.write_latency_ms_count += 1
                self._metrics.last_write_ts = time.time()

                # Track queue high watermark
                current_depth = self._queue.qsize()
                if current_depth > self._metrics.queue_high_watermark:
                    self._metrics.queue_high_watermark = current_depth

                return True

            except Exception as e:
                if attempt < self._max_retries - 1:
                    wait_time = 2**attempt  # 1s, 2s, 4s exponential backoff
                    logger.warning(
                        f"Write failed (attempt {attempt + 1}/{self._max_retries}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    self._metrics.retries_total += 1
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Write failed after {self._max_retries} attempts "
                        f"({len(batch)} items): {e}"
                    )
                    self._metrics.writes_failed += len(batch)
                    return False

        return False

    def _flush_batch(self, batch: list[WriteRequest]) -> None:
        """Flush a batch of write requests."""
        runs = []
        spans = []
        events = []
        replay_entries = []

        for req in batch:
            if req.write_type == WriteType.RUN:
                runs.append(req.data)
            elif req.write_type == WriteType.SPAN:
                spans.append(req.data)
            elif req.write_type == WriteType.EVENT:
                events.append(req.data)
            elif req.write_type == WriteType.REPLAY_CACHE:
                replay_entries.append(req.data)

        if runs:
            self._do_write_runs(runs)
        if spans:
            self._do_write_spans(spans)
        if events:
            self._do_write_events(events)
        if replay_entries:
            self._do_write_replay_cache(replay_entries)

    def _enqueue(self, write_type: WriteType, data: dict[str, Any]) -> bool:
        """
        Enqueue a write request.

        Returns True if enqueued, False if queue is full (data dropped).
        """
        try:
            self._queue.put_nowait(WriteRequest(write_type, data))
            return True
        except queue.Full:
            logger.warning(f"Sink queue full, dropping {write_type.value}")
            return False

    def write_run(self, run: RunContext | dict[str, Any]) -> None:
        """
        Write a run to the sink.

        PII processing is applied before storage if a handler is configured.

        Args:
            run: RunContext or dict representation.
        """
        data = run.to_dict() if hasattr(run, "to_dict") else run
        data = self._process_pii(data)

        if self.async_writes:
            self._enqueue(WriteType.RUN, data)
        else:
            self._do_write_runs([data])

    def write_span(self, span: SpanContext | dict[str, Any]) -> None:
        """
        Write a span to the sink.

        PII processing is applied before storage if a handler is configured.

        Args:
            span: SpanContext or dict representation.
        """
        data = span.to_dict() if hasattr(span, "to_dict") else span
        data = self._process_pii(data)

        if self.async_writes:
            self._enqueue(WriteType.SPAN, data)
        else:
            self._do_write_spans([data])

    def write_event(self, event: dict[str, Any]) -> None:
        """
        Write an event to the sink.

        PII processing is applied before storage if a handler is configured.

        Args:
            event: Event dictionary.
        """
        event = self._process_pii(event)

        if self.async_writes:
            self._enqueue(WriteType.EVENT, event)
        else:
            self._do_write_events([event])

    def write_replay_cache(self, entry: dict[str, Any]) -> None:
        """
        Write a replay cache entry.

        Args:
            entry: Replay cache entry dictionary.
        """
        if self.async_writes:
            self._enqueue(WriteType.REPLAY_CACHE, entry)
        else:
            self._do_write_replay_cache([entry])

    @abstractmethod
    def _do_write_runs(self, runs: list[dict[str, Any]]) -> None:
        """Write runs to storage (batch)."""
        pass

    @abstractmethod
    def _do_write_spans(self, spans: list[dict[str, Any]]) -> None:
        """Write spans to storage (batch)."""
        pass

    @abstractmethod
    def _do_write_events(self, events: list[dict[str, Any]]) -> None:
        """Write events to storage (batch)."""
        pass

    def _do_write_replay_cache(  # noqa: B027
        self, entries: list[dict[str, Any]]
    ) -> None:
        """Write replay cache entries (optional, default no-op)."""
        del entries  # Unused in base implementation

    def flush(self) -> None:
        """Flush any pending writes."""
        if not self.async_writes:
            return

        # Wait for queue to drain
        self._queue.join()

    def close(self) -> None:
        """Close the sink and release resources."""
        if self._writer_thread and self._writer_thread.is_alive():
            self._shutdown.set()
            self._queue.put(None)  # Signal shutdown
            self._writer_thread.join(timeout=5.0)

        self._do_close()

    def _do_close(self) -> None:  # noqa: B027
        """Subclass-specific cleanup (optional override)."""

    # Query methods (for viewer)

    def get_runs(
        self,
        name: str | None = None,  # noqa: ARG002
        status: str | None = None,  # noqa: ARG002
        min_risk: int | None = None,  # noqa: ARG002
        tag: str | None = None,  # noqa: ARG002
        limit: int = 100,  # noqa: ARG002
        offset: int = 0,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """
        Query runs with optional filters.

        Args:
            name: Filter by run name (partial match).
            status: Filter by status.
            min_risk: Filter by minimum risk score.
            tag: Filter by eval tag.
            limit: Maximum results.
            offset: Result offset for pagination.

        Returns:
            List of run dictionaries.
        """
        return []

    def get_run(self, run_id: str) -> dict[str, Any] | None:  # noqa: ARG002
        """Get a single run by ID."""
        return None

    def get_spans(self, run_id: str) -> list[dict[str, Any]]:  # noqa: ARG002
        """Get all spans for a run."""
        return []

    def get_events(self, run_id: str) -> list[dict[str, Any]]:  # noqa: ARG002
        """Get all events for a run."""
        return []

    def get_replay_cache_entry(self, key: str) -> dict[str, Any] | None:  # noqa: ARG002
        """Get a replay cache entry by key."""
        return None


class NullSink(Sink):
    """A sink that discards all data (used when mode=off)."""

    def _do_initialize(self) -> None:
        pass

    def _do_write_runs(self, runs: list[dict[str, Any]]) -> None:
        pass

    def _do_write_spans(self, spans: list[dict[str, Any]]) -> None:
        pass

    def _do_write_events(self, events: list[dict[str, Any]]) -> None:
        pass


def create_sink(config: Config) -> Sink:
    """
    Create the appropriate sink based on configuration.

    Args:
        config: Application configuration.

    Returns:
        Configured Sink instance.
    """
    from agent_observe.config import CaptureMode, SinkType
    from agent_observe.pii import create_pii_handler

    # Create PII handler if configured
    pii_handler = create_pii_handler(config.pii)
    if pii_handler:
        logger.info(f"PII handling enabled with action: {pii_handler.config.action}")

    # If observability is off, use null sink
    if config.mode == CaptureMode.OFF:
        logger.info("Observability mode is OFF, using NullSink")
        return NullSink(async_writes=False)

    sink_type = config.resolve_sink_type()
    logger.info(f"Creating {sink_type.value} sink")

    sink: Sink | None = None

    try:
        if sink_type == SinkType.SQLITE:
            from agent_observe.sinks.sqlite_sink import SQLiteSink

            sink = SQLiteSink(path=config.sqlite_path)

        elif sink_type == SinkType.JSONL:
            from agent_observe.sinks.jsonl_sink import JSONLSink

            sink = JSONLSink(directory=config.jsonl_dir)

        elif sink_type == SinkType.POSTGRES:
            from agent_observe.sinks.postgres_sink import PostgresSink

            if not config.database_url:
                raise ValueError("DATABASE_URL required for Postgres sink")
            sink = PostgresSink(
                database_url=config.database_url,
                schema=config.pg_schema,
            )

        elif sink_type == SinkType.OTLP:
            from agent_observe.sinks.otel_sink import OTLPSink

            if not config.otlp_endpoint:
                raise ValueError("OTEL_EXPORTER_OTLP_ENDPOINT required for OTLP sink")
            sink = OTLPSink(endpoint=config.otlp_endpoint)

        else:
            # Fallback to JSONL
            logger.warning(f"Unknown sink type {sink_type}, falling back to JSONL")
            from agent_observe.sinks.jsonl_sink import JSONLSink

            sink = JSONLSink(directory=config.jsonl_dir)

        # Set PII handler on the sink
        if sink and pii_handler:
            sink.pii_handler = pii_handler

        return sink

    except ImportError as e:
        logger.warning(f"Failed to import sink {sink_type.value}: {e}, falling back to JSONL")
        from agent_observe.sinks.jsonl_sink import JSONLSink

        fallback = JSONLSink(directory=config.jsonl_dir)
        if pii_handler:
            fallback.pii_handler = pii_handler
        return fallback

    except Exception as e:
        logger.error(f"Failed to create sink {sink_type.value}: {e}, falling back to NullSink")
        return NullSink(async_writes=False)
