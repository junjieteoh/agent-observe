"""
Sinks for agent-observe telemetry data.

Sinks handle the persistence of runs, spans, and events to various backends.
"""

from agent_observe.sinks.base import Sink, create_sink

__all__ = [
    "Sink",
    "create_sink",
]
