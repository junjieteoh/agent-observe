"""
Viewer module for agent-observe.

Provides a FastAPI-based web UI for browsing runs, spans, and events.
"""

from agent_observe.viewer.app import create_app

__all__ = ["create_app"]
