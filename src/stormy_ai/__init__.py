"""Stormy AI — LangGraph weather briefing agent."""

from stormy_ai import config as _config  # noqa: F401 — load .env before graph
from stormy_ai.logging_config import configure_logging

configure_logging()

from stormy_ai.agent import graph  # noqa: E402

__all__ = ["graph"]
