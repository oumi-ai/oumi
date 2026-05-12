# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Exceptions raised by the agent harness."""

from __future__ import annotations


class AgentSessionError(Exception):
    """Programmer / wiring error that should never reach the chat loop."""


class UnknownToolError(Exception):
    """The LLM emitted a tool name that is not registered."""


class InvalidToolArgumentsError(Exception):
    """Tool-call arguments failed JSON parse or schema validation."""
