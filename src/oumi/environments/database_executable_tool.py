# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Database-specific ExecutableTool subclass."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from oumi.environments.executable_tool import ExecutableTool


@dataclass
class DatabaseExecutableTool(ExecutableTool):
    """`ExecutableTool` for ``DatabaseExecutableEnvironment``.

    Adds an optional per-tool ``statement_timeout_ms`` that may **only** tighten
    the env-level timeout (validated by the env at construction time).
    """

    statement_timeout_ms: int | None = None

    @classmethod
    def create(cls, raw: Any) -> DatabaseExecutableTool:
        """Create a DatabaseExecutableTool from raw config data."""
        if isinstance(raw, DatabaseExecutableTool):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Tool definitions must be tool objects or mappings, got {type(raw)}"
            )
        return cls(
            id=raw["id"],
            name=raw["name"],
            description=raw["description"],
            parameters=dict(raw.get("parameters") or {"type": "object"}),
            output_schema=(
                dict(raw["output_schema"])
                if raw.get("output_schema") is not None
                else None
            ),
            read_only=raw.get("read_only", True),
            executor=raw["executor"],
            statement_timeout_ms=raw.get("statement_timeout_ms"),
        )
