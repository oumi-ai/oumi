# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Database-specific ``ExecutableTool`` subclass."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from oumi.environments.executable_tool import ExecutableTool


@dataclass
class DatabaseExecutableTool(ExecutableTool):
    """``ExecutableTool`` for ``DatabaseExecutableEnvironment``.

    Adds an optional per-tool ``statement_timeout_ms`` that may **only**
    tighten the env-level timeout. Cross-validation against the env's value
    happens in the env's ``from_params`` (implementation phase).
    """

    statement_timeout_ms: int | None = None

    @classmethod
    def create(cls, raw: Any) -> DatabaseExecutableTool:
        """Construct a ``DatabaseExecutableTool`` from raw config data."""
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
