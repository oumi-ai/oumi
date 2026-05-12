# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Flattens tools across environments and routes wire-name to owning env."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from oumi.agents.exceptions import InvalidToolArgumentsError, UnknownToolError
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.tool_params import ToolArgumentError, ToolParams
from oumi.core.types.tool_call import ToolDefinition


@dataclass
class ToolRouter:
    """Read-only routing layer between the LLM wire format and environments."""

    tool_specs: list[ToolDefinition]
    _tool_id_to_env_id: dict[str, str] = field(default_factory=dict)
    _tool_by_id: dict[str, ToolParams] = field(default_factory=dict)

    @classmethod
    def from_environment_config(cls, env_config: EnvironmentConfig) -> ToolRouter:
        """Build a router from the flattened tool list in ``env_config``."""
        tools = env_config.all_tools
        return cls(
            tool_specs=[t.to_tool_definition() for t in tools],
            _tool_id_to_env_id=dict(env_config.tool_environment_map),
            _tool_by_id={t.id: t for t in tools},
        )

    def route(self, wire_name: str) -> tuple[str, ToolParams]:
        """Resolve a wire tool name to ``(env_id, tool)``."""
        tool = self._tool_by_id.get(wire_name)
        if tool is None:
            raise UnknownToolError(
                f"Tool '{wire_name}' is not registered. Available: "
                f"{sorted(self._tool_by_id)}"
            )
        env_id = self._tool_id_to_env_id[wire_name]
        return env_id, tool

    def parse_and_validate_arguments(
        self, tool: ToolParams, raw_arguments: str
    ) -> dict[str, Any]:
        """Parse the wire JSON string and validate against the tool's schema."""
        raw = raw_arguments or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise InvalidToolArgumentsError(
                f"Tool '{tool.id}' arguments are not valid JSON: {e}"
            ) from e
        if not isinstance(parsed, dict):
            raise InvalidToolArgumentsError(
                f"Tool '{tool.id}' arguments must be a JSON object, got "
                f"{type(parsed).__name__}."
            )
        try:
            tool.validate_arguments(parsed)
        except ToolArgumentError as e:
            raise InvalidToolArgumentsError(
                f"Tool '{tool.id}' arguments failed schema validation: {e}"
            ) from e
        return parsed
