# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Shared base class for tools that resolve a dotted-path Python executor."""

from __future__ import annotations

from dataclasses import dataclass

from oumi.core.configs.params.tool_params import ToolParams


@dataclass
class ExecutableTool(ToolParams):
    """`ToolParams` variant for envs that take user-supplied dotted-path executors."""

    executor: str = ""

    def __post_init__(self) -> None:
        """Validate inherited fields and enforce non-empty executor."""
        super().__post_init__()
        if not self.executor:
            raise ValueError(
                f"{type(self).__name__} '{self.id}' must declare a non-empty "
                f"executor (dotted import path)."
            )
