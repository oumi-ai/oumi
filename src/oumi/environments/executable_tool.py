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

"""Shared base class for tools that resolve a dotted-path Python executor."""

from __future__ import annotations

from dataclasses import dataclass

from oumi.core.configs.params.tool_params import ToolParams


@dataclass
class ExecutableTool(ToolParams):
    """`ToolParams` variant for envs that take user-supplied dotted-path executors.

    Subclasses (``DatabaseExecutableTool``, future ``HTTPExecutableTool``, etc.)
    inherit this and may add transport-specific per-tool overrides.
    """

    executor: str = ""

    def __post_init__(self) -> None:
        """Validate inherited fields and enforce non-empty executor."""
        super().__post_init__()
        if not self.executor:
            raise ValueError(
                f"{type(self).__name__} '{self.id}' must declare a non-empty "
                f"executor (dotted import path)."
            )
