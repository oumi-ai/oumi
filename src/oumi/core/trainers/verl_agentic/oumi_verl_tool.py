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

"""verl `BaseTool` adapter that routes tool calls into one Oumi env per rollout."""

from __future__ import annotations

import json
from functools import cache
from typing import Any
from uuid import uuid4

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.trainers.verl_agentic.env_provider import get_or_build_router

# verl isn't importable off-cluster; fall back to stubs so this module still imports.
try:
    from verl.tools.base_tool import BaseTool  # pyright: ignore[reportMissingImports]
    from verl.tools.schemas import (  # pyright: ignore[reportMissingImports]
        OpenAIFunctionToolSchema,
        ToolResponse,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised only where verl is absent
    BaseTool = object  # type: ignore[assignment,misc]
    OpenAIFunctionToolSchema = Any  # type: ignore[assignment,misc]
    ToolResponse = None  # type: ignore[assignment,misc]


@cache
def _load_env_config(path: str) -> EnvironmentConfig:
    return EnvironmentConfig.from_yaml(path)


class OumiVerlTool(BaseTool):  # pyright: ignore[reportGeneralTypeIssues]
    """One instance per Oumi tool; all instances of a rollout share one env."""

    def __init__(
        self,
        config: dict,
        tool_schema: OpenAIFunctionToolSchema,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Loads this tool's shared `EnvironmentConfig` from `config`."""
        super().__init__(config, tool_schema)  # pyright: ignore[reportCallIssue]
        self._tool_id = tool_schema.function.name
        self._env_config = _load_env_config(config["oumi_env_config"])

    async def create(self, instance_id=None, **kwargs):
        """Mints an instance id; the env itself is created lazily in `execute`."""
        return instance_id or uuid4().hex, ToolResponse()  # pyright: ignore[reportOptionalCall]

    async def execute(self, instance_id, parameters, agent_data=None, **kwargs):
        """Routes one tool call into this rollout's shared Oumi env."""
        router = get_or_build_router(agent_data, self._env_config)
        args = {k: v for k, v in (parameters or {}).items() if v is not None}
        result = router.route_batch([(self._tool_id, args)])[0]
        out = result.output
        text = out if isinstance(out, str) else json.dumps(out)
        return ToolResponse(text=text), 0.0, {}  # pyright: ignore[reportOptionalCall]

    async def release(self, instance_id, **kwargs):
        """No-op: env teardown is rollout-scoped (see env_provider), not per-call."""
        return None
