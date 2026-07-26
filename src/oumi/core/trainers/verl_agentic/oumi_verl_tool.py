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

"""verl adapter backed by a rollout-scoped Oumi tool router."""

from __future__ import annotations

import json
from functools import cache
from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool  # pyright: ignore[reportMissingImports]
from verl.tools.schemas import (  # pyright: ignore[reportMissingImports]
    OpenAIFunctionToolSchema,
    ToolResponse,
)

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.synthesis.tool_router import ToolRouter
from oumi.core.trainers.verl_agentic.env_provider import get_or_build_router
from oumi.utils.logging import logger
from oumi.utils.packaging import is_verl_v0_7_or_later


@cache
def _load_env_config(path: str) -> EnvironmentConfig:
    config = EnvironmentConfig.from_yaml(path)
    config.finalize_and_validate()
    return config


@cache
def _parent_router(path: str) -> ToolRouter:
    """The process-wide template each rollout clones; its envs are built once."""
    return ToolRouter.from_environment_config(_load_env_config(path))


class OumiVerlTool(BaseTool):
    """A verl tool backed by the router shared within a rollout."""

    def __init__(
        self,
        config: dict,
        tool_schema: OpenAIFunctionToolSchema,
    ) -> None:
        """Loads and validates the configured `EnvironmentConfig`."""
        if not is_verl_v0_7_or_later():
            raise RuntimeError("OumiVerlTool requires verl 0.7.0 or later.")
        super().__init__(config, tool_schema)
        self._tool_id = tool_schema.function.name
        self._env_config_path = config["oumi_env_config"]
        # Validate at construction so a bad config fails before any rollout starts.
        _load_env_config(self._env_config_path)

    async def create(
        self, instance_id: str | None = None, **kwargs: Any
    ) -> tuple[str, ToolResponse]:
        """Mints an instance id; the router is created lazily in `execute`."""
        return instance_id or uuid4().hex, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        agent_data: Any = None,
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        """Routes one tool call through this rollout's shared Oumi router."""
        router = get_or_build_router(agent_data, _parent_router(self._env_config_path))
        # verl emits `None` for unset optional args; drop them so the executor
        # sees an absent key rather than an explicit null.
        args = {k: v for k, v in (parameters or {}).items() if v is not None}
        try:
            result = router.route_batch([(self._tool_id, args)])[0]
        except Exception as e:
            # A bad tool call is an observation the policy can recover from, not a
            # reason to kill the trajectory.
            logger.warning(f"Tool '{self._tool_id}' call failed: {e}")
            return ToolResponse(text=f"Tool error: {e}"), 0.0, {}
        out = result.output
        text = out if isinstance(out, str) else json.dumps(out)
        return ToolResponse(text=text), 0.0, {}

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        """No-op because router teardown occurs at the rollout boundary."""
        return None
