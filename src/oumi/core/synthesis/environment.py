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

"""Stateful environment and registry for agentic tool synthesis."""

from collections.abc import Callable
from typing import Any

from oumi.core.configs.params.tool_params import (
    ToolAttribute,
    ToolEnvironmentAttribute,
)
from oumi.core.types.conversation import Conversation, Message

_MAX_RESULT_RETRIES = 2


class GeneratedToolEnvironment:
    """Stateful environment for tool synthesis.

    Maintains a JSON state document that evolves as tools read/write to it.
    Builds prompts and applies responses — does not call inference itself.
    """

    def __init__(self, config: ToolEnvironmentAttribute):
        """Initialize the environment with config."""
        raise NotImplementedError

    @property
    def state(self) -> dict[str, Any]:
        """Current state of the environment."""
        raise NotImplementedError

    def summarize_for_planner(self) -> dict[str, Any]:
        """Return a compact state view for planner grounding."""
        raise NotImplementedError

    def set_state(self, state: dict[str, Any], validate: bool = True) -> bool:
        """Set state, optionally skipping schema validation."""
        raise NotImplementedError

    def set_schema(self, schema: dict[str, Any]) -> None:
        """Set the state schema."""
        raise NotImplementedError

    def build_result_prompt(
        self,
        tool: ToolAttribute,
        arguments: dict[str, Any],
        retry: bool = False,
    ) -> Conversation:
        """Build the prompt for generating a tool result."""
        raise NotImplementedError

    def build_write_state_update_prompt(
        self,
        tool: ToolAttribute,
        arguments: dict[str, Any],
        retry: bool = False,
        retry_error: str | None = None,
    ) -> Conversation:
        """Build prompt for state-first write."""
        raise NotImplementedError

    def build_write_result_prompt(
        self,
        tool: ToolAttribute,
        arguments: dict[str, Any],
        patch_ops: list[dict[str, Any]],
        patch_succeeded: bool,
        pre_patch_state: dict[str, Any] | None = None,
        retry: bool = False,
        retry_error: str | None = None,
    ) -> Conversation:
        """Build prompt for generating a write tool result after state update."""
        raise NotImplementedError

    def apply_result(self, response: Conversation) -> str:
        """Extract the tool result text from an inference response."""
        raise NotImplementedError

    def apply_state_update_returning_patch(
        self, response: Conversation
    ) -> tuple[bool, list[dict[str, Any]], str | None]:
        """Parse and apply a write-state update. Returns (succeeded, ops, error)."""
        raise NotImplementedError

    def apply_state_update(self, response: Conversation) -> bool:
        """Convenience wrapper: apply patch and return success bool."""
        raise NotImplementedError


class EnvironmentRegistry:
    """Builds environments once, then copies N times for parallel samples."""

    def __init__(self):
        """Initialize an empty registry."""
        raise NotImplementedError

    def register_static(self, config: ToolEnvironmentAttribute) -> None:
        """Register an environment that needs no LLM generation."""
        raise NotImplementedError

    def build(
        self,
        config: ToolEnvironmentAttribute,
        tools: list[ToolAttribute],
        inference_engine: Any,
        inference_config: Any,
        scenario_context: str | None = None,
    ) -> None:
        """Build a fully populated environment."""
        raise NotImplementedError

    def create_copies(self, env_id: str, n: int) -> list[GeneratedToolEnvironment]:
        """Return n independent deepcopies of a built environment."""
        raise NotImplementedError


def resolve_env_tool(
    tool_executor: Any,
    tool_call: dict,
    idx_envs: dict[str, GeneratedToolEnvironment] | None,
) -> tuple[GeneratedToolEnvironment | None, ToolAttribute | None]:
    """Look up (env, tool) for an environment-bound tool call."""
    raise NotImplementedError


def is_env_tool_missing_env(tool_executor: Any, tool_call: dict) -> bool:
    """Return True if tool_call targets an ENVIRONMENT tool whose env is missing."""
    raise NotImplementedError


def serialize_env_states(envs: dict[str, GeneratedToolEnvironment]) -> str:
    """Serialize planner-oriented environment summaries as stable JSON."""
    raise NotImplementedError


def init_sample_environments(
    samples: list[dict],
    tools: list[ToolAttribute],
    env_configs: dict[str, ToolEnvironmentAttribute],
    formatter: Any,
    inference_engine: Any,
    inference_config: Any,
) -> list[dict[str, GeneratedToolEnvironment] | None]:
    """Create per-sample environments, reusing builds when config is identical."""
    raise NotImplementedError


def process_env_tool_calls(
    env_items: list[
        tuple[int, str, dict, str, GeneratedToolEnvironment, ToolAttribute]
    ],
    env_result_prompts: list[Conversation],
    turn_tool_msgs: dict[int, list[Message]],
    output_messages: list[list[dict]],
    inference_engine: Any,
    inference_config: Any,
    record_fn: Callable[..., None],
) -> list[int]:
    """Execute batched env tool calls."""
    raise NotImplementedError
