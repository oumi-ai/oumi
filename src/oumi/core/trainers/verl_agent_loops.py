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

"""Single-turn verl agent loop that renders a row's tool definitions.

verl's own ``SingleTurnAgentLoop`` applies the chat template without tools, so a
row's schemas never reach the policy. Its tool schemas come only from a
config-level ``tool_config_path``, which is one fixed set for the whole run
rather than per row. This loop reads the per-row schemas the Oumi dataset
exporter writes and hands them to the model's own chat template, matching how
the same conversation is rendered for SFT.

Hydra imports this module inside the rollout worker, via the ``_target_`` in
``verl_agent_loop_config.yaml``; nothing in Oumi imports it, so the verl imports
below only have to resolve where rollouts actually run.

The override targets ``apply_chat_template``, which verl's single-turn loop
tokenizes with through verl 0.8. verl 0.9 replaces it with
``ct_build_initial_tokens``, keeping the same ``tools`` argument, so moving past
Oumi's ``verl<0.8`` pin moves the override to that name. ``run`` raises when the
overridden method is never called, rather than letting the schemas go missing
from the prompt unnoticed.
"""

from __future__ import annotations

import json
from typing import Any

from verl.experimental.agent_loop.agent_loop import (  # pyright: ignore[reportMissingImports]
    AgentLoopOutput,
)
from verl.experimental.agent_loop.single_turn_agent_loop import (  # pyright: ignore[reportMissingImports]
    SingleTurnAgentLoop,
)

# The verl row column holding the JSON-encoded tool schemas. Must match
# `_TOOLS_JSON_KEY` in `verl_grpo_trainer`, which writes it.
_TOOLS_JSON_KEY = "tools_json"

# The agent loop's registered name, as declared in `verl_agent_loop_config.yaml`.
OUMI_SINGLE_TURN_AGENT_NAME = "oumi_single_turn_agent"


def _decode_tools(tools_json: str | None) -> list[dict] | None:
    """Decodes the row's tool schemas, or None when the row declares none.

    Args:
        tools_json: The row's ``tools_json`` column.

    Returns:
        The tool schemas as a list of OpenAI-format dicts, or ``None``.

    Raises:
        ValueError: If the column is present but does not decode to a list of
            objects. The exporter writes it from validated tool definitions, so
            this means the Parquet row did not come from Oumi; rendering the
            prompt without the tools would silently train against the wrong
            prompt, which is the bug this loop exists to prevent.
    """
    if not tools_json:
        return None
    if not isinstance(tools_json, str):
        raise ValueError(
            f"'{_TOOLS_JSON_KEY}' must be a JSON string, "
            f"got {type(tools_json).__name__}."
        )
    try:
        tools = json.loads(tools_json)
    except ValueError as e:
        raise ValueError(f"'{_TOOLS_JSON_KEY}' is not valid JSON.") from e
    if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
        raise ValueError(
            f"'{_TOOLS_JSON_KEY}' must encode a list of tool schema objects."
        )
    return tools or None


class ToolAwareSingleTurnAgentLoop(SingleTurnAgentLoop):
    """Single-turn agent loop that renders the row's tools into the prompt.

    The tools are taken from ``run()``'s kwargs and injected in
    ``apply_chat_template``, rather than by reimplementing ``run()``, so the
    rollout keeps whatever the installed verl's single-turn loop does with
    vision inputs, sampling and metrics.

    verl instantiates one agent loop per rollout sample, so holding the row's
    tools on the instance does not leak them between samples.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the loop with no row tools until ``run()`` supplies them."""
        super().__init__(*args, **kwargs)
        self._row_tools: list[dict] | None = None
        self._template_applied = False

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Runs one single-turn rollout, rendering this row's tools.

        Raises:
            RuntimeError: If the row carries tools but the installed verl never
                called the overridden template method, which means it tokenizes
                the prompt somewhere else now and the schemas were dropped.
        """
        self._row_tools = _decode_tools(kwargs.get(_TOOLS_JSON_KEY))
        self._template_applied = False
        output = await super().run(sampling_params, **kwargs)
        if self._row_tools is not None and not self._template_applied:
            raise RuntimeError(
                f"{type(self).__name__} never saw apply_chat_template, so this "
                "row's tool definitions did not reach the prompt. The installed "
                "verl builds prompts elsewhere (verl 0.9 replaced it with "
                "'ct_build_initial_tokens'); override that method instead."
            )
        return output

    async def apply_chat_template(
        self, messages: list[dict], tools: list[dict] | None = None, **kwargs: Any
    ):
        """Applies the chat template, defaulting ``tools`` to the row's schemas."""
        self._template_applied = True
        return await super().apply_chat_template(
            messages, tools=tools if tools is not None else self._row_tools, **kwargs
        )
