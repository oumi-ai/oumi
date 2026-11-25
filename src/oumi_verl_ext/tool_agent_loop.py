# Copyright 2026 - Oumi
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

"""verl 0.9.1's tool agent loop, also returning the conversation for the judge.

verl imports it in its rollout workers, so it imports only verl, the standard
library and ``oumi_verl_ext``.
"""

import json
from typing import Any

from verl.experimental.agent_loop.agent_loop import (  # pyright: ignore[reportMissingImports]
    AgentLoopOutput,
)
from verl.experimental.agent_loop.tool_agent_loop import (  # pyright: ignore[reportMissingImports]
    AgentData,
    AgentState,
    ToolAgentLoop,
)

from oumi_verl_ext.tool_episode import judge_transcript


class OumiToolAgentLoop(ToolAgentLoop):
    """verl's tool loop, adding the judge's transcript and tools to extra_fields.

    verl builds one loop per trajectory, so the instance state is per trajectory.
    """

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Runs verl's episode, then adds the judge's transcript and tools."""
        self._messages: list[dict[str, Any]] = []
        self._unparsed_turn_text: str | None = None
        # In a split where other rows carry tools_kwargs, parquet reads a row
        # without it back as None, and verl's _call_tool would call .get on it.
        if kwargs.get("tools_kwargs") is None:
            kwargs["tools_kwargs"] = {}
        output = await super().run(sampling_params, **kwargs)
        transcript = judge_transcript(
            self._messages,
            self._unparsed_turn_text,
            self.tokenizer.all_special_tokens,
        )
        output.extra_fields["oumi_tool_transcript_json"] = json.dumps(transcript)
        output.extra_fields["oumi_tool_schemas_json"] = json.dumps(self.tool_schemas)
        return output

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        self._messages = agent_data.messages
        prior_length = len(agent_data.response_mask)
        message_count = len(agent_data.messages)
        state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination
        )
        if len(agent_data.messages) == message_count:
            # verl stopped before parsing this turn; keep only its in-budget tokens.
            self._unparsed_turn_text = self.tokenizer.decode(
                agent_data.response_ids[: self.response_length - prior_length]
            )
        return state
