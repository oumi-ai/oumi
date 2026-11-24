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

from oumi_verl_ext.tool_episode import (
    DROPPED_CALLS_KEY,
    MALFORMED_CALLS_KEY,
    call_problems,
    judge_transcript,
)
from oumi_verl_ext.tool_schemas import with_full_schemas


class OumiToolAgentLoop(ToolAgentLoop):
    """verl's tool loop, adding the judge's transcript and tools to extra_fields.

    It also counts the calls verl skips without a metric: malformed and dropped.
    verl builds one loop per trajectory, so the instance state is per trajectory.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Shows the model and the judge the tool config file's full tool schemas."""
        super().__init__(*args, **kwargs)
        self.tool_schemas = with_full_schemas(
            self.tool_schemas, self.rollout_config.multi_turn.get("tool_config_path")
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Runs verl's episode, then adds the judge's transcript, tools and counts."""
        self._messages: list[dict[str, Any]] = []
        self._unparsed_turn_text: str | None = None
        self._malformed_calls = self._dropped_calls = 0
        output = await super().run(sampling_params, **kwargs)
        transcript = judge_transcript(
            self._messages,
            self._unparsed_turn_text,
            self.tokenizer.all_special_tokens,
        )
        output.extra_fields["oumi_tool_transcript_json"] = json.dumps(transcript)
        output.extra_fields["oumi_tool_schemas_json"] = json.dumps(self.tool_schemas)
        output.extra_fields[MALFORMED_CALLS_KEY] = self._malformed_calls
        output.extra_fields[DROPPED_CALLS_KEY] = self._dropped_calls
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
        else:
            malformed, dropped = call_problems(
                self.tokenizer.decode(agent_data.response_ids),
                getattr(self.tool_parser, "tool_call_start_token", None),
                len(agent_data.tool_calls),
                self.max_parallel_calls,
            )
            self._malformed_calls += malformed
            self._dropped_calls += dropped
        return state
