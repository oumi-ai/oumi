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

"""verl 0.9.1's tool agent loop with per-row tools, for multi-turn tool-call GRPO.

verl registers it from ``agent_loops.yaml`` and imports it in the rollout workers,
so this module imports only verl, the standard library and ``oumi_verl_ext``.
Single-turn tool-call GRPO is the same loop with ``max_assistant_turns: 1``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from verl.experimental.agent_loop.agent_loop import (  # pyright: ignore[reportMissingImports]
    AgentLoopOutput,
)
from verl.experimental.agent_loop.tool_agent_loop import (  # pyright: ignore[reportMissingImports]
    AgentData,
    AgentState,
    ToolAgentLoop,
)
from verl.tools.base_tool import BaseTool  # pyright: ignore[reportMissingImports]
from verl.tools.schemas import (  # pyright: ignore[reportMissingImports]
    OpenAIFunctionToolSchema,
)

from oumi_verl_ext.tool_episode import (
    ENDS_EPISODE_CONFIG_KEY,
    TOOLS_JSON_KEY,
    CallCounts,
    EpisodeTools,
    choose_episode_tools,
    episode_extra_fields,
    judge_transcript,
    parse_row_tools,
    verl_parser_schema,
)


class OumiToolAgentLoop(ToolAgentLoop):
    """verl's tool loop with row tools, ending tools, the last-turn fix and counters.

    verl builds one loop per trajectory, so the instance state is per trajectory.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Takes over both turn caps from verl."""
        super().__init__(*args, **kwargs)
        self._turn_cap = self.max_assistant_turns
        self._user_turn_cap = self.max_user_turns
        # verl checks both caps before parsing a turn, so that turn's calls would go
        # unseen; with both nulled, only the response budget ends a turn unparsed.
        self.max_assistant_turns = None
        self.max_user_turns = None
        self._call_start = getattr(self.tool_parser, "tool_call_start_token", None)
        self._counts = CallCounts()
        self._truncated_turn_text: str | None = None
        self._row_tools: list[dict[str, Any]] = []
        self._episode = EpisodeTools(schemas=[], executable=frozenset())
        self._messages: list[dict[str, Any]] = []

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Runs verl's episode, then adds the judge's transcript and the counters."""
        self._row_tools = parse_row_tools(kwargs.get(TOOLS_JSON_KEY))
        # In a split where other rows carry tools_kwargs, parquet reads a row
        # without it back as None, and verl's _call_tool would call .get on it.
        if kwargs.get("tools_kwargs") is None:
            kwargs["tools_kwargs"] = {}
        output = await super().run(sampling_params, **kwargs)
        transcript = judge_transcript(
            self._messages,
            self._truncated_turn_text,
            self.tokenizer.all_special_tokens,
        )
        output.extra_fields.update(
            episode_extra_fields(transcript, self._episode.schemas, self._counts)
        )
        return output

    async def _handle_pending_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        self._messages = agent_data.messages
        run_tools = agent_data._active_tools
        self._episode = choose_episode_tools(
            self._row_tools,
            {
                name: tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
                for name, tool in run_tools.items()
            },
            frozenset(
                name
                for name, tool in run_tools.items()
                if isinstance(tool, BaseTool)
                and tool.config.get(ENDS_EPISODE_CONFIG_KEY)
            ),
        )
        agent_data._active_tool_schemas = self._episode.schemas
        schemas = {tool["function"]["name"]: tool for tool in self._episode.schemas}
        # Stand-ins that let verl's parser read calls to tools the loop never runs.
        agent_data._active_tools = {
            **run_tools,
            **{
                name: SimpleNamespace(
                    tool_schema=OpenAIFunctionToolSchema(
                        **verl_parser_schema(schemas[name])
                    )
                )
                for name in self._episode.ending
            },
        }
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        prior_length = len(agent_data.response_mask)
        message_count = len(agent_data.messages)
        state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination
        )
        if len(agent_data.messages) == message_count:
            # The response budget ended the turn before verl parsed it, so
            # agent_data.tool_calls still holds the previous turn's calls. verl caps
            # a turn by the context left, not the budget left, so keep only the
            # turn's tokens that the trained response keeps.
            text = self.tokenizer.decode(
                agent_data.response_ids[: self.response_length - prior_length]
            )
            self._truncated_turn_text = text
            self._counts.add_turn(
                attempted=self._attempted_calls(text, parsed=0),
                call_names=(),
                max_parallel_calls=self.max_parallel_calls,
                cut=True,
            )
            return state
        # verl checks the user cap only after the next generate, so the model still
        # answers from the last tool results; calls in that turn don't run.
        call_names = [call.name for call in agent_data.tool_calls]
        ends = self._counts.add_turn(
            attempted=self._attempted_calls(
                self.tokenizer.decode(agent_data.response_ids), parsed=len(call_names)
            ),
            call_names=call_names,
            max_parallel_calls=self.max_parallel_calls,
            ending=self._episode.ending,
            capped=bool(self._user_turn_cap)
            and agent_data.user_turns >= self._user_turn_cap,
        )
        return AgentState.TERMINATED if ends else state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        if self._turn_cap and agent_data.assistant_turns >= self._turn_cap:
            return AgentState.TERMINATED
        return state

    def _attempted_calls(self, text: str, parsed: int) -> int:
        # Without a call-start marker (gpt-oss, kimi), malformed calls can't be seen.
        return text.count(self._call_start) if self._call_start else parsed
