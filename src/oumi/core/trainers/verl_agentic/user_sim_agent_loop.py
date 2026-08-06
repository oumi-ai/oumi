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

"""verl agent loop adding persona-driven simulated-user turns to ToolAgentLoop."""

from __future__ import annotations

import asyncio
from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import (  # pyright: ignore[reportMissingImports]
    AgentState,
    ToolAgentLoop,
)

from oumi.core.rollout.user_sim import (
    DEFAULT_DONE_SENTINEL,
    DEFAULT_MAX_TURNS,
    RolloutState,
    next_user_turn,
)
from oumi.core.trainers.verl_agentic.user_sim_provider import infer_one, user_sim_engine


# Deliberately NOT decorated with verl's @register. That decorator does
# `_agent_loop_registry[name] = {"_target_": fqdn}` — an unconditional overwrite that
# drops every other key. Since importing this module is what resolves `_target_`, the
# decorator would wipe the `user_sim_inference` path the agent-loop YAML supplies.
# Registration comes from `agent_loop_config_path`; see user_sim_agent_loop.yaml.
class UserSimToolAgentLoop(ToolAgentLoop):
    """ToolAgentLoop plus a simulated user that speaks when the assistant stops."""

    def __init__(
        self, *args: Any, user_sim_inference: str | None = None, **kwargs: Any
    ):
        """Stores the user-sim config path; the engine itself is cached per process."""
        super().__init__(*args, **kwargs)
        self._sim_config_path = user_sim_inference
        self._sim: RolloutState | None = None

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        """Runs the rollout.

        Returns:
            verl's `AgentLoopOutput`, with `sim_user_turns` in `extra_fields` when
            a simulated user took part.
        """
        sim_kwargs = (kwargs.get("extra_info") or {}).get("interaction_kwargs") or {}
        if sim_kwargs:
            if not self._sim_config_path:
                raise ValueError(
                    "Row carries interaction_kwargs but the agent-loop config has no "
                    "'user_sim_inference' path. Add it to the YAML that "
                    "agent_loop_config_path points at."
                )
            self._sim = RolloutState(
                persona=sim_kwargs["user_persona"],
                goal=sim_kwargs.get("goal", ""),
                max_turns=sim_kwargs.get("max_turns") or DEFAULT_MAX_TURNS,
            )
        output = await super().run(sampling_params, **kwargs)
        # Written once, after the rollout: incremental writes to extra_fields make the
        # dict truthy, which stops verl copying engine telemetry into it.
        if self._sim is not None:
            output.extra_fields["sim_user_turns"] = self._sim.turn_idx
        return output

    async def _handle_generating_state(
        self,
        agent_data: Any,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        """Adds a simulated-user turn when the assistant stops without calling a tool.

        Returns:
            The next agent state.
        """
        state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination
        )
        # `_sim` and `_sim_config_path` are set together in `run()`; both are absent
        # for tool-only rows, which take the stock ToolAgentLoop path.
        if state is not AgentState.TERMINATED or self._sim is None:
            return state
        if self._hard_cap_hit(agent_data, ignore_termination):
            return state
        return await self._run_simulated_user_turn(
            agent_data, self._sim, str(self._sim_config_path)
        )

    def _hard_cap_hit(self, agent_data: Any, ignore_termination: bool = False) -> bool:
        """Mirrors ToolAgentLoop._handle_generating_state's termination checks.

        Returns:
            Whether the parent terminated because a hard cap was reached.
        """
        if (
            not ignore_termination
            and len(agent_data.response_mask) >= self.response_length
        ):
            return True
        if (
            self.max_assistant_turns
            and agent_data.assistant_turns >= self.max_assistant_turns
        ):
            return True
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return True
        return False

    async def _run_simulated_user_turn(
        self, agent_data: Any, sim: RolloutState, config_path: str
    ) -> AgentState:
        """Generates one simulated-user turn and appends it as environment text.

        Returns:
            `GENERATING` if the conversation continues, else `TERMINATED`.
        """
        await self._sync_assistant_message(agent_data)
        engine, cfg = user_sim_engine(config_path)
        done, text, score = await asyncio.to_thread(
            next_user_turn,
            sim,
            agent_data.messages,
            lambda conversation: infer_one(engine, cfg, conversation),
            DEFAULT_DONE_SENTINEL,
        )
        agent_data.user_turns += 1
        agent_data.turn_scores.append(score)
        if done:
            return AgentState.TERMINATED
        fits = await self._append_environment_turn(
            agent_data, [{"role": "user", "content": text}]
        )
        return AgentState.GENERATING if fits else AgentState.TERMINATED

    async def _append_environment_turn(
        self, agent_data: Any, messages: list[dict[str, Any]]
    ) -> bool:
        """Appends environment-authored messages, which never carry gradient.

        Mirrors `_handle_processing_tools_state`: the mask is 0 because the policy did
        not produce these tokens, and the length check precedes any mutation.

        Returns:
            Whether the turn fit within the response budget.
        """
        agent_data.messages.extend(messages)
        response_ids = await self.apply_chat_template(
            messages, remove_system_prompt=True
        )
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return False
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        return True

    async def _sync_assistant_message(self, agent_data: Any) -> None:
        """Copies the assistant's turn into `messages`, which stock verl never does.

        Tokens-only bookkeeping: the assistant's ids are already in `prompt_ids` with
        mask 1, so this must not touch them. The user simulator reads `messages`.
        """
        text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(
                agent_data.response_ids, skip_special_tokens=True
            ),
        )
        agent_data.messages.append({"role": "assistant", "content": text})
