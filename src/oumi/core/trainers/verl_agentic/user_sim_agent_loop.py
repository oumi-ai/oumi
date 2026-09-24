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
import functools
from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import (  # pyright: ignore[reportMissingImports]
    AgentData,
    AgentState,
    ToolAgentLoop,
)

from oumi.core.rollout.user_sim import RolloutState, next_user_turn
from oumi.core.trainers.verl_agentic.user_sim_provider import infer_one


# Deliberately not decorated with verl's `@register`: it overwrites the registry
# entry with `{"_target_": ...}` alone, which would drop the `user_sim_inference`
# key that the agent-loop YAML (`agent_loop_config_path`) supplies.
class UserSimToolAgentLoop(ToolAgentLoop):
    """ToolAgentLoop plus a simulated user that replies when the assistant stops.

    Rows without `interaction_kwargs` behave exactly like `ToolAgentLoop`.
    """

    def __init__(
        self, *args: Any, user_sim_inference: str | None = None, **kwargs: Any
    ):
        """Initializes the loop.

        Args:
            *args: Forwarded to `ToolAgentLoop`.
            user_sim_inference: Path to the simulator's `InferenceConfig` YAML.
            **kwargs: Forwarded to `ToolAgentLoop`.
        """
        super().__init__(*args, **kwargs)
        self._sim_config_path = user_sim_inference
        # verl creates one loop per rollout, so the fields below are per-conversation.
        self._sim: RolloutState | None = None
        # What the simulated user has seen. Kept apart from verl's `messages`, which
        # holds tool results without the assistant turn that requested them.
        self._sim_history: list[dict[str, Any]] = []

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        """Runs one rollout.

        Returns:
            verl's `AgentLoopOutput`; `extra_fields["sim_user_turns"]` is set when
            a simulated user took part.

        Raises:
            ValueError: If the row has `interaction_kwargs` but the loop has no
                `user_sim_inference` config.
        """
        sim_kwargs = kwargs["extra_info"].get("interaction_kwargs")
        if sim_kwargs:
            if not self._sim_config_path:
                raise ValueError(
                    "Row carries interaction_kwargs but the agent-loop config has no "
                    "'user_sim_inference' path. Add it to the YAML that "
                    "agent_loop_config_path points at."
                )
            self._sim = RolloutState(
                persona=sim_kwargs["user_persona"],
                goal=sim_kwargs["goal"],
                max_turns=sim_kwargs["max_turns"],
            )
            self._sim_history = [dict(m) for m in kwargs["raw_prompt"]]
        output = await super().run(sampling_params, **kwargs)
        # Set only after the parent returns: verl copies engine telemetry into
        # `extra_fields` only when it is still empty.
        if self._sim is not None:
            output.extra_fields["sim_user_turns"] = self._sim.turn_idx
        return output

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        """Lets the simulated user reply once the assistant stops calling tools."""
        state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination
        )
        sim, config_path = self._sim, self._sim_config_path
        # PROCESSING_TOOLS means the assistant has not addressed the user yet.
        # The `config_path` check only narrows the type; `run()` guarantees it.
        if sim is None or config_path is None or state is not AgentState.TERMINATED:
            return state
        assistant_text = await asyncio.to_thread(
            self.tokenizer.decode, agent_data.response_ids, skip_special_tokens=True
        )
        self._sim_history.append({"role": "assistant", "content": assistant_text})
        if self._hard_cap_hit(agent_data, ignore_termination):
            return state
        return await self._run_simulated_user_turn(agent_data, sim, config_path)

    def _hard_cap_hit(self, agent_data: AgentData, ignore_termination: bool) -> bool:
        """Mirrors the termination checks in `ToolAgentLoop._handle_generating_state`.

        Returns:
            Whether the parent stopped because a length or turn cap was reached.
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
        return bool(
            self.max_user_turns and agent_data.user_turns >= self.max_user_turns
        )

    async def _run_simulated_user_turn(
        self, agent_data: AgentData, sim: RolloutState, config_path: str
    ) -> AgentState:
        """Generates one simulated-user reply and appends it to the rollout.

        Returns:
            `GENERATING` if the conversation continues, else `TERMINATED`.
        """
        if sim.turn_idx >= sim.max_turns:
            return AgentState.TERMINATED
        done, text = await asyncio.to_thread(
            next_user_turn,
            sim,
            self._sim_history,
            functools.partial(infer_one, config_path),
        )
        agent_data.user_turns += 1
        if done:
            return AgentState.TERMINATED
        message = {"role": "user", "content": text}
        self._sim_history.append(message)
        fits = await self._append_environment_turn(agent_data, message)
        return AgentState.GENERATING if fits else AgentState.TERMINATED

    async def _append_environment_turn(
        self, agent_data: AgentData, message: dict[str, Any]
    ) -> bool:
        """Appends a message the policy did not write, with loss mask 0.

        Mirrors `ToolAgentLoop._handle_processing_tools_state`. Nothing is mutated
        if the turn does not fit in the response budget.

        Returns:
            Whether the turn fit within the response budget.
        """
        response_ids = await self.apply_chat_template(
            [message], remove_system_prompt=True
        )
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return False
        agent_data.messages.append(message)
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        return True
