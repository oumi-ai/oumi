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

"""verl Interaction that produces persona-conditioned simulated-user turns."""

import asyncio
from uuid import uuid4

from verl.interactions.base import BaseInteraction

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.trainers.verl_conversational_turn import (
    DEFAULT_DONE_SENTINEL,
    RolloutState,
    next_user_turn,
)
from oumi.core.types.conversation import Conversation


class OumiVerlInteraction(BaseInteraction):
    """Drives the simulated user; verl owns the loop and the policy turns."""

    def __init__(self, config: dict):
        """Build the user-sim inference engine from `config["user_sim_inference"]`."""
        super().__init__(config)
        self._infer_config = InferenceConfig.from_yaml(config["user_sim_inference"])
        self._engine = build_inference_engine(
            engine_type=self._infer_config.engine,
            model_params=self._infer_config.model,
            remote_params=self._infer_config.remote_params,
        )
        self._default_max_turns = config.get("max_turns", 6)
        self._done_sentinel = config.get("done_sentinel", DEFAULT_DONE_SENTINEL)
        self._state: dict[str, RolloutState] = {}

    async def start_interaction(
        self, instance_id=None, *, user_persona, max_turns=None, goal=None, **kw
    ) -> str:
        """Register per-rollout user-sim state, keyed by `instance_id`."""
        iid = instance_id or uuid4().hex
        self._state[iid] = RolloutState(
            persona=user_persona,
            max_turns=max_turns or self._default_max_turns,
            goal=goal,
        )
        return iid

    async def generate_response(self, instance_id, messages, **kw):
        """Delegate the next simulated-user turn to `next_user_turn`."""
        state = self._state[instance_id]
        done, text, score = await asyncio.to_thread(
            next_user_turn, state, messages, self._infer_one, self._done_sentinel
        )
        return done, text, score, {}

    def _infer_one(self, conversation: Conversation) -> str:
        results = self._engine.infer(
            [conversation], inference_config=self._infer_config
        )
        if not results or not results[0].messages:
            return ""
        content = results[0].messages[-1].content
        return content if isinstance(content, str) else ""

    async def calculate_score(self, instance_id, **kw) -> float:
        """No turn-level score; trajectory reward is custom_reward_function's job."""
        return 0.0  # shaping deferred to Phase 2

    async def finalize_interaction(self, instance_id, **kw) -> None:
        """Drop the rollout's user-sim state."""
        self._state.pop(instance_id, None)
