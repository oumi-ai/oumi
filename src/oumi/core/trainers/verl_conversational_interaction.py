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
import contextlib
import threading
from uuid import uuid4

from verl.interactions.base import (  # pyright: ignore[reportMissingImports]
    BaseInteraction,
)

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.trainers.user_sim import (
    DEFAULT_DONE_SENTINEL,
    DEFAULT_MAX_TURNS,
    RolloutState,
    next_user_turn,
)
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class OumiVerlInteraction(BaseInteraction):
    """Simulates user turns for verl rollouts."""

    def __init__(self, config: dict):
        """Build the user-sim inference engine from config."""
        super().__init__(config)
        self._infer_config = InferenceConfig.from_yaml(config["user_sim_inference"])
        self._engine = build_inference_engine(
            engine_type=self._infer_config.engine or InferenceEngineType.NATIVE,
            model_params=self._infer_config.model,
            remote_params=self._infer_config.remote_params,
        )
        # verl shares one interaction across concurrent rollouts, and local engines
        # aren't safe for concurrent infer(); remote (HTTP) ones are.
        self._infer_lock = (
            contextlib.nullcontext()
            if isinstance(self._engine, RemoteInferenceEngine)
            else threading.Lock()
        )
        self._default_max_turns = config.get("max_turns", DEFAULT_MAX_TURNS)
        self._done_sentinel = config.get("done_sentinel", DEFAULT_DONE_SENTINEL)
        self._state: dict[str, RolloutState] = {}

    async def start_interaction(
        self, instance_id=None, *, user_persona, goal="", max_turns=None, **kwargs
    ) -> str:
        """Register per-rollout user-sim state, keyed by instance_id."""
        iid = instance_id or uuid4().hex
        self._state[iid] = RolloutState(
            persona=user_persona,
            goal=goal,
            max_turns=(max_turns if max_turns is not None else self._default_max_turns),
        )
        return iid

    async def generate_response(
        self, instance_id, messages, **kwargs
    ) -> tuple[bool, str, float, dict]:
        """Produce the next simulated-user turn.

        Returns:
            A tuple of whether to terminate, the simulated-user response, its score,
            and metadata.
        """
        state = self._state[instance_id]
        done, text, score = await asyncio.to_thread(
            next_user_turn, state, messages, self._infer_one, self._done_sentinel
        )
        return done, text, score, {}

    def _infer_one(self, conversation: Conversation) -> str:
        with self._infer_lock:
            results = self._engine.infer(
                [conversation], inference_config=self._infer_config
            )
        content = results[0].messages[-1].content
        if not isinstance(content, str):
            raise RuntimeError(
                f"user-sim engine returned non-text content: {type(content)}"
            )
        return content

    async def calculate_score(self, instance_id, **kwargs) -> float:
        """Turn-level score is unused; reward is trajectory-level."""
        return 0.0

    async def finalize_interaction(self, instance_id, **kwargs) -> None:
        """Drop the rollout's user-sim state."""
        self._state.pop(instance_id, None)
