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

"""Process-wide user-simulator inference engine for the verl agent-loop adapter."""

import functools
from typing import cast

from oumi.builders import inference_engines
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


@functools.cache
def user_sim_engine(config_path: str) -> RemoteInferenceEngine:
    """Builds the user-simulator engine once per process.

    verl creates one agent loop per rollout, so building the engine in the loop
    would build one per rollout. The engine must be remote: in-process engines
    would load model weights in every rollout worker.

    Args:
        config_path: Path to an `InferenceConfig` YAML with a remote engine.

    Returns:
        The engine, configured with the YAML's generation params.

    Raises:
        ValueError: If the config has no engine or a non-remote engine.
    """
    cfg = InferenceConfig.from_yaml(config_path)
    if cfg.engine is None or not issubclass(
        inference_engines.ENGINE_MAP[cfg.engine], RemoteInferenceEngine
    ):
        raise ValueError(
            f"The user simulator needs a remote inference engine; got {cfg.engine} "
            f"from {config_path}."
        )
    return cast(
        RemoteInferenceEngine,
        inference_engines.build_inference_engine(
            engine_type=cfg.engine,
            model_params=cfg.model,
            remote_params=cfg.remote_params,
            generation_params=cfg.generation,
        ),
    )


async def generate_reply(config_path: str, conversation: Conversation) -> str:
    """Generates one reply on the caller's event loop.

    Args:
        config_path: Path to the simulator's `InferenceConfig` YAML.
        conversation: The prompt.

    Returns:
        The reply text.

    Raises:
        RuntimeError: If the engine returns non-text content.
    """
    response = await user_sim_engine(config_path).generate_one(conversation)
    content = response.messages[-1].content
    if not isinstance(content, str):
        raise RuntimeError(f"User-sim engine returned non-text content: {content!r}")
    return content
