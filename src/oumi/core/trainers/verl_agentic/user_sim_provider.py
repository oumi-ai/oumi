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

from __future__ import annotations

import functools

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.types.conversation import Conversation

# Engines that hold the model in-process. One engine is shared by every concurrent
# rollout in a worker, so these would serialize all of them. Rejected before
# `build_inference_engine` so no weights are loaded just to be refused.
# test_every_engine_is_classified keeps this set in sync with ENGINE_MAP.
_IN_PROCESS_ENGINES = frozenset(
    {
        InferenceEngineType.NATIVE,
        InferenceEngineType.VLLM,
        InferenceEngineType.LLAMACPP,
    }
)


@functools.cache
def user_sim_engine(
    config_path: str,
) -> tuple[BaseInferenceEngine, InferenceConfig]:
    """Builds the user-simulator engine once per process.

    verl creates one agent loop per trajectory, so building the engine in the
    loop's `__init__` would build one per rollout.

    Args:
        config_path: Path to an `InferenceConfig` YAML with a remote engine.

    Returns:
        The engine and the config it was built from.

    Raises:
        ValueError: If the config has no engine or an in-process engine.
    """
    cfg = InferenceConfig.from_yaml(config_path)
    if cfg.engine is None:
        raise ValueError(f"No inference engine set in {config_path}.")
    if cfg.engine in _IN_PROCESS_ENGINES:
        raise ValueError(
            "The user simulator requires a remote inference engine; got "
            f"{cfg.engine} from {config_path}. Use REMOTE_VLLM, SGLANG, or a "
            "hosted provider."
        )
    engine = build_inference_engine(
        engine_type=cfg.engine,
        model_params=cfg.model,
        remote_params=cfg.remote_params,
    )
    return engine, cfg


def infer_one(config_path: str, conversation: Conversation) -> str:
    """Runs one blocking generation with the engine from `user_sim_engine`.

    Args:
        config_path: Path to the simulator's `InferenceConfig` YAML.
        conversation: The prompt.

    Returns:
        The generated reply text.

    Raises:
        RuntimeError: If the engine returns non-text content.
    """
    engine, cfg = user_sim_engine(config_path)
    results = engine.infer([conversation], inference_config=cfg)
    content = results[0].messages[-1].content
    if not isinstance(content, str):
        raise RuntimeError(
            f"User-sim engine returned non-text content: {type(content)}"
        )
    return content
