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

from functools import cache

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine

# Engines that hold a model in-process. Rejected before `build_inference_engine` so we
# never pay to load weights just to refuse them.
_IN_PROCESS_ENGINES = frozenset(
    {
        InferenceEngineType.NATIVE,
        InferenceEngineType.VLLM,
        InferenceEngineType.LLAMACPP,
    }
)


@cache
def user_sim_engine(config_path: str) -> tuple[RemoteInferenceEngine, InferenceConfig]:
    """Builds the user-sim engine once per process.

    Agent loops are instantiated per trajectory, so building the engine in the loop's
    `__init__` would construct one per rollout.

    Returns:
        The engine and the config it was built from.
    """
    cfg = InferenceConfig.from_yaml(config_path)
    if cfg.engine is None:
        raise ValueError(f"No inference engine set in {config_path}.")
    if cfg.engine in _IN_PROCESS_ENGINES:
        raise ValueError(
            f"The user simulator requires a remote inference engine; got {cfg.engine} "
            f"from {config_path}. One engine is shared by every concurrent rollout in a "
            "worker, so an in-process engine would need a lock that serializes them all. "
            "Use REMOTE_VLLM, SGLANG, or a hosted provider."
        )
    engine = build_inference_engine(
        engine_type=cfg.engine,
        model_params=cfg.model,
        remote_params=cfg.remote_params,
    )
    # Free: every engine that loads weights was already rejected above, so nothing
    # expensive reaches this. Catches a new engine type added between releases.
    if not isinstance(engine, RemoteInferenceEngine):
        raise ValueError(
            f"User-sim engine {type(engine).__name__} is not remote-backed; "
            f"got {cfg.engine} from {config_path}."
        )
    return engine, cfg


def infer_one(
    engine: RemoteInferenceEngine, cfg: InferenceConfig, conversation: Conversation
) -> str:
    """Runs one blocking user-sim generation.

    Returns:
        The simulated user's reply text.
    """
    results = engine.infer([conversation], inference_config=cfg)
    content = results[0].messages[-1].content
    if not isinstance(content, str):
        raise RuntimeError(f"user-sim engine returned non-text content: {type(content)}")
    return content
