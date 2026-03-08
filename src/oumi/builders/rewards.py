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

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any

from oumi.core.configs import TrainingParams
from oumi.core.registry import REGISTRY


def _apply_reward_function_kwargs(
    reward_func: Callable, reward_function_kwargs: dict[str, Any] | None
) -> Callable:
    if not reward_function_kwargs:
        return reward_func

    @wraps(reward_func)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        # Configured kwargs take precedence over per-sample or call-time kwargs.
        merged_kwargs = {**kwargs, **reward_function_kwargs}
        return reward_func(*args, **merged_kwargs)

    return _wrapped


def build_reward_functions(config: TrainingParams) -> list[Callable]:
    """Builds the reward functions.

    Example config (YAML)::

        reward_functions:
          - rubric_reward
          - gsm8k
        reward_function_kwargs:
          rubric_reward:
            judge_panel_path: "configs/.../judge_panel.yaml"
          gsm8k:
            strict: true
    """
    if not config.reward_functions:
        return []

    # Import to ensure GRPO reward functions are added to REGISTRY.
    import oumi.datasets.grpo.rewards as grpo_rewards  # noqa: F401

    function_names = [name for name in config.reward_functions if name]
    kwargs_map = config.reward_function_kwargs or {}

    if kwargs_map:
        if not isinstance(kwargs_map, Mapping):
            raise ValueError(
                "reward_function_kwargs must be a dict keyed by reward "
                "function name, e.g. {rubric_reward: {judge_panel_path: ...}}."
            )
        unexpected_keys = set(kwargs_map.keys()) - set(function_names)
        if unexpected_keys:
            raise ValueError(
                "reward_function_kwargs must be a dict keyed by reward function "
                "name. Unexpected keys not listed in reward_functions: "
                f"{sorted(unexpected_keys)}."
            )
        for key, value in kwargs_map.items():
            if not isinstance(value, Mapping):
                raise ValueError(
                    "reward_function_kwargs entries must be dicts. "
                    f"Entry '{key}' is {type(value).__name__}. Use "
                    "reward_function_kwargs: {<func_name>: {arg: value}}."
                )

    result: list[Callable] = []
    for name in function_names:
        reward_function = REGISTRY.get_reward_function(name)
        if not reward_function:
            raise KeyError(f"reward_function `{name}` was not found in the registry.")
        result.append(
            _apply_reward_function_kwargs(reward_function, kwargs_map.get(name))
        )

    return result
