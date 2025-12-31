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


def _is_per_function_kwargs(
    reward_function_kwargs: dict[str, Any] | None, function_names: list[str]
) -> bool:
    # Treat kwargs as per-function only when keys match reward function names
    # and each value is itself a kwargs mapping.
    if not reward_function_kwargs:
        return False
    if not isinstance(reward_function_kwargs, Mapping):
        return False
    if not set(reward_function_kwargs.keys()).issubset(set(function_names)):
        return False
    return all(isinstance(value, Mapping) for value in reward_function_kwargs.values())


def build_reward_functions(config: TrainingParams) -> list[Callable]:
    """Builds the reward functions."""
    result: list[Callable] = []
    if config.reward_functions is not None:
        # Import to ensure GRPO reward functions are added to REGISTRY.
        import oumi.datasets.grpo.rewards as grpo_rewards  # noqa: F401

        function_names = [name for name in config.reward_functions if name]
        # Detect per-function kwargs when keyed by reward function name.
        # Example (per-function):
        #   reward_function_kwargs:
        #     rubric_reward: {judge_panel_path: "configs/.../judge_panel.yaml"}
        #     gsm8k: {strict: true}
        # Example (global for all reward functions):
        #   reward_function_kwargs: {judge_model: "gpt-4o-mini"}
        per_function_kwargs = _is_per_function_kwargs(
            config.reward_function_kwargs, function_names
        )
        if (
            config.reward_function_kwargs
            and isinstance(config.reward_function_kwargs, Mapping)
            and set(config.reward_function_kwargs.keys()).issubset(set(function_names))
            and not per_function_kwargs
        ):
            bad_key, bad_value = next(
                (
                    (key, value)
                    for key, value in config.reward_function_kwargs.items()
                    if not isinstance(value, Mapping)
                ),
                ("<unknown>", None),
            )
            raise ValueError(
                "reward_function_kwargs is keyed by reward function names, but "
                f"entry '{bad_key}' is not a mapping (got "
                f"{type(bad_value).__name__}). Use "
                "reward_function_kwargs: {<func_name>: {arg: value}} for "
                "per-function kwargs, or a flat mapping to apply the same "
                "kwargs to all reward functions."
            )
        for name in function_names:
            reward_function = REGISTRY.get_reward_function(name)
            if not reward_function:
                raise KeyError(
                    f"reward_function `{name}` was not found in the registry."
                )
            function_kwargs: dict[str, Any] | None = None
            if config.reward_function_kwargs:
                function_kwargs = (
                    config.reward_function_kwargs.get(name, {})
                    if per_function_kwargs
                    else config.reward_function_kwargs
                )
            result.append(
                _apply_reward_function_kwargs(reward_function, function_kwargs)
            )

    return result
