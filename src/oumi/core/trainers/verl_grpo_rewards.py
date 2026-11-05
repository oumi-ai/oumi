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

"""Weighted-sum reward for VERL_GRPO runs with several reward functions.

verl's ``custom_reward_function`` names a single function. TRL's GRPO trainer
takes a list of reward functions and sums them with ``reward_weights``;
:func:`combined_reward` gives VERL_GRPO the same behavior. The trainer points
verl at this module function and passes the function names, weights and
per-function kwargs through ``custom_reward_function.reward_kwargs``.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from oumi.core.registry import REGISTRY

# The key verl reads the reward from when a reward function returns a dict; the
# other keys are recorded per sample as ``reward_extra_info``.
VERL_REWARD_SCORE_KEY = "score"


def _to_plain(value: Any) -> Any:
    """Convert OmegaConf containers to plain Python so they can be unpacked."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ModuleNotFoundError:
        pass
    return value


def _score_from_output(name: str, output: Any) -> float:
    """Read a reward function's scalar score from a number or a ``score`` dict."""
    if isinstance(output, Mapping):
        if VERL_REWARD_SCORE_KEY not in output:
            raise ValueError(
                f"Reward function '{name}' returned a dict without a "
                f"'{VERL_REWARD_SCORE_KEY}' key: {sorted(output)}"
            )
        output = output[VERL_REWARD_SCORE_KEY]
    if isinstance(output, bool) or not isinstance(output, (int, float)):
        raise TypeError(
            f"Reward function '{name}' returned {type(output).__name__}; "
            "expected a number or a dict with a numeric 'score'."
        )
    return float(output)


def resolve_reward_functions(reward_function_names: Sequence[str]) -> list:
    """Look up registered reward functions by name, in order."""
    # Import to ensure the built-in GRPO reward functions are in the registry.
    import oumi.datasets.grpo.rewards  # noqa: F401

    functions = []
    for name in reward_function_names:
        function = REGISTRY.get_reward_function(name)
        if function is None:
            raise ValueError(
                f"Reward function '{name}' is not registered. Register it with "
                "@register(name, RegistryType.REWARD_FUNCTION) in a module listed "
                "in OUMI_EXTRA_DEPS_FILE."
            )
        functions.append(function)
    return functions


def combined_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
    *,
    reward_function_names: Sequence[str],
    reward_weights: Sequence[float] | None = None,
    reward_function_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Call every named reward function and return their weighted sum.

    Matches TRL's combination rule: ``sum(weight_i * score_i)`` with missing
    weights defaulting to 1 and a NaN score contributing nothing. Each
    function may return a number or a dict carrying a numeric ``score``.

    Returns:
        ``{"score": weighted_sum, <function name>: score, ...}`` so verl trains
        on the sum and records every function's own score.
    """
    names = [str(name) for name in _to_plain(reward_function_names)]
    weights = _to_plain(reward_weights)
    weights = [1.0] * len(names) if weights is None else [float(w) for w in weights]
    if len(weights) != len(names):
        raise ValueError(
            f"Number of reward_weights ({len(weights)}) must match the number "
            f"of reward functions ({len(names)})."
        )
    kwargs_map = _to_plain(reward_function_kwargs) or {}
    functions = resolve_reward_functions(names)

    total = 0.0
    scores: dict[str, float] = {}
    for name, function, weight in zip(names, functions, weights):
        function_kwargs = dict(_to_plain(kwargs_map.get(name)) or {})
        output = function(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
            **function_kwargs,
        )
        score = _score_from_output(name, output)
        scores[name] = score
        if not math.isnan(score):
            total += weight * score
    return {VERL_REWARD_SCORE_KEY: total, **scores}
