# Copyright 2026 - Oumi
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

"""verl custom reward function that scores rollouts through the oumi bridge.

Fails closed: any problem raises ``BridgeError`` and no default score is ever
returned.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Final

from oumi_verl_ext.bridge_client import (
    REWARD_PATH,
    BridgeError,
    post_json,
    to_json_value,
)

REWARD_TIMEOUT_S: Final = 600.0


def oumi_registry_reward(
    data_source: object,
    solution_str: str,
    ground_truth: object,
    extra_info: Mapping[str, object] | None = None,
    *,
    reward_functions: Sequence[str],
    reward_weights: Sequence[float] | None = None,
    reward_function_kwargs: Mapping[str, Mapping[str, object]] | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """Scores one rollout with oumi's registered reward functions.

    Returns:
        The bridge's ``scores``: the weighted ``score``, each function's own
        score, and any ``<function>/<key>`` extras.

    Raises:
        BridgeError: If the bridge is unreachable, answers with an error, or
            returns no finite numeric ``score``.
    """
    response = post_json(
        REWARD_PATH,
        {
            "data_source": to_json_value(data_source),
            "solution_str": solution_str,
            "ground_truth": to_json_value(ground_truth),
            "extra_info": to_json_value(extra_info),
            "reward_functions": to_json_value(reward_functions),
            "reward_weights": to_json_value(reward_weights),
            "reward_function_kwargs": to_json_value(reward_function_kwargs),
        },
        timeout_s=REWARD_TIMEOUT_S,
    )
    scores = response.get("scores")
    if not isinstance(scores, dict):
        raise BridgeError(f"Bridge response has no 'scores' object: {response}")
    score = scores.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
    ):
        raise BridgeError(f"Bridge returned a non-finite or missing score: {score!r}")
    return scores
