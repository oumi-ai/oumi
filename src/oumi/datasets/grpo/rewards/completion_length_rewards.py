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

import math
import re

from oumi.core.registry import RegistryType, register


def compute_soft_target_token_length_reward(num_tokens: int, *, target_tokens: int):
    """Returns maximum reward for inputs that are `target_tokens` long.

    The reward reduces smoothly if the actual number of tokens deviates
    from `target_tokens`.
    """
    x = float(num_tokens) / target_tokens
    return x * math.exp(-x)


def _compute_completion_target_token_length_reward(completions, *, target_tokens: int):
    return [
        compute_soft_target_token_length_reward(
            len(re.split(r"\s+", content)), target_tokens=target_tokens
        )
        for content in completions
    ]


@register("soft_5tokens_completions", RegistryType.REWARD_FUNCTION)
def _soft_5tokens_completions(completions, **kwargs):
    return _compute_completion_target_token_length_reward(completions, target_tokens=5)


@register("soft_10tokens_completions", RegistryType.REWARD_FUNCTION)
def _soft_10tokens_completions(completions, **kwargs):
    return _compute_completion_target_token_length_reward(completions, target_tokens=10)


@register("soft_20tokens_completions", RegistryType.REWARD_FUNCTION)
def _soft_20tokens_completions(completions, **kwargs):
    return _compute_completion_target_token_length_reward(completions, target_tokens=20)
