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


def _whitespace_tokenize(s: str) -> list[str]:
    return re.split(r"\s+", s)


def compute_soft_target_token_length_reward(num_tokens: int, *, target_tokens: int):
    """Returns maximum reward for inputs that are `target_tokens` long.

    The reward is in the [0,1] range and reduces smoothly from the maximum value of 1.0
    if the actual number of tokens deviates from `target_tokens`.

    The reward is proportional to: `x*exp(-x)` where `x := num_tokens/target_tokens`.
    """
    x = num_tokens / target_tokens
    return x * math.exp(-x) * math.e


def _compute_completion_soft_target_token_length_reward(
    completions: list[str], *, target_tokens: int
):
    return [
        compute_soft_target_token_length_reward(
            len(_whitespace_tokenize(content)), target_tokens=target_tokens
        )
        for content in completions
    ]


def compute_sharp_target_token_length_reward(num_tokens: int, *, target_tokens: int):
    """Returns maximum reward for inputs that are `target_tokens` long.

    The reward reduces sharply if the actual number of tokens deviates
    from `target_tokens`.

    The reward is computed as: `-|num_tokens - target_tokens|`, which penalizes
    token counts not equal to `target_tokens`.
    """
    return -abs(num_tokens - target_tokens)


def _compute_completion_sharp_target_token_length_reward(
    completions: list[str], *, target_tokens: int
):
    return [
        compute_sharp_target_token_length_reward(
            len(_whitespace_tokenize(content)), target_tokens=target_tokens
        )
        for content in completions
    ]


# Simple toy length-based reward functions for experimentation and demonstration
# purposes. In practice, most users are expected to define and  use custom reward
# functions, not these.
# For more details on custom reward functions used in trl's GRPOTrainer, see:
# https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function.


@register("soft_5tokens_completions", RegistryType.REWARD_FUNCTION)
def _soft_5tokens_completions(completions: list[str], **kwargs):
    return _compute_completion_soft_target_token_length_reward(
        completions, target_tokens=5
    )


@register("soft_10tokens_completions", RegistryType.REWARD_FUNCTION)
def _soft_10tokens_completions(completions: list[str], **kwargs):
    return _compute_completion_soft_target_token_length_reward(
        completions, target_tokens=10
    )


@register("soft_20tokens_completions", RegistryType.REWARD_FUNCTION)
def _soft_20tokens_completions(completions: list[str], **kwargs):
    return _compute_completion_soft_target_token_length_reward(
        completions, target_tokens=20
    )


@register("sharp_5tokens_completions", RegistryType.REWARD_FUNCTION)
def _sharp_5tokens_completions(completions: list[str], **kwargs):
    return _compute_completion_sharp_target_token_length_reward(
        completions, target_tokens=5
    )


@register("sharp_10tokens_completions", RegistryType.REWARD_FUNCTION)
def _sharp_10tokens_completions(completions: list[str], **kwargs):
    return _compute_completion_sharp_target_token_length_reward(
        completions, target_tokens=10
    )


@register("sharp_20tokens_completions", RegistryType.REWARD_FUNCTION)
def _sharp_20tokens_completions(completions: list[str], **kwargs):
    return _compute_completion_sharp_target_token_length_reward(
        completions, target_tokens=20
    )
