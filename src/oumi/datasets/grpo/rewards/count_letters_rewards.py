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

import re
from typing import Any, Optional

from oumi.core.registry import RegistryType, register


def _extract_prediction(response: str) -> Optional[int]:
    r"""Returns the numeric answer extracted from `\boxed{...}`, or None otherwise."""
    regex_result = re.findall(r"\\boxed\{([-+]?\d+)\}", response)
    if not regex_result or len(regex_result) != 1:
        return None
    number_str = regex_result[0]
    # Except clause shouldn't trigger because the regex should only find ints.
    try:
        return int(number_str)
    except ValueError:
        return None


def compute_letter_count_reward(completion: str, target_count: int) -> float:
    """Computes the rewards for counting the letters in a string.

    The last group of consecutive digits in the completion is assumed to be the letter
    count. We're also assuming it's counting the correct letter. The reward is the
    negative of the absolute difference between the count and the target count, plus 0.1
    if the answer was properly formatted.

    For example, for the string "There are 2 'r's in strawberry", and the target count
    is 3, the reward is -1.

    Args:
        completion: The completion string from the LLM.
        target_count: The target count of letters.

    Returns:
        The reward value, calculated as the negative of the absolute difference between
        the count and the target count. The count is assumed to be the last group of
        consecutive digits in the completion string.
    """
    count = _extract_prediction(completion)
    formatting_reward = 0.1 if count is not None else 0
    if count is None:
        count = 0
    return -abs(count - target_count) + formatting_reward


@register("count_letters", RegistryType.REWARD_FUNCTION)
def _count_letters(
    completions: list[list[dict[str, Any]]],
    letter_count: list[int],
    **kwargs: dict[str, Any],
) -> list[float]:
    """Custom reward function for counting letters in a string.

    For more details on custom reward functions used in trl's GRPOTrainer, see:
    https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function.

    Args:
        completions: The list of completions from the LLM.
        letter_count: The list of target count of letters.
        kwargs: Unused.

    Returns:
        The reward values for each completion, calculated as the negative of the
        absolute difference between the count and the target count. The count is assumed
        to be the last group of consecutive digits in the completion string.
    """
    completions_strs = [c[0]["content"] for c in completions]
    return [
        compute_letter_count_reward(c, t)
        for c, t in zip(completions_strs, letter_count)
    ]
