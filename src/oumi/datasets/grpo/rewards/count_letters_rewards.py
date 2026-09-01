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
from typing import Any

from oumi.core.registry import RegistryType, register


def _extract_prediction(response: str) -> int | None:
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

    Args:
        completion: The completion string from the LLM.
        target_count: The target count of letters.

    Returns:
        The reward value.
    """
    count = _extract_prediction(completion)

    # Lowest reward goes to unparseable responses
    if count is None:
        return -3.0

    delta = abs(count - target_count)

    # Reward scales from [0, -2) as delta increases
    # Ensures that "worse" answers (where the counts are off by a higher amount) are
    # penalized while never reaching -3.0 which is reserved for unparseable answers.
    return (1 / (delta + 0.5)) - 2


@register("count_letters_verl", RegistryType.REWARD_FUNCTION)
def count_letters_verl(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    r"""verl-style reward function for counting letters in a string.

    Unlike `count_letters`, which follows TRL's batched reward interface, this
    function follows verl's per-sample interface so it can be used with the
    VERL_GRPO trainer. The target letter count is read from the dataset's
    `reward_model.ground_truth` field (see `LetterCountGrpoDataset.transform`).

    Args:
        data_source: The dataset name for the sample (unused).
        solution_str: The model's decoded response.
        ground_truth: The target letter count, as a string.
        extra_info: Extra information about the sample (unused).

    Returns:
        The reward value: 0.0 for an exact count, decaying towards -2.0 as the
        predicted count gets further from the target, and -3.0 when the
        response contains no parseable ``\boxed{N}`` answer.
    """
    try:
        target_count = int(ground_truth)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Expected an integer-valued ground_truth, got {ground_truth!r}."
        ) from e
    return compute_letter_count_reward(solution_str, target_count)


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
