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

"""Reward for tasks whose answer is a single number."""

import re

from oumi.core.registry import RegistryType, register

_NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_number(text: str) -> float | None:
    """Returns the last number written in `text`, or None if there is none.

    Decimal points stay part of the number, so "$8.00" is 8 rather than 8 and 0.
    """
    matches = _NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def _completion_text(completion) -> str:
    """Returns the text of a completion, which TRL passes as str or a message list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, str):
            return last
        return str(last.get("content", ""))
    return str(completion)


@register("numeric_match", RegistryType.REWARD_FUNCTION)
def numeric_match_reward(completions, answer, **kwargs):
    """Rewards a completion that ends with the expected number.

    The last number in the completion is compared to `answer`, so prose around
    the answer is ignored.
    """
    expected = answer if isinstance(answer, list) else [answer] * len(completions)
    rewards = []
    for completion, want in zip(completions, expected):
        found = extract_number(_completion_text(completion))
        rewards.append(1.0 if found is not None and found == float(want) else 0.0)
    return rewards
