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

from oumi.core.registry import RegistryType, register


def _find_last_number(s: str) -> int:
    """Finds the last number (aka adjacent numeric digits) in a string."""
    return int(re.findall(r"\d+", s)[-1])


def compute_letter_count_reward(completion: str, target_count: int) -> int:
    """Counts the number of letters in a string."""
    try:
        count = _find_last_number(completion)
    except Exception:
        count = 0
    return -abs(count - target_count)


@register("count_letters", RegistryType.REWARD_FUNCTION)
def _count_letters(completions, letter_count, **kwargs):
    """Reward function for counting letters in a string."""
    return [
        compute_letter_count_reward(c, t) for c, t in zip(completions, letter_count)
    ]
