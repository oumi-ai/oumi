# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

"""Derived from https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/echo.py.

This file was slightly modified to be an Oumi reward registry function.
"""

from oumi.core.registry import RegistryType, register


@register("env_reward", RegistryType.REWARD_FUNCTION)
def reward_from_env(completions, **kwargs):
    """Reward function that uses the environment reward."""
    # Extract environment rewards from kwargs (propagated via extra_fields)
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return [float(reward) for reward in env_rewards]
    else:
        # Fallback if env_reward is not available
        return [0.0] * len(completions)
