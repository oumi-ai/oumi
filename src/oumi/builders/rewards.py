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

from typing import Callable

from oumi.core.configs import TrainingParams
from oumi.core.registry import REGISTRY


def build_reward_functions(config: TrainingParams) -> list[Callable]:
    """Builds the metrics function."""
    result: list[Callable] = []
    if config.reward_functions is not None:
        reward_function_names: list[str] = (
            config.reward_functions
            if isinstance(config.reward_functions, list)
            else [config.reward_functions]
        )
        for name in reward_function_names:
            reward_function = REGISTRY.get_reward_function(name)
            if not reward_function:
                raise KeyError(
                    f"reward_function `{name}` was not found in the registry."
                )
            result.append(reward_function)

    return result
