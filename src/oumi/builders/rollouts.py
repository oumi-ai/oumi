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

from collections.abc import Callable

from oumi.core.configs import TrainingParams
from oumi.core.registry import REGISTRY
from oumi.core.types.exceptions import RegistryLookupError


def build_rollout_function(config: TrainingParams) -> Callable | None:
    """Builds the rollout function."""
    if config.grpo.rollout_function is None:
        return None
    # Import to ensure GRPO rollout functions are added to REGISTRY.
    import oumi.datasets.grpo.rollouts as grpo_rollouts  # noqa: F401

    rollout_function = REGISTRY.get_rollout_function(config.grpo.rollout_function)
    if not rollout_function:
        raise RegistryLookupError(
            f"rollout_function '{config.grpo.rollout_function}' not found in registry. "
            "Ensure the function is registered before use."
        )
    return rollout_function
