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

from typing import Callable, Optional

from oumi.core.configs import TrainingParams
from oumi.core.registry import REGISTRY


def build_metrics_function(config: TrainingParams) -> Optional[Callable]:
    """Builds the metrics function."""
    metrics_function = None
    if config.metrics_function:
        metrics_function = REGISTRY.get_metrics_function(config.metrics_function)
        if not metrics_function:
            raise KeyError(
                f"metrics_function `{config.metrics_function}` "
                "was not found in the registry."
            )

    return metrics_function
