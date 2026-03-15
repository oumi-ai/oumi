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

from __future__ import annotations

from pathlib import Path
from typing import Any

from oumi.core.agentic import AideOptimizer, BaseAgenticOptimizer
from oumi.core.configs.params.aide_params import AideParams


def build_agentic_optimizer(
    aide_params: AideParams,
    task_desc: dict[str, Any],
    workspace_dir: Path,
    base_training_config: str | None = None,
) -> BaseAgenticOptimizer:
    """Build an agentic optimizer based on the configuration.

    Currently only AIDE ML is supported. Future backends can be added
    by extending :class:`~oumi.core.agentic.BaseAgenticOptimizer`.

    Args:
        aide_params: Configuration parameters for the optimizer.
        task_desc: Task description dict for the optimization agent.
        workspace_dir: Working directory for generated scripts.
        base_training_config: Path to a base Oumi training config YAML.

    Returns:
        An instance of the appropriate optimizer implementation.
    """
    return AideOptimizer(
        aide_params=aide_params,
        task_desc=task_desc,
        workspace_dir=workspace_dir,
        base_training_config=base_training_config,
    )
