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

"""Prompt optimization module for Oumi."""

from oumi.core.prompt_optimization.base import BaseOptimizer, OptimizationResult
from oumi.core.prompt_optimization.checkpointing import (
    CheckpointManager,
    OptimizationCheckpoint,
    can_resume_from_checkpoint,
    print_checkpoint_summary,
)
from oumi.core.prompt_optimization.cost_tracking import (
    CostEstimate,
    estimate_optimization_cost,
    get_model_costs,
    should_warn_about_cost,
)
from oumi.core.prompt_optimization.dspy_optimizers import (
    BootstrapFewShotOptimizer,
    BootstrapFewShotWithOptunaOptimizer,
    GepaOptimizer,
    MiproOptimizer,
)
from oumi.core.prompt_optimization.evolutionary_optimizer import EvolutionaryOptimizer
from oumi.core.prompt_optimization.metrics import get_metric_fn
from oumi.core.prompt_optimization.validation import (
    ConfigValidationError,
    DatasetValidationError,
)

__all__ = [
    "BaseOptimizer",
    "OptimizationResult",
    "MiproOptimizer",
    "GepaOptimizer",
    "BootstrapFewShotOptimizer",
    "BootstrapFewShotWithOptunaOptimizer",
    "EvolutionaryOptimizer",
    "get_metric_fn",
    "DatasetValidationError",
    "ConfigValidationError",
    "CostEstimate",
    "estimate_optimization_cost",
    "get_model_costs",
    "should_warn_about_cost",
    "CheckpointManager",
    "OptimizationCheckpoint",
    "can_resume_from_checkpoint",
    "print_checkpoint_summary",
]
