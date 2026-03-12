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

"""Core agentic optimization module for the Oumi library.

This module provides agentic code optimization implementations that use
LLM-powered tree search to iteratively generate, test, and refine code
solutions. Unlike traditional hyperparameter tuning (see :mod:`oumi.core.tuners`),
agentic optimizers operate in *code space* — modifying training configs,
reward functions, evaluation logic, and full pipelines.

Example:
    >>> from oumi.core.agentic import AideOptimizer  # doctest: +SKIP
    >>> optimizer = AideOptimizer(aide_params=params, task_desc=desc)  # doctest: +SKIP
    >>> optimizer.step(exec_callback=interpreter.run)  # doctest: +SKIP

Note:
    For detailed information on each optimizer, please refer to their respective
        class documentation.
"""  # noqa: E501

from oumi.core.agentic.aide_optimizer import AideOptimizer
from oumi.core.agentic.base_agentic_optimizer import BaseAgenticOptimizer

__all__ = [
    "AideOptimizer",
    "BaseAgenticOptimizer",
]
