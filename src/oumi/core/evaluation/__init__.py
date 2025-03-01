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

"""Core evaluator module for the Oumi library.

This module provides various evaluator implementations for use in the Oumi framework.
These evaluators subclass the `BaseEvaluator` class and provide implementations for
evaluating models with popular evaluation libraries, such as `LM Harness` and
`AlpacaEval`. The evaluators are designed to be modular and provide a consistent
interface for evaluating across different tasks.

Example:
    >>> from oumi.core.evaluators import LmHarnessEvaluator
    >>> from oumi.core.configs import EvaluationConfig
    >>> config = EvaluationConfig.from_yaml("evaluation_config.yaml") # doctest: +SKIP
    >>> evaluator = LmHarnessEvaluator() # doctest: +SKIP
    >>> evaluator.evaluate(config) # doctest: +SKIP

Note:
    For detailed information on each evaluator, please refer to their respective
        class documentation at oumi.core.evaluators.<name>_evaluator.
"""

from oumi.core.evaluation.evaluation_result import EvaluationResult
from oumi.core.evaluation.evaluator import Evaluator

__all__ = [
    "Evaluator",
    "EvaluationResult",
]
