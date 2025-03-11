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

This module provides an evaluator for evaluating models with popular evaluation
libraries, such as `LM Harness` and `AlpacaEval`. It also allows users to define their
own custom evaluation function. The evaluator is designed to be modular and provide a
consistent interface for evaluating across different tasks.

Example:
    >>> from oumi.core.configs import EvaluationConfig
    >>> from oumi.core.evaluation import Evaluator
    >>> config = EvaluationConfig.from_yaml("evaluation_config.yaml") # doctest: +SKIP
    >>> evaluator = Evaluator() # doctest: +SKIP
    >>> result = evaluator.evaluate(evaluation_config) # doctest: +SKIP
"""

from oumi.core.evaluation.evaluation_result import EvaluationResult
from oumi.core.evaluation.evaluator import Evaluator

__all__ = [
    "Evaluator",
    "EvaluationResult",
]
