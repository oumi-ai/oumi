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

from oumi.core.configs.params.evaluation_params import (
    EvaluationPlatform,
    EvaluationTaskParams,
)
from oumi.core.evaluators import BaseEvaluator, LmHarnessEvaluator


def evaluator_factory(task_params: EvaluationTaskParams) -> BaseEvaluator:
    """Factory method to create an evaluator based on the task parameters."""
    evaluation_platform: EvaluationPlatform = task_params.get_evaluation_platform()

    if evaluation_platform == EvaluationPlatform.LM_HARNESS:
        return LmHarnessEvaluator()
    elif evaluation_platform == EvaluationPlatform.ALPACA_EVAL:
        ################################################################################
        ############################# FIXME: ADD EVALUATOR #############################
        ################################################################################
        raise NotImplementedError("AlpacaEvalEvaluator not implemented")
    else:
        raise ValueError("Unknown evaluation platform")
