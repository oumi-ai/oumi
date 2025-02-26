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

from oumi.core.configs import (
    EvaluationConfig,
    EvaluationTaskParams,
    LMHarnessTaskParams,
)
from oumi.core.configs.params.evaluation_params import EvaluationPlatform
from oumi.core.evaluators.base_evaluator import BaseEvaluator
from oumi.core.evaluators.evaluation_result import EvaluationResult
from oumi.evaluation.lm_harness import evaluate as evaluate_lm_harness
from oumi.evaluation.platform_prerequisites import check_prerequisites


class LmHarnessEvaluator(BaseEvaluator):
    """Evaluator for the LM Harness evaluation backend."""

    def _check_task_prerequisites(self, task_params: EvaluationTaskParams) -> None:
        """Checks whether the task prerequisites are satisfied."""
        check_prerequisites(
            evaluation_platform=EvaluationPlatform.LM_HARNESS,
            task_name=task_params.task_name,
        )

    def _evaluate_task(
        self,
        task_params: EvaluationTaskParams,
        config: EvaluationConfig,
        # Arguments specific to the LM Harness evaluator.
        random_seed: int = 0,
        numpy_random_seed: int = 1234,
        torch_random_seed: int = 1234,
    ) -> EvaluationResult:
        """Evaluates on the provided task, using the configuration."""
        lm_harness_task_params = task_params.get_evaluation_platform_task_params()
        assert isinstance(lm_harness_task_params, LMHarnessTaskParams)

        ################################################################################
        ############################ FIXME: TEMPORARY HACK #############################
        ################################################################################
        # This is a temporary hack and will be fixed in the next commit (before the PR
        # is merged). The file `oumi/evaluation/lm_harness.py` will be deleted and the
        # `evaluate_lm_harness()` function will become part class.
        lm_harness_result, lm_harness_task_config = evaluate_lm_harness(
            task_params=lm_harness_task_params,
            output_dir="",  # FIXME: config.output_dir
            model_params=config.model,
            generation_params=config.generation,
            enable_wandb=config.enable_wandb,
            inference_engine_type=config.inference_engine,
            inference_remote_params=config.inference_remote_params,
            run_name=config.run_name,
            random_seed=random_seed,
            numpy_random_seed=numpy_random_seed,
            torch_random_seed=torch_random_seed,
        )
        ################################################################################
        ################################################################################
        ################################################################################

        return EvaluationResult(
            task_name=task_params.task_name,
            task_result=lm_harness_result,
            backend_config=lm_harness_task_config,
        )
