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

from typing import Any

from oumi.core.configs import EvaluationConfig
from oumi.core.configs.params.evaluation_params import (
    AlpacaEvalTaskParams,
    EvaluationPlatform,
    LMHarnessTaskParams,
)
from oumi.evaluation.alpaca_eval import evaluate as evaluate_alpaca_eval
from oumi.evaluation.lm_harness import evaluate as evaluate_lm_harness
from oumi.evaluation.platform_prerequisites import check_prerequisites


def evaluate(config: EvaluationConfig) -> list[dict[str, Any]]:
    """Evaluates a model using the provided configuration.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        A list of evaluation results (one for each task). Each evaluation result is a
        dictionary of metric names and their corresponding values.
    """
    results = []
    for task in config.tasks:
        check_prerequisites(
            evaluation_platform=task.get_evaluation_platform(),
            task_name=task.task_name,
        )

        if task.get_evaluation_platform() == EvaluationPlatform.LM_HARNESS:
            lm_harness_task_params = task.get_evaluation_platform_task_params()
            assert isinstance(lm_harness_task_params, LMHarnessTaskParams)
            result = evaluate_lm_harness(
                task_params=lm_harness_task_params,
                output_dir=config.output_dir,
                model_params=config.model,
                generation_params=config.generation,
                enable_wandb=config.enable_wandb,
                run_name=config.run_name,
            )
            results.append(result)
        elif task.get_evaluation_platform() == EvaluationPlatform.ALPACA_EVAL:
            alpaca_eval_task_params = task.get_evaluation_platform_task_params()
            assert isinstance(alpaca_eval_task_params, AlpacaEvalTaskParams)
            if not config.inference_engine:
                raise ValueError(
                    "Inference engine must be specified for Alpaca Eval evaluation."
                )
            result = evaluate_alpaca_eval(
                task_params=alpaca_eval_task_params,
                output_dir=config.output_dir,
                model_params=config.model,
                generation_params=config.generation,
                inference_engine_type=config.inference_engine,
                inference_remote_params=config.inference_remote_params,
                run_name=config.run_name,
            )
            results.append(result)
        else:
            raise ValueError("Unknown evaluation platform")
    return results
