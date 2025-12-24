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

from typing import Any, overload

from oumi.core.configs import EvaluationConfig, ModelParams
from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.evaluation.evaluation_result import EvaluationResult
from oumi.core.evaluation.evaluator import Evaluator
from oumi.utils.provider_detection import detect_provider, is_yaml_path


@overload
def evaluate(config: EvaluationConfig) -> list[dict[str, Any]]: ...


@overload
def evaluate(
    model: str,
    tasks: list[str] | None = None,
    *,
    num_samples: int | None = None,
    batch_size: int = 8,
    output_dir: str = "./eval_output",
    enable_wandb: bool = False,
) -> list[dict[str, Any]]: ...


def evaluate(
    config_or_model: EvaluationConfig | str,
    tasks: list[str] | None = None,
    *,
    num_samples: int | None = None,
    batch_size: int = 8,
    output_dir: str = "./eval_output",
    enable_wandb: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate a model using LM Harness or custom evaluation tasks.

    This function supports two modes:

    1. **Config mode**: Pass an EvaluationConfig object or YAML path
        >>> evaluate(EvaluationConfig.from_yaml("eval.yaml"))
        >>> evaluate("eval_config.yaml")

    2. **Simple mode**: Pass model name and task list
        >>> evaluate("meta-llama/Llama-3.1-8B", tasks=["mmlu", "hellaswag"])

    Args:
        config_or_model: Either an EvaluationConfig object, a YAML config path,
            or a model name (HuggingFace model ID or provider-prefixed name).
        tasks: List of evaluation task names (e.g., ["mmlu", "hellaswag"]).
            Only used in simple mode. See LM Harness for available tasks.
        num_samples: Number of samples per task (None = all samples).
            Only used in simple mode.
        batch_size: Batch size for evaluation. Only used in simple mode.
        output_dir: Output directory for evaluation results. Only used in simple mode.
        enable_wandb: Whether to enable W&B logging. Only used in simple mode.

    Returns:
        A list of evaluation results (one for each task). Each evaluation result is a
        dictionary of metric names and their corresponding values.

    Raises:
        ValueError: If tasks is not provided in simple mode.

    Examples:
        Simple evaluation with default settings:
            >>> results = evaluate("meta-llama/Llama-3.1-8B", tasks=["mmlu"])

        Evaluation with sampling:
            >>> results = evaluate(
            ...     "gpt-4o",
            ...     tasks=["mmlu", "hellaswag"],
            ...     num_samples=100,
            ... )

        From YAML config:
            >>> results = evaluate("eval_config.yaml")

        Full config object:
            >>> results = evaluate(EvaluationConfig(...))
    """
    # Handle EvaluationConfig directly
    if isinstance(config_or_model, EvaluationConfig):
        return _evaluate_impl(config_or_model)

    # Handle YAML path
    if is_yaml_path(config_or_model):
        config = EvaluationConfig.from_yaml(config_or_model)
        return _evaluate_impl(config)

    # Simple mode - model name and tasks
    model_name = config_or_model

    if not tasks:
        raise ValueError(
            "In simple mode, 'tasks' must be provided. "
            "Example: evaluate('model-name', tasks=['mmlu', 'hellaswag'])"
        )

    # Detect provider for inference engine
    engine_type, clean_model_name = detect_provider(model_name)

    # Build task params
    task_params = [
        EvaluationTaskParams(
            task_name=task,
            num_samples=num_samples,
        )
        for task in tasks
    ]

    config = EvaluationConfig(
        tasks=task_params,
        model=ModelParams(model_name=clean_model_name),
        inference_engine=engine_type,
        output_dir=output_dir,
        enable_wandb=enable_wandb,
    )

    return _evaluate_impl(config)


def _evaluate_impl(config: EvaluationConfig) -> list[dict[str, Any]]:
    """Internal implementation of model evaluation."""
    evaluator = Evaluator()
    results: list[EvaluationResult] = evaluator.evaluate(config)
    return [result.task_result for result in results]
