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

import copy
import inspect
import time
from dataclasses import fields
from datetime import datetime
from typing import Any, Callable, Optional, Union

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs import (
    AlpacaEvalTaskParams,
    EvaluationConfig,
    EvaluationTaskParams,
    LMHarnessTaskParams,
)
from oumi.core.configs.params.evaluation_params import EvaluationBackend
from oumi.core.distributed import is_world_process_zero
from oumi.core.evaluation.backends.alpaca_eval import evaluate as evaluate_alpaca_eval
from oumi.core.evaluation.backends.lm_harness import evaluate as evaluate_lm_harness
from oumi.core.evaluation.evaluation_result import EvaluationResult
from oumi.core.evaluation.utils.platform_prerequisites import check_prerequisites
from oumi.core.evaluation.utils.save_utils import save_evaluation_output
from oumi.core.inference import BaseInferenceEngine
from oumi.core.registry import REGISTRY

_EVALUATION_FN_INFERENCE_ENGINE_INPUT_PARAM_NAME = "inference_engine"
_EVALUATION_FN_TASK_PARAMS_INPUT_PARAM_NAME = "task_params"
_EVALUATION_FN_CONFIG_INPUT_PARAM_NAME = "config"

# Reserved keys that a custom evaluation function might define as inputs. The values of
# these keys, if defined as inputs, will be automatically populated by the Evaluator.
# The user is NOT allowed to pass these as keyword arguments when calling the
# `Evaluator.evaluate()` function.
RESERVED_KEYS = {
    _EVALUATION_FN_INFERENCE_ENGINE_INPUT_PARAM_NAME,
    _EVALUATION_FN_TASK_PARAMS_INPUT_PARAM_NAME,
    _EVALUATION_FN_CONFIG_INPUT_PARAM_NAME,
}


class Evaluator:
    """A class for evaluating language models on various tasks.

    Currently, the evaluator supports a wide range of tasks that are handled by three
    separate backends: LM Harness, Alpaca Eval, and Custom.

    - LM Harness: Framework by EleutherAI for evaluating language models (mostly) on
        standardized benchmarks (multiple-choice, word match, etc). The backend supports
        a large number of popular benchmarks, which can be found at:
        https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks.
    - Alpaca Eval: Framework for evaluating the instruction-following capabilities of
        language models, as well as whether their responses are helpful, accurate, and
        relevant. The instruction set consists of 805 open-ended questions, while the
        evaluation is based on "LLM-as-judge" and prioritizes human-alignment, aiming
        to assess whether the model responses meet the expectations of human evaluators.
    - Custom: Users can register their own evaluation functions using the decorator
        `@register_evaluation_function` and run custom evaluations based on their
        functions. Note that the `task_name` should be the registry key for the custom
        evaluation function to be used.
    """

    _inference_engine: Optional[BaseInferenceEngine] = None
    """Inference engine used for evaluation, if needed by the tasks."""

    def evaluate(self, config: EvaluationConfig, **kwargs) -> list[EvaluationResult]:
        """Evaluates a model using the provided evaluation configuration.

        Args:
            config: The desired configuration for evaluation.
            kwargs: Additional keyword arguments required by evaluator backends.

        Returns:
            List of evaluation results (one per task, in the same order with `tasks`).
        """
        # Create a copy of the evaluation config, without tasks, so that there is no
        # redundant information in the `config` input parameter of `self.evaluate_task`.
        config_without_tasks = copy.deepcopy(config)
        config_without_tasks.tasks = []

        # Evaluate on each task included in the configuration, serially.
        evaluation_results = []
        for task in config.tasks:
            evaluation_result = self.evaluate_task(
                task_params=task, config=config_without_tasks, **kwargs
            )
            evaluation_results.append(evaluation_result)

        return evaluation_results

    def evaluate_task(
        self,
        task_params: EvaluationTaskParams,
        config: EvaluationConfig,
        **kwargs,
    ) -> EvaluationResult:
        """Evaluates a model using the provided configuration on a specific task.

        Args:
            task_params: The task parameters for evaluation.
            config: The desired evaluation configuration for evaluation.
            kwargs: Additional keyword arguments required by evaluator backends.

        Returns:
            The results for evaluating on the task.
        """
        # Find the proper backend to execute the evaluation task.
        evaluation_backend: EvaluationBackend = task_params.get_evaluation_backend()

        # Ensure the task prerequisites are satisfied; fast-fail if not.
        check_prerequisites(
            evaluation_backend=evaluation_backend,
            task_name=task_params.task_name,
        )

        # Get a timestamp at the beginning of the current run.
        start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()

        # Redirect the evaluation execution to the appropriate evaluation backend.
        if evaluation_backend == EvaluationBackend.LM_HARNESS:
            lm_harness_task_params = self._get_backend_task_params(task_params)
            assert isinstance(lm_harness_task_params, LMHarnessTaskParams)

            # Destroy the inference engine, if created by a previous task. LM Harness
            # uses its own inference engine, which is created internally.
            if self._inference_engine:
                del self._inference_engine
                self._inference_engine = None

            evaluation_result = evaluate_lm_harness(
                task_params=lm_harness_task_params,
                config=config,
                **kwargs,  # random_seed, numpy_random_seed, torch_random_seed
            )
        elif evaluation_backend == EvaluationBackend.ALPACA_EVAL:
            alpaca_eval_task_params = self._get_backend_task_params(task_params)
            assert isinstance(alpaca_eval_task_params, AlpacaEvalTaskParams)

            evaluation_result = evaluate_alpaca_eval(
                task_params=alpaca_eval_task_params,
                config=config,
                inference_engine=self._get_inference_engine(config),
                **kwargs,
            )
        elif evaluation_backend == EvaluationBackend.CUSTOM:
            evaluation_fn_name = task_params.task_name or ""
            evaluation_fn = self._get_custom_evaluation_fn(evaluation_fn_name)
            custom_kwargs = self._merge_kwargs(kwargs, task_params.eval_kwargs)
            self._validate_custom_kwargs(
                custom_kwargs=custom_kwargs,
                evaluation_fn=evaluation_fn,
                evaluation_fn_name=evaluation_fn_name,
            )
            self._add_reserved_keys_into_custom_kwargs(
                custom_kwargs=custom_kwargs,
                evaluation_fn=evaluation_fn,
                task_params=task_params,
                config=config,
            )
            evaluation_output = evaluation_fn(**custom_kwargs)

            if isinstance(evaluation_output, EvaluationResult):
                evaluation_result = evaluation_output
            elif isinstance(evaluation_output, dict):
                evaluation_result = EvaluationResult(
                    task_name=task_params.task_name,
                    task_result={"results": {task_params.task_name: evaluation_output}},
                )
            else:
                raise ValueError(
                    f"The custom evaluation function `{task_params.task_name}` must "
                    "return either a `dict` or an `EvaluationResult` object, but it is "
                    f"currently returning an object of type `{type(evaluation_output)}`"
                    ". Please ensure that the function returns the correct object."
                )
        else:
            raise ValueError(f"Unknown evaluation backend: {evaluation_backend}")

        # Calculate the elapsed time for the evaluation run.
        evaluation_result.elapsed_time_sec = int(time.time() - start_time)
        evaluation_result.start_time = start_time_str

        # Save the output, if an output directory has been provided.
        if config.output_dir and is_world_process_zero():
            self.save_output(
                task_params=task_params,
                evaluation_result=evaluation_result,
                base_output_dir=config.output_dir,
                config=config,
            )
        return evaluation_result

    def save_output(
        self,
        task_params: EvaluationTaskParams,
        evaluation_result: EvaluationResult,
        base_output_dir: str,
        config: Optional[EvaluationConfig],
    ) -> None:
        """Saves the evaluation's output to the specified output directory.

        Args:
            task_params: The task parameters used for this evaluation.
            evaluation_result: The evaluation result.
            base_output_dir: The directory where the evaluation results will be saved.
            config: The evaluation configuration.

        Returns:
            None
        """
        save_evaluation_output(
            backend_name=task_params.evaluation_backend,
            task_params=task_params,
            evaluation_result=evaluation_result,
            base_output_dir=base_output_dir,
            config=config,
        )

    @staticmethod
    def _get_custom_evaluation_fn(task_name: Optional[str]) -> Callable:
        """Retrieve the evaluation function of the custom task."""
        if not task_name:
            raise ValueError(
                "Missing `task_name` for custom Oumi evaluation. Please specify the "
                "task name, which should be corresponding to a registered evaluation "
                "function, using the decorator `@register_evaluation_function`."
            )
        # Import to ensure custom evaluation functions are added to REGISTRY.
        import oumi.evaluation.registry as evaluation_registry  # noqa: F401

        if evaluation_fn := REGISTRY.get_evaluation_function(task_name):
            return evaluation_fn
        else:
            raise ValueError(
                f"Task name `{task_name}` not found in the registry. For custom Oumi "
                "evaluations, the task name must match the name of a registered "
                "evaluation function. You can register a new function with the "
                "decorator `@register_evaluation_function`."
            )

    @staticmethod
    def _get_backend_task_params(
        task_params: EvaluationTaskParams,
    ) -> Union[LMHarnessTaskParams, AlpacaEvalTaskParams]:
        """Returns the evaluation backend-specific task parameters."""
        if task_params.get_evaluation_backend() == EvaluationBackend.LM_HARNESS:
            target_class = LMHarnessTaskParams
        elif task_params.get_evaluation_backend() == EvaluationBackend.ALPACA_EVAL:
            target_class = AlpacaEvalTaskParams
        elif task_params.get_evaluation_backend() == EvaluationBackend.CUSTOM:
            raise ValueError(
                "The custom evaluation backend is not subclassing EvaluationTaskParams."
                " Thus, `Evaluator._get_backend_task_params()` should not be called "
                " when evaluation_backend is set to `EvaluationBackend.CUSTOM`."
            )
        else:
            raise ValueError(f"Unknown backend: {task_params.evaluation_backend}")

        init_kwargs = Evaluator._get_init_kwargs_for_task_params_class(
            task_params=task_params, target_class=target_class
        )
        return target_class(**init_kwargs)

    @staticmethod
    def _get_init_kwargs_for_task_params_class(
        task_params: EvaluationTaskParams,
        target_class: type[EvaluationTaskParams],
    ) -> dict[str, Any]:
        """Returns the init keyword arguments for a `target_class` of name *TaskParams.

        Given a target class of name <evaluation backend>TaskParams, which subclasses
        `EvaluationTaskParams`, this method returns a 'flattened' dict with all
        arguments needed to instantiate it. The dict includes all the parameters which
        are already members of `EvaluationTaskParams`, as well as additional parameters
        which are only known to the target class (stored under `eval_kwargs`).
        By 'flattened', we mean that all known parameters that are stored under the
        `eval_kwargs` dict are moved one level up, to the (flat) dict that is returned.
        In contrast, all unknown (to the target class) parameters remain (unflattened)
        inside the `eval_kwargs` dict.

        Example:
            Assuming these are the input parameters:
            task_params: EvaluationTaskParams(       # <- `num_fewshot` is NOT a member
                evaluation_backend=EvaluationBackend.LM_HARNESS,
                task_name="mmlu",
                eval_kwargs={"num_fewshot": 10, "some_param": 20},
            )
            target_class: LMHarnessTaskParams        # <- `num_fewshot` is a member

            This function will return:
            {
                "evaluation_backend": EvaluationBackend.LM_HARNESS,
                "task_name": "mmlu",
                "num_fewshot": 10,
                "eval_kwargs": {"some_param": 20}
            }
        """
        task_params = copy.deepcopy(task_params)

        # Find all keys in `eval_kwargs` which are known to the target class.
        known_keys = []
        if task_params.eval_kwargs:
            field_names = [field.name for field in fields(target_class)]
            known_keys.extend(k for k in task_params.eval_kwargs if k in field_names)

        # Identify all kwargs known to the current class.
        init_keys = [
            key
            for key in dir(task_params)
            if not callable(getattr(task_params, key)) and not key.startswith("_")
        ]
        init_kwargs = {key: getattr(task_params, key) for key in init_keys}

        # Move known kwargs one level up: from `eval_kwargs` to the top-level dict.
        for key in known_keys:
            if key in init_kwargs:
                raise ValueError(
                    f"Parameter `{key}` is present twice, in both task parameters and "
                    "`eval_kwargs` dictionary. Please remove it from one of them."
                )
            init_kwargs[key] = init_kwargs["eval_kwargs"].pop(key)

        return init_kwargs

    @staticmethod
    def _merge_kwargs(
        kwargs_1: dict[str, Any],
        kwargs_2: dict[str, Any],
    ) -> dict[str, Any]:
        """Merges two keyword argument dictionaries."""
        if overlapping_keys := kwargs_1.keys() & kwargs_2.keys():
            raise ValueError(
                "The two keyword argument dictionaries contain overlapping keys: "
                f"{overlapping_keys}. Please ensure that the keys in the following "
                f"dictionaries are unique: `{kwargs_1.keys()}` and `{kwargs_2.keys()}`"
            )
        return kwargs_1 | kwargs_2

    @staticmethod
    def _validate_custom_kwargs(
        custom_kwargs: dict[str, Any],
        evaluation_fn: Callable,
        evaluation_fn_name: str,
    ) -> None:
        """Validates the keyword arguments of the custom evaluation function."""
        # Ensure that user-provided keyword arguments, which are passed into method
        # `Evaluator.evaluate`, do NOT contain any reserved keys.
        if reserved_keys_used := RESERVED_KEYS & custom_kwargs.keys():
            raise RuntimeError(
                "Reserved keys are present when calling `Evaluator.evaluate()`. "
                "You are not allowed to pass the following keyword arguments into "
                f"the `{evaluation_fn_name}` function: {sorted(RESERVED_KEYS)}. "
                "However, you have passed the following reserved keys: "
                f"{sorted(reserved_keys_used)}. These keys can (optionally) be inputs "
                f"of your registered evaluation function `{evaluation_fn_name}`. "
                "If you choose to use them, they will be automatically populated "
                "by the Evaluator. "
                f"The `{_EVALUATION_FN_INFERENCE_ENGINE_INPUT_PARAM_NAME}` input "
                "will provide you with an inference engine that is generated "
                "according to the `EvaluationConfig.inference_engine` type that "
                "you have specified in the evaluation config. "
                f"Then, `{_EVALUATION_FN_TASK_PARAMS_INPUT_PARAM_NAME}`, "
                f"`{_EVALUATION_FN_TASK_PARAMS_INPUT_PARAM_NAME}` will provide you "
                "with the task parameters and the evaluation configuration, "
                "respectively."
            )

        # Ensure that user-provided keyword arguments, which are passed into method
        # `Evaluator.evaluate`, match the expected input parameters of the custom
        # evaluation function `evaluation_fn`.
        fn_signature = inspect.signature(evaluation_fn)
        fn_input_params = [param.name for param in fn_signature.parameters.values()]

        provided_keys: set[str] = custom_kwargs.keys() - set(RESERVED_KEYS)
        expected_keys: set[str] = set(fn_input_params) - set(RESERVED_KEYS)

        if unrecognized_keys := provided_keys - expected_keys:
            raise RuntimeError(
                "Unrecognized keyword arguments are present when calling "
                "`Evaluator.evaluate()`. You have passed the following unrecognized "
                f"keys: {sorted(unrecognized_keys)}. Please ensure that the provided "
                "keys match the expected input parameters of the custom evaluation "
                f"function `{evaluation_fn_name}`. The expected input parameters "
                f"of the function are: {fn_input_params}."
            )
        elif missing_keys := expected_keys - provided_keys:
            raise RuntimeError(
                "Missing keyword arguments have been identified when calling "
                "`Evaluator.evaluate()`. You have not passed the following expected "
                f"keys: {missing_keys}. Please ensure that the provided keys match "
                "the expected input parameters of the custom evaluation function "
                f"`{evaluation_fn_name}`. The expected input parameters of the "
                f"function are: {fn_input_params}."
            )

    def _add_reserved_keys_into_custom_kwargs(
        self,
        custom_kwargs: dict[str, Any],
        evaluation_fn: Callable,
        task_params: EvaluationTaskParams,
        config: EvaluationConfig,
    ) -> None:
        """Adds reserved keys into the keyword arguments, if needed.

        Reserved keys are keys that, if defined in the custom evaluation function
        (`evaluation_fn`), are automatically populated by the Evaluator. This function
        is responsible to add them into the keyword arguments.
        """
        fn_signature = inspect.signature(evaluation_fn)
        fn_input_params = [param.name for param in fn_signature.parameters.values()]

        if _EVALUATION_FN_TASK_PARAMS_INPUT_PARAM_NAME in fn_input_params:
            custom_kwargs[_EVALUATION_FN_TASK_PARAMS_INPUT_PARAM_NAME] = task_params
        if _EVALUATION_FN_CONFIG_INPUT_PARAM_NAME in fn_input_params:
            custom_kwargs[_EVALUATION_FN_CONFIG_INPUT_PARAM_NAME] = config
        if _EVALUATION_FN_INFERENCE_ENGINE_INPUT_PARAM_NAME in fn_input_params:
            custom_kwargs[_EVALUATION_FN_INFERENCE_ENGINE_INPUT_PARAM_NAME] = (
                self._get_inference_engine(config)
            )

    def _add_inference_engine_if_needed(
        self,
        evaluation_function: Callable,
        kwargs: dict[str, Any],
        config: EvaluationConfig,
    ) -> None:
        """Adds an inference engine to the keyword arguments (`kwargs`), if needed."""
        # Check if the evaluation function requires an inference engine.
        fn_signature = inspect.signature(evaluation_function)
        fn_input_params = [param.name for param in fn_signature.parameters.values()]
        if _EVALUATION_FN_INFERENCE_ENGINE_INPUT_PARAM_NAME not in fn_input_params:
            return

        # Ensure an inference engine is not already provided in the keyword arguments.
        if kwargs.get(_EVALUATION_FN_INFERENCE_ENGINE_INPUT_PARAM_NAME):
            raise RuntimeError(
                "The inference engine is already provided in the keyword arguments. "
                f"The input param `{_EVALUATION_FN_INFERENCE_ENGINE_INPUT_PARAM_NAME}` "
                "is reserved for an inference engine that is generated according to "
                "the evaluation config's `EvaluationConfig.inference_engine` field and "
                "should not be populated by users."
            )

        # Add inference engine in kwargs.
        kwargs[_EVALUATION_FN_INFERENCE_ENGINE_INPUT_PARAM_NAME] = (
            self._get_inference_engine(config)
        )

    def _get_inference_engine(self, config: EvaluationConfig) -> BaseInferenceEngine:
        """Returns the inference engine based on the evaluation configuration."""
        if not self._inference_engine:
            self._inference_engine = build_inference_engine(
                engine_type=config.inference_engine,
                model_params=config.model,
                remote_params=config.inference_remote_params,
                generation_params=config.generation,
            )
        return self._inference_engine
