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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from oumi.core.configs.params.base_params import BaseParams
from oumi.utils.logging import logger


class EvaluationBackend(Enum):
    """Enum representing the evaluation backend to use."""

    LM_HARNESS = "lm_harness"
    ALPACA_EVAL = "alpaca_eval"
    CUSTOM = "custom"


@dataclass
class EvaluationTaskParams(BaseParams):
    """Configuration parameters for model evaluation tasks.

    Supported backends:

    - LM Harness: Framework for evaluating language models on standard benchmarks.
      A list of all supported tasks can be found at:
      https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks.
    - Alpaca Eval: Framework for evaluating language models on instruction-following
      and quality of responses on open-ended questions.
    - Custom: Users can register their own evaluation functions using the decorator
      `@register_evaluation_function`. The task_name should be the registry key for
      the custom evaluation function to be used.

    Examples:
        .. code-block:: python

            # LM Harness evaluation on MMLU
            params = EvaluationTaskParams(
                evaluation_backend="lm_harness",
                task_name="mmlu",
                eval_kwargs={"num_fewshot": 5}
            )


        .. code-block:: python

            # Alpaca Eval 2.0 evaluation
            params = EvaluationTaskParams(
                evaluation_backend="alpaca_eval"
            )


        .. code-block:: python

            # Custom evaluation
            @register_evaluation_function("my_evaluation_function")
            def my_evaluation(task_params, config):
                accuracy = ...
                return EvaluationResult(task_result={"accuracy": accuracy})

            params = EvaluationTaskParams(
                task_name="my_evaluation_function",
                evaluation_backend="custom"
            )
    """

    evaluation_backend: str = ""
    """The evaluation backend to use for the current task."""

    task_name: Optional[str] = None
    """The task to evaluate or the custom evaluation function to use.

    For LM Harness evaluations (when the evaluation_backend is set to
    EvaluationBackend.LM_HARNESS), the `task_name` corresponds to a predefined task
    to evaluate on (e.g. "mmlu"). A list of all supported tasks by the LM Harness
    backend can be found by running: `lm-eval --tasks list`.

    For custom evaluations (when evaluation_backend is set to EvaluationBackend.CUSTOM),
    the `task_name` should be the registry key for the custom evaluation function to be
    used. Users can register new evaluation functions using the decorator
    `@register_evaluation_function`.
    """

    num_samples: Optional[int] = None
    """Number of samples/examples to evaluate from this dataset.

    Mostly for debugging, in order to reduce the runtime.
    If not set (None): the entire dataset is evaluated.
    If set, this must be a positive integer.
    """

    log_samples: Optional[bool] = False
    """Whether to log the samples used for evaluation.

    If not set (False): the model samples used for evaluation will not be logged.
    If set to True: the model samples generated during inference and used for
    evaluation will be logged in `backend_config.json`. The backend may also log
    other intermediate results related to inference.
    """

    eval_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to the evaluation function.

    This allows for passing any evaluation-specific parameters that are not
    covered by other fields in TaskParams classes.
    """

    evaluation_platform: Optional[str] = ""
    """DEPRECATED; Please use `evaluation_backend` instead."""

    def get_evaluation_backend(self) -> EvaluationBackend:
        """Returns the evaluation backend as an Enum."""
        if not self.evaluation_backend:
            raise ValueError(
                "Missing `evaluation_backend`. When running evaluations, it is "
                "necessary to specify the evaluation backend to use for EACH task. "
                "The available backends can be found in the following enum: "
                "`oumi.core.configs.params.evaluation_params.EvaluationBackend`. "
                f"Current options: {EvaluationTaskParams.list_evaluation_backends()}."
            )
        elif self.evaluation_backend == EvaluationBackend.LM_HARNESS.value:
            return EvaluationBackend.LM_HARNESS
        elif self.evaluation_backend == EvaluationBackend.ALPACA_EVAL.value:
            return EvaluationBackend.ALPACA_EVAL
        elif self.evaluation_backend == EvaluationBackend.CUSTOM.value:
            return EvaluationBackend.CUSTOM
        else:
            raise ValueError(f"Unknown evaluation backend: {self.evaluation_backend}")

    @staticmethod
    def list_evaluation_backends() -> str:
        """Returns a string listing all available evaluation backends."""
        return ", ".join([backend.value for backend in EvaluationBackend])

    def __post_init__(self):
        """Verifies params."""
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError("`num_samples` must be None or a positive integer.")

        # Handle deprecated evaluation_platform parameter.
        if not self.evaluation_platform and not self.evaluation_backend:
            raise ValueError("`evaluation_backend` must be set!")
        if (
            self.evaluation_platform
            and self.evaluation_backend
            and self.evaluation_platform != self.evaluation_backend
        ):
            raise ValueError(
                "Conflicting values for `evaluation_platform` and `evaluation_backend`!"
            )
        if self.evaluation_platform:
            logger.warning(
                "The input parameter `evaluation_platform` is deprecated and will be "
                "removed at v0.2.0. Please use `evaluation_backend` instead."
            )
            self.evaluation_backend = self.evaluation_platform
            self.evaluation_platform = ""


@dataclass
class LMHarnessTaskParams(EvaluationTaskParams):
    """Parameters for the LM Harness evaluation framework.

    LM Harness is a comprehensive benchmarking suite for evaluating language models
    across various tasks.
    """

    num_fewshot: Optional[int] = None
    """Number of few-shot examples (with responses) to add in the prompt, in order to
    teach the model how to respond to the specific dataset's prompts.

    If not set (None): LM Harness will decide the value.
    If set to 0: no few-shot examples will be added in the prompt.
    """

    def __post_init__(self):
        """Verifies params."""
        if not self.task_name:
            raise ValueError("`task_name` must be a valid LM Harness task.")
        if self.num_fewshot and self.num_fewshot < 0:
            raise ValueError("`num_fewshot` must be non-negative.")


@dataclass
class AlpacaEvalTaskParams(EvaluationTaskParams):
    """Parameters for the AlpacaEval evaluation framework.

    AlpacaEval is an LLM-based automatic evaluation suite that is fast, cheap,
    replicable, and validated against 20K human annotations. The latest version
    (AlpacaEval 2.0) contains 805 prompts (tatsu-lab/alpaca_eval), which are open-ended
    questions. A model annotator (judge) is used to evaluate the quality of model's
    responses for these questions and calculates win rates vs. reference responses.
    The default judge is GPT4 Turbo.
    """

    version: Optional[float] = 2.0
    """The version of AlpacaEval to use. Options: 1.0 or 2.0 (default)."""

    def __post_init__(self):
        """Verifies params."""
        if self.version not in [1.0, 2.0]:
            raise ValueError("AlpacaEval `version` must be 1.0 or 2.0.")
