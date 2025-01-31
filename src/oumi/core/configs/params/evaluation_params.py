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

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Optional

from omegaconf import MISSING

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.data_params import DatasetSplitParams


class EvaluationPlatform(Enum):
    """Enum representing the evaluation platform to use."""

    LM_HARNESS = "lm_harness"
    ALPACA_EVAL = "alpaca_eval"


@dataclass
class EvaluationTaskParams(BaseParams):
    """Configuration parameters for model evaluation tasks.

    Supported platforms:

    - LM Harness: Framework for evaluating language models on standard benchmarks.
      A list of all supported tasks can be found at:
      https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks.
    - Alpaca Eval: Framework for evaluating language models on instruction-following
      and quality of responses on open-ended questions.

    Examples:
        .. code-block:: python

            # LM Harness evaluation on MMLU
            params = EvaluationTaskParams(
                evaluation_platform="lm_harness",
                task_name="mmlu",
                eval_kwargs={"num_fewshot": 5}
            )


        .. code-block:: python

            # Alpaca Eval 2.0 evaluation
            params = EvaluationTaskParams(
                evaluation_platform="alpaca_eval"
            )
    """

    evaluation_platform: str = MISSING
    """The evaluation platform to use for the current task."""

    task_name: Optional[str] = None
    """The task to evaluate."""

    num_samples: Optional[int] = None
    """Number of samples/examples to evaluate from this dataset.

    Mostly for debugging, in order to reduce the runtime.
    If not set (None): the entire dataset is evaluated.
    If set, this must be a positive integer.
    """

    eval_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to the evaluation function.

    This allows for passing any evaluation-specific parameters that are not
    covered by other fields in TaskParams classes.
    """

    def get_evaluation_platform(self) -> EvaluationPlatform:
        """Returns the evaluation platform as an Enum."""
        if not self.evaluation_platform:
            raise ValueError(
                "Missing `evaluation_platform`. When running evaluations, it is "
                "necessary to specify the evaluation platform to use for EACH task. "
                "The available platforms can be found in the following enum: "
                "`oumi.core.configs.params.evaluation_params.EvaluationPlatform`. "
                f"Current options: {EvaluationTaskParams.list_evaluation_platforms()}."
            )
        elif self.evaluation_platform == EvaluationPlatform.LM_HARNESS.value:
            return EvaluationPlatform.LM_HARNESS
        elif self.evaluation_platform == EvaluationPlatform.ALPACA_EVAL.value:
            return EvaluationPlatform.ALPACA_EVAL
        else:
            raise ValueError(f"Unknown evaluation platform: {self.evaluation_platform}")

    def get_evaluation_platform_task_params(self):
        """Returns the evaluation platform-specific task parameters."""
        if self.get_evaluation_platform() == EvaluationPlatform.LM_HARNESS:
            target_class = LMHarnessTaskParams
        elif self.get_evaluation_platform() == EvaluationPlatform.ALPACA_EVAL:
            target_class = AlpacaEvalTaskParams
        else:
            raise ValueError(f"Unknown evaluation platform: {self.evaluation_platform}")

        init_kwargs = self._get_init_kwargs_for_task_params_class(target_class)
        return target_class(**init_kwargs)

    @staticmethod
    def list_evaluation_platforms() -> str:
        """Returns a string listing all available evaluation platforms."""
        return ", ".join([platform.value for platform in EvaluationPlatform])

    def _get_init_kwargs_for_task_params_class(self, target_class) -> dict[str, Any]:
        """Returns the init keyword arguments for a `target_class` of name *TaskParams.

        Given a target class of name <evaluation platform>_TaskParams, which inherits
        from the current class, this method returns a 'flattened' dict that includes all
        arguments needed to instantiate it. The dict includes all the parameters which
        are already members of the current class, as well as additional parameters which
        are only known to the target class (stored under `eval_kwargs`). By 'flattened',
        we mean that all known parameters that are stored under the `eval_kwargs` dict
        are moved one level up, to the (flat) dict that is returned. In contrast, all
        unknown (to the target class) parameters remain (unflattened) inside the
        `eval_kwargs` dict.
        """
        # Find all keys in `eval_kwargs` which are known to the target class.
        known_keys = []
        if self.eval_kwargs:
            field_names = [field.name for field in fields(target_class)]
            known_keys.extend(key for key in self.eval_kwargs if key in field_names)

        # Identify all kwargs known to the current class.
        init_keys = [
            key
            for key in dir(self)
            if not callable(getattr(self, key)) and not key.startswith("_")
        ]
        init_kwargs = {key: getattr(self, key) for key in init_keys}

        # Move known kwargs one level up: from `eval_kwargs` to the top-level dict.
        for key in known_keys:
            if key in init_kwargs:
                raise ValueError(
                    f"Parameter `{key}` is present twice, in both task parameters and "
                    "`eval_kwargs` dictionary. Please remove it from one of them."
                )
            init_kwargs[key] = init_kwargs["eval_kwargs"].pop(key)

        return init_kwargs

    def __post_init__(self):
        """Verifies params."""
        if (
            self.get_evaluation_platform() == EvaluationPlatform.LM_HARNESS
            and not self.task_name
        ):
            raise ValueError("`task_name` must be a valid LM Harness task.")
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError("`num_samples` must be None or a positive integer.")


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


@dataclass
class CustomEvaluationParams(BaseParams):
    """Parameters for running custom evaluations."""

    data: DatasetSplitParams = field(default_factory=DatasetSplitParams)
    """Parameters for the dataset split to be used in evaluation.

    This includes specifications for train, validation, and test splits,
    as well as any data preprocessing parameters.
    """
