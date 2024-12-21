from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from omegaconf import MISSING

from oumi.core.configs.inference_config import InferenceEngineType
from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.data_params import DatasetSplitParams
from oumi.core.configs.params.remote_params import RemoteParams


class EvaluationPlatform(Enum):
    """Enum representing the evaluation platform to use."""

    LM_HARNESS = "lm_harness"
    ALPACA_EVAL = "alpaca_eval"


@dataclass
class BaseTaskParams(BaseParams):
    """Base task parameters, which are applicable to ALL evaluation platforms."""

    num_samples: Optional[int] = None
    """Number of samples/examples to evaluate from this dataset.

    Mostly for debugging, in order to reduce the runtime.
    If not set (None): the entire dataset is evaluated.
    If set, this must be a positive integer.
    """

    def __post_init__(self):
        """Verifies params."""
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError("`num_samples` must be None or a positive integer.")


@dataclass
class LMHarnessTaskParams(BaseTaskParams):
    """Parameters for the LM Harness evaluation framework.

    LM Harness is a comprehensive benchmarking suite for evaluating language models
    across various tasks.
    """

    task_name: str = MISSING
    """The LM Harness task to evaluate.

    A list of all tasks is available at
    https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
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
class AlpacaEvalTaskParams(BaseTaskParams):
    """Parameters for the AlpacaEval evaluation framework.

    AlpacaEval is an LLM-based automatic evaluation suite that is fast, cheap,
    replicable, and validated against 20K human annotations. The latest version
    (AlpacaEval 2.0) contains 805 prompts (tatsu-lab/alpaca_eval), which are open-ended
    questions. A model annotator (judge) is used to evaluate the quality of model's
    responses for these questions and calculates win rates vs. reference responses.
    The default judge is GPT4 Turbo.
    """

    inference_engine: Optional[InferenceEngineType] = None
    """The inference engine to use for generation."""

    inference_remote_params: Optional[RemoteParams] = None
    """Parameters for running inference against a remote API."""


@dataclass
class EvaluationTaskParams(BaseParams):
    """Wrapper for task params of different evaluation platforms."""

    lm_harness_task_params: Optional[LMHarnessTaskParams] = None
    """Used when the task will be evaluated using the LM Harness evaluation platform.
    Only a single *_task_params variable can be set in this class, so this is mutually
    exclusive with `alpaca_eval_task_params`.
    """

    alpaca_eval_task_params: Optional[AlpacaEvalTaskParams] = None
    """Used when the task will be evaluated using the AlpacaEval evaluation platform.
    Only a single *_task_params variable can be set in this class, so this is mutually
    exclusive with `lm_harness_task_params`."""

    def evaluation_platform(self):
        """Returns the evaluation platform to use for the current task."""
        if self.lm_harness_task_params:
            return EvaluationPlatform.LM_HARNESS
        elif self.alpaca_eval_task_params:
            return EvaluationPlatform.ALPACA_EVAL
        else:
            raise ValueError("No task params available")

    def __post_init__(self):
        """Verifies params."""
        if not any([self.lm_harness_task_params, self.alpaca_eval_task_params]):
            raise ValueError(
                "At least one task params variable must be set. Please define either "
                "`lm_harness_task_params` or `alpaca_eval_task_params`"
            )
        if all([self.lm_harness_task_params, self.alpaca_eval_task_params]):
            raise ValueError(
                "Only one task params variable can be set. Please define either "
                "`lm_harness_task_params` or `alpaca_eval_task_params`"
            )


@dataclass
class CustomEvaluationParams(BaseParams):
    """Parameters for running custom evaluations."""

    data: DatasetSplitParams = field(default_factory=DatasetSplitParams)
    """Parameters for the dataset split to be used in evaluation.

    This includes specifications for train, validation, and test splits,
    as well as any data preprocessing parameters.
    """
