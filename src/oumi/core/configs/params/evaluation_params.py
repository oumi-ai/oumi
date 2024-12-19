from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from omegaconf import MISSING

from oumi.core.configs import InferenceEngineType
from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.data_params import DatasetSplitParams
from oumi.core.configs.params.remote_params import RemoteParams


class EvaluationPlatform(Enum):
    """Enum representing the evaluation platform to use."""

    LM_HARNESS = "lm_harness"
    ALPACA_EVAL = "alpaca_eval"


@dataclass
class EvalTaskParams(BaseParams):
    """Parameters for a single evaluation task."""

    evaluation_platform: EvaluationPlatform = MISSING
    """The name of the evaluation platform that can run this task."""

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
class LMHarnessParams(EvalTaskParams):
    """Parameters for the LM Harness evaluation framework.

    LM Harness is a comprehensive benchmarking suite for evaluating language models
    across various tasks.
    """

    evaluation_platform: EvaluationPlatform = EvaluationPlatform.LM_HARNESS
    """The name of the evaluation platform that can run this task."""

    tasks: list[str] = MISSING
    """The LM Harness tasks to evaluate.

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
        if not self.tasks:
            raise ValueError("`tasks` must include at least 1 task.")
        if self.num_fewshot and self.num_fewshot < 0:
            raise ValueError("`num_fewshot` must be non-negative.")


@dataclass
class AlpacaEvalParams(EvalTaskParams):
    """Parameters for the AlpacaEval evaluation framework.

    AlpacaEval is an LLM-based automatic evaluation suite that is fast, cheap,
    replicable, and validated against 20K human annotations. The latest version
    (AlpacaEval 2.0) contains 805 prompts (tatsu-lab/alpaca_eval), which are open-ended
    questions. A model annotator (judge) is used to evaluate the quality of model's
    responses for these questions and calculates win rates vs. reference responses.
    The default judge is GPT4 Turbo.
    """

    evaluation_platform: EvaluationPlatform = EvaluationPlatform.ALPACA_EVAL
    """The name of the evaluation platform that can run this task."""

    inference_engine: Optional[InferenceEngineType] = None
    """The inference engine to use for generation."""

    inference_remote_params: Optional[RemoteParams] = None
    """Parameters for running inference against a remote API."""


@dataclass
class CustomEvaluationParams(BaseParams):
    """Parameters for running custom evaluations."""

    data: DatasetSplitParams = field(default_factory=DatasetSplitParams)
    """Parameters for the dataset split to be used in evaluation.

    This includes specifications for train, validation, and test splits,
    as well as any data preprocessing parameters.
    """
