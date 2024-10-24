from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.data_params import DatasetSplitParams


@dataclass
class LMHarnessParams(BaseParams):
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

    num_samples: Optional[int] = None
    """Number of samples/examples to evaluate from this dataset.

    Mostly for debugging, in order to reduce the runtime.
    If not set (None): the entire dataset is evaluated.
    If set, this must be a positive integer.
    """


@dataclass
class CustomEvaluationParams(BaseParams):
    """Parameters for running custom evaluations."""

    data: DatasetSplitParams = field(default_factory=DatasetSplitParams)
    """Parameters for the dataset split to be used in evaluation.

    This includes specifications for train, validation, and test splits,
    as well as any data preprocessing parameters.
    """
