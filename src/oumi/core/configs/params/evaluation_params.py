from dataclasses import dataclass, field
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
class BaseEvaluationTaskParams(BaseParams):
    """Base task parameters, which are applicable to ALL evaluation platforms."""

    evaluation_platform: EvaluationPlatform = MISSING
    """The evaluation platform to use for the current task."""

    num_samples: Optional[int] = None
    """Number of samples/examples to evaluate from this dataset.

    Mostly for debugging, in order to reduce the runtime.
    If not set (None): the entire dataset is evaluated.
    If set, this must be a positive integer.
    """

    eval_kwargs: Optional[dict[str, Any]] = None
    """Additional keyword arguments to pass to the evaluation function.

    This allows for passing any evaluation-specific parameters that are not
    covered by other fields in *TaskParams classes.
    """

    def __post_init__(self):
        """Verifies params."""
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError("`num_samples` must be None or a positive integer.")


@dataclass
class LMHarnessTaskParams(BaseEvaluationTaskParams):
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
class AlpacaEvalTaskParams(BaseEvaluationTaskParams):
    """Parameters for the AlpacaEval evaluation framework.

    AlpacaEval is an LLM-based automatic evaluation suite that is fast, cheap,
    replicable, and validated against 20K human annotations. The latest version
    (AlpacaEval 2.0) contains 805 prompts (tatsu-lab/alpaca_eval), which are open-ended
    questions. A model annotator (judge) is used to evaluate the quality of model's
    responses for these questions and calculates win rates vs. reference responses.
    The default judge is GPT4 Turbo.
    """

    placeholder = None


@dataclass
class CustomEvaluationParams(BaseParams):
    """Parameters for running custom evaluations."""

    data: DatasetSplitParams = field(default_factory=DatasetSplitParams)
    """Parameters for the dataset split to be used in evaluation.

    This includes specifications for train, validation, and test splits,
    as well as any data preprocessing parameters.
    """


def evaluation_task_params_factory(
    task_params: dict[str, Any],
) -> BaseEvaluationTaskParams:
    """Factory that creates the evaluation task params object, based on a dict."""
    if "evaluation_platform" not in task_params:
        raise ValueError(
            "Missing `evaluation_platform` in task params. When running evaluations, "
            "it is necessary to specify the evaluation platform to use for EACH task. "
            "The available platforms can be found in the following enum: "
            "`oumi.core.configs.params.evaluation_params.EvaluationPlatform`. Current "
            f"options are: {', '.join([p.value for p in EvaluationPlatform])}."
        )
    elif task_params["evaluation_platform"] == EvaluationPlatform.LM_HARNESS.value:
        for param in task_params:
            if (not hasattr(LMHarnessTaskParams, param)) and (
                not hasattr(BaseEvaluationTaskParams, param)
            ):
                raise ValueError(
                    f"Unknown parameter: {param}. Please check the following class for "
                    "the available parameters of the LM Harness evaluation platform: "
                    "`oumi.core.configs.params.evaluation_params.LMHarnessTaskParams`. "
                    "If you wish to add a parameter that is supported by LM Harness "
                    "but not supported by Oumi, you can add this in the `eval_kwargs` "
                    "dictionary and it will be passed to the evaluation function. "
                )
        return LMHarnessTaskParams(**task_params)
    elif task_params["evaluation_platform"] == EvaluationPlatform.ALPACA_EVAL.value:
        raise NotImplementedError("Alpaca Eval is not yet supported.")
    else:
        raise ValueError(
            f"Unknown evaluation platform: `{task_params['evaluation_platform']}`. "
            f"Current options are: {', '.join([p.value for p in EvaluationPlatform])}."
        )
