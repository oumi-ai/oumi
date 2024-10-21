from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.evaluation_params import LMHarnessParams
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.utils.str_utils import sanitize_run_name


class EvaluationFramework(Enum):
    """Enum representing the evaluation framework to use."""

    OUMI = "oumi"
    LM_HARNESS = "lm_harness"


@dataclass
class EvaluationConfig(BaseConfig):
    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model to be evaluated.

    This includes model architecture, size, dtype,
    and any specific configurations required for the evaluation task.
    """

    generation: GenerationParams = field(default_factory=GenerationParams)
    """Parameters for text generation during evaluation.

    This includes settings such as temperature, top-k, top-p,
    maximum length, and any other parameters that control the
    text generation process.
    """

    lm_harness_params: Optional[LMHarnessParams] = None
    """Parameters for the LM Harness evaluation framework.

    LM Harness is a comprehensive benchmarking suite for evaluating language models
    across various tasks.
    If specified, the tasks provided in the LMHarnessParams will be evaluated.
    """

    run_name: Optional[str] = None
    """A unique identifier for the current training run.
    This name is used to identify the run in Weights & Biases.
    """

    enable_wandb: bool = False
    """Whether to enable Weights & Biases (wandb) logging.
    If True, wandb will be used for experiment tracking and visualization.
    After enabling, you must set the `WANDB_API_KEY` environment variable.
    Alternatively, you can use the `wandb login` command to authenticate.
    """

    output_dir: str = "output"
    """Where to write computed evaluations."""

    def __post_init__(self):
        """Verifies params."""
        self.run_name = sanitize_run_name(self.run_name)
        if self.lm_harness_params is not None:
            if (
                self.lm_harness_params.num_fewshot
                and self.lm_harness_params.num_fewshot < 0
            ):
                raise ValueError("`num_fewshot` must be non-negative.")
            if (
                self.lm_harness_params.num_samples is not None
                and self.lm_harness_params.num_samples <= 0
            ):
                raise ValueError("`num_samples` must be None or a positive integer.")
