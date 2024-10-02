from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.generation_config import GenerationConfig
from oumi.core.configs.params.evaluation_params import LMHarnessParams
from oumi.core.configs.params.model_params import ModelParams


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

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    """Configuration for text generation during evaluation.

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

    output_dir: str = "output"
    """Where to write computed evaluations."""

    def __post_init__(self):
        """Verifies params."""
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
