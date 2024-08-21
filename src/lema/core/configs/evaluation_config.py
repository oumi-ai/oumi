from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from lema.core.configs.base_config import BaseConfig
from lema.core.configs.generation_config import GenerationConfig
from lema.core.configs.params.data_params import DatasetSplitParams
from lema.core.configs.params.model_params import ModelParams


class EvaluationFramework(Enum):
    """Enum representing the evaluation framework to use."""

    LEMA = "lema"
    LM_HARNESS = "lm_harness"


@dataclass
class EvaluationConfig(BaseConfig):
    data: DatasetSplitParams = field(default_factory=DatasetSplitParams)
    model: ModelParams = field(default_factory=ModelParams)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation_framework: EvaluationFramework = EvaluationFramework.LM_HARNESS
    #: Number of few-shot examples (with responses) to add in the prompt, in order to
    #: teach the model how to respond to the specific dataset's prompts.
    #: If not set (None): LM Harness will decide the value.
    #: If set to 0: no few-shot examples will be added in the prompt.
    num_shots: Optional[int] = None
    #: Number of samples/examples to evaluate from this dataset. Mostly for debugging,
    #: in order to reduce the runtime. If not set (None): the entire dataset is
    #: evaluated. If set, this must be a positive integer.
    num_samples: Optional[int] = None
    #: Where to write computed evaluations.
    output_dir: str = "output"

    def __post_init__(self):
        """Verifies params."""
        if not isinstance(self.evaluation_framework, EvaluationFramework):
            raise ValueError(
                "`evaluation_framework` must belong to class `EvaluationFramework`."
            )
        if self.evaluation_framework not in list(EvaluationFramework):
            raise ValueError(
                f"Unknown `evaluation_framework` value: {self.evaluation_framework}."
            )
        if self.num_shots and self.num_shots < 0:
            raise ValueError("`num_shots` must be non-negative.")
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError("`num_samples` must be None or a positive integer.")
