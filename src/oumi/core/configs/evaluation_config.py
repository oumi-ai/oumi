from dataclasses import dataclass, field
from typing import Optional, Union

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.evaluation_params import AlpacaEvalParams, LMHarnessParams
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.utils.str_utils import sanitize_run_name


@dataclass
class EvaluationConfig(BaseConfig):
    tasks: Optional[list[Union[LMHarnessParams, AlpacaEvalParams]]] = None
    """List of all the evaluation tasks to run."""

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
