from dataclasses import dataclass, field

from omegaconf import MISSING

from lema.core.configs.base_config import BaseConfig
from lema.core.configs.evaluation_config import EvaluationConfig


@dataclass
class AsyncEvaluationConfig(BaseConfig):
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    #: The directory to poll for new checkpoints.
    checkpoints_dir: str = MISSING

    #: The time in seconds between the end of the previous evaluation and the start of
    #: the next polling attempt. Cannot be negative.
    polling_interval: float = MISSING

    #: The number of times to retry polling before exiting the current job.
    #: A retry occurs when the job reads the target directory but cannot find a new
    #: model checkpoint to evaluate. Defaults to 5. Cannot be negative.
    num_retries: int = 5

    def __post_init__(self):
        """Verifies/populates params."""
        if self.polling_interval < 0:
            raise ValueError("`polling_interval` must be non-negative.")
        if self.num_retries < 0:
            raise ValueError("`num_retries` must be non-negative.")
