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

from dataclasses import dataclass, field

from omegaconf import MISSING

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.evaluation_config import EvaluationConfig


@dataclass
class AsyncEvaluationConfig(BaseConfig):
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    """The evaluation configuration to use for each checkpoint.

    This field specifies the EvaluationConfig object that defines the parameters
    for evaluating each checkpoint. It includes settings for the dataset,
    model, generation, and evaluation framework to be used.
    """

    checkpoints_dir: str = MISSING
    """The directory to poll for new checkpoints."""

    polling_interval: float = MISSING
    """The time in seconds between the end of the previous evaluation and the start of
    the next polling attempt. Cannot be negative.
    """

    num_retries: int = 5
    """The number of times to retry polling before exiting the current job.

    A retry occurs when the job reads the target directory but cannot find a new
    model checkpoint to evaluate. Defaults to 5. Cannot be negative.
    """

    def __post_init__(self):
        """Verifies/populates params."""
        if self.polling_interval < 0:
            raise ValueError("`polling_interval` must be non-negative.")
        if self.num_retries < 0:
            raise ValueError("`num_retries` must be non-negative.")
