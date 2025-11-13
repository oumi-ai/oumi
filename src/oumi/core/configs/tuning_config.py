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

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.data_params import DataParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.tuning_params import TuningParams


@dataclass
class TuningConfig(BaseConfig):
    data: DataParams = field(default_factory=DataParams)
    """Parameters for the dataset.

    This field contains all the necessary settings for data processing and loading.
    It includes options for train and evaluation datasets and preprocessing steps.

    For more details, see the :class:`oumi.core.configs.params.data_params.DataParams`
    class.
    """

    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model.

    This field defines the model architecture, size, and other model-specific settings.
    It includes options for model type, pretrained weights, and tokenizer configuration.

    For more details, see :class:`oumi.core.configs.params.model_params.ModelParams`
    class.
    """

    tuning_params: TuningParams = field(default_factory=TuningParams)
    """Parameters for the tuning process.

    This field includes settings reagarding the tuning process and specific to the
    tuning algorithm, such as the number of jobs to run in parallel.

    For more details, see :class:`oumi.core.configs.params.model_params.TuningParams`
    """

    def __post_init__(self):
        """Verifies/populates params."""
        # for param_name, values in self.tuning_params.tunable_params.items():
        #     if self.tuning_params.tuner_type == TunerType.OPTUNA:
