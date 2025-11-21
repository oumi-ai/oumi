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

from oumi.core.configs import TunerType, TuningParams
from oumi.core.tuners import BaseTuner, OptunaTuner


def build_tuner(tuning_params: TuningParams) -> BaseTuner:
    """Build a tuner based on the configuration.

    Args:
        tuning_params: Tuning configuration parameters.

    Returns:
        An instance of the appropriate tuner implementation.

    Raises:
        NotImplementedError: If the tuner type is not supported.
    """
    if tuning_params.tuner_type == TunerType.OPTUNA:
        return OptunaTuner(tuning_params)

    raise NotImplementedError(f"Tuner type {tuning_params.tuner_type} not supported.")
