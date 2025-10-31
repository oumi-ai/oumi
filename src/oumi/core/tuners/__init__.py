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

"""Core tuners module for the Oumi (Open Universal Machine Intelligence) library.

This module provides various tuners implementations for use in the Oumi framework.
These tuners are designed to facilitate the hyper parameter tuning process.

Example:
    >>> from oumi.core.trainers import Tuner
    >>> tuner = Tuner(model=my_model, dataset=my_dataset, tuning_params=params) # doctest: +SKIP
    >>> trainer.optimize() # doctest: +SKIP

Note:
    For detailed information on each tuner, please refer to their respective
        class documentation.
"""  # noqa: E501

from oumi.core.tuners.base_tuner import BaseTuner
from oumi.core.tuners.optuna_tuner import OptunaTuner

__all__ = [
    "BaseTuner",
    "OptunaTuner",
]
