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

"""Core trainers module for the Oumi (Open Universal Machine Intelligence) library.

This module provides various trainer implementations for use in the Oumi framework.
These trainers are designed to facilitate the training process for different
types of models and tasks.

Example:
    >>> from oumi.core.trainers import Trainer
    >>> trainer = Trainer(model=my_model, dataset=my_dataset) # doctest: +SKIP
    >>> trainer.train() # doctest: +SKIP

Note:
    For detailed information on each trainer, please refer to their respective
        class documentation.
"""

from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.core.trainers.hf_trainer import HuggingFaceTrainer
from oumi.core.trainers.oumi_trainer import Trainer
from oumi.core.trainers.verl_grpo_trainer import VerlGrpoTrainer

__all__ = [
    "BaseTrainer",
    "HuggingFaceTrainer",
    "Trainer",
    "VerlGrpoTrainer",
]
