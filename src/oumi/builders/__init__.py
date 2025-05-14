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

"""Builders module for the Oumi (Open Universal Machine Intelligence) library.

This module provides builder functions to construct and configure
different components of the Oumi framework, including datasets, models,
optimizers, and trainers.

The builder functions encapsulate the complexity of creating these components,
allowing for easier setup and configuration of machine learning experiments.
"""

from oumi.builders.callbacks import build_training_callbacks
from oumi.builders.collators import build_collator_from_config, build_data_collator
from oumi.builders.data import (
    build_dataset,
    build_dataset_mixture,
)
from oumi.builders.metrics import build_metrics_function
from oumi.builders.models import (
    build_chat_template,
    build_model,
    build_peft_model,
    build_tokenizer,
    is_image_text_llm,
)
from oumi.builders.optimizers import build_optimizer
from oumi.builders.processors import build_processor
from oumi.builders.rewards import build_reward_functions
from oumi.builders.training import build_trainer

__all__ = [
    "build_chat_template",
    "build_dataset_mixture",
    "build_dataset",
    "build_data_collator",
    "build_metrics_function",
    "build_model",
    "build_optimizer",
    "build_peft_model",
    "build_processor",
    "build_reward_functions",
    "build_tokenizer",
    "build_trainer",
    "build_training_callbacks",
    "is_image_text_llm",
    "build_collator_from_config",
]
