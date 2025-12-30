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

import pytest

from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.prompt_optimization_params import (
    PromptOptimizationParams,
)
from oumi.core.configs.prompt_config import PromptOptimizationConfig


def test_prompt_config_requires_engine():
    """Prompt optimization must specify an inference engine."""
    config = PromptOptimizationConfig(
        model=ModelParams(model_name="test-model"),
        generation=GenerationParams(),
        optimization=PromptOptimizationParams(optimizer="mipro", num_trials=1),
        train_dataset_path="dummy.jsonl",
        output_dir="dummy_output",
    )

    with pytest.raises(
        ValueError,
        match="engine must be specified for prompt optimization inference",
    ):
        config.finalize_and_validate()
