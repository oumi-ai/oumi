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

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.synthesis_params import GeneralSynthesisParams
from oumi.core.configs.synthesis_config import SynthesisConfig, SynthesisStrategy


def test_default_synthesis_config():
    """Test default initialization of SynthesisConfig."""
    config = SynthesisConfig()

    assert config.strategy == SynthesisStrategy.GENERAL
    assert isinstance(config.strategy_params, GeneralSynthesisParams)
    assert isinstance(config.inference_config, InferenceConfig)
    assert config.num_samples == 1


def test_custom_synthesis_config():
    """Test custom initialization of SynthesisConfig."""
    custom_params = GeneralSynthesisParams()
    custom_inference = InferenceConfig()

    config = SynthesisConfig(
        strategy=SynthesisStrategy.GENERAL,
        strategy_params=custom_params,
        inference_config=custom_inference,
        num_samples=5,
    )

    assert config.strategy == SynthesisStrategy.GENERAL
    assert config.strategy_params == custom_params
    assert config.inference_config == custom_inference
    assert config.num_samples == 5


def test_invalid_strategy():
    """Test that invalid strategy raises ValueError."""
    config = SynthesisConfig()
    config.strategy = "invalid_strategy"  # type: ignore

    with pytest.raises(ValueError, match="Unsupported synthesis strategy"):
        config.__post_init__()


def test_invalid_input_path():
    """Test that setting input_path raises ValueError."""
    inference_config = InferenceConfig(input_path="some/path")

    with pytest.raises(ValueError, match="Input path is not supported"):
        SynthesisConfig(inference_config=inference_config)


def test_invalid_output_path():
    """Test that setting output_path raises ValueError."""
    inference_config = InferenceConfig(output_path="some/path")

    with pytest.raises(ValueError, match="Output path is not supported"):
        SynthesisConfig(inference_config=inference_config)
