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

"""Tests for DeepSpeedParams functionality and OmegaConf integration."""

import pytest
from omegaconf import DictConfig, OmegaConf

from oumi.core.configs.params.deepspeed_params import (
    DeepSpeedOffloadDevice,
    DeepSpeedParams,
    DeepSpeedPrecision,
    OffloadConfig,
    ZeRORuntimeStage,
)


def test_deepspeed_params_basic_instantiation():
    """Test basic instantiation of DeepSpeedParams."""
    # Test basic instantiation
    params = DeepSpeedParams()
    assert params.enable_deepspeed is False
    assert params.zero_stage == ZeRORuntimeStage.ZERO_0

    # Test instantiation with parameters
    params = DeepSpeedParams(enable_deepspeed=True, zero_stage=ZeRORuntimeStage.ZERO_3)
    assert params.enable_deepspeed is True
    assert params.zero_stage == ZeRORuntimeStage.ZERO_3


def test_deepspeed_params_with_offload_config():
    """Test DeepSpeedParams instantiation with nested OffloadConfig."""
    # Test instantiation with nested dataclass (requires ZeRO stage 1, 2, or 3)
    params = DeepSpeedParams(
        enable_deepspeed=True,
        zero_stage=ZeRORuntimeStage.ZERO_2,
        offload_optimizer=OffloadConfig(),
    )
    assert params.enable_deepspeed is True
    assert params.zero_stage == ZeRORuntimeStage.ZERO_2
    assert params.offload_optimizer is not None

    # Test that optimizer offloading works with Stage 1
    params_stage1 = DeepSpeedParams(
        enable_deepspeed=True,
        zero_stage=ZeRORuntimeStage.ZERO_1,
        offload_optimizer=OffloadConfig(),
    )
    assert params_stage1.zero_stage == ZeRORuntimeStage.ZERO_1
    assert params_stage1.offload_optimizer is not None


def test_deepspeed_params_validation():
    """Test that DeepSpeedParams validation works correctly."""
    # Test parameter offloading validation - only supported with ZeRO stage 3
    with pytest.raises(
        ValueError, match="Parameter offloading is only supported with ZeRO stage 3"
    ):
        DeepSpeedParams(
            zero_stage=ZeRORuntimeStage.ZERO_2, offload_param=OffloadConfig()
        )

    # Test optimizer offloading validation - requires ZeRO stage 1, 2, or 3
    with pytest.raises(
        ValueError, match="Optimizer offloading requires ZeRO stage 1, 2, or 3"
    ):
        DeepSpeedParams(
            zero_stage=ZeRORuntimeStage.ZERO_0, offload_optimizer=OffloadConfig()
        )


def test_omegaconf_basic_integration():
    """Test basic DeepSpeedParams integration with OmegaConf."""
    # Create DeepSpeedParams instance
    params = DeepSpeedParams()

    # Convert to OmegaConf DictConfig
    config = OmegaConf.structured(params)

    # Verify basic properties
    assert isinstance(config, DictConfig)
    assert config.enable_deepspeed is False
    assert config.zero_stage == ZeRORuntimeStage.ZERO_0


def test_omegaconf_nested_dataclass():
    """Test OmegaConf integration with nested dataclasses (OffloadConfig)."""
    # Create DeepSpeedParams with offload config
    offload_config = OffloadConfig(
        device=DeepSpeedOffloadDevice.CPU, pin_memory=True, buffer_count=4
    )
    params = DeepSpeedParams(
        enable_deepspeed=True,
        zero_stage=ZeRORuntimeStage.ZERO_2,
        offload_optimizer=offload_config,
    )

    # Convert to OmegaConf DictConfig
    config = OmegaConf.structured(params)

    # Verify nested structure
    assert config.enable_deepspeed is True
    assert config.offload_optimizer.device == DeepSpeedOffloadDevice.CPU
    assert config.offload_optimizer.pin_memory is True
    assert config.offload_optimizer.buffer_count == 4


def test_omegaconf_merge():
    """Test merging configurations with OmegaConf."""
    # Base configuration
    base_params = DeepSpeedParams(
        enable_deepspeed=True,
        zero_stage=ZeRORuntimeStage.ZERO_2,
        precision=DeepSpeedPrecision.FP16,
    )
    base_config = OmegaConf.structured(base_params)

    # Override configuration
    override_config = OmegaConf.create(
        {"zero_stage": "ZERO_3", "precision": "BF16", "steps_per_print": 20}
    )

    # Merge configurations
    merged_config = OmegaConf.merge(base_config, override_config)

    # Verify merged values
    assert merged_config.enable_deepspeed is True
    assert merged_config.zero_stage == ZeRORuntimeStage.ZERO_3
    assert merged_config.precision == DeepSpeedPrecision.BF16
    assert merged_config.steps_per_print == 20


def test_to_deepspeed():
    """Test conversion to DeepSpeed configuration format."""
    params = DeepSpeedParams(
        enable_deepspeed=True,
        zero_stage=ZeRORuntimeStage.ZERO_3,
        precision=DeepSpeedPrecision.BF16,
        offload_optimizer=OffloadConfig(device=DeepSpeedOffloadDevice.CPU),
    )

    # Convert to DeepSpeed config
    ds_config = params.to_deepspeed()

    # Verify structure
    assert isinstance(ds_config, dict)
    assert "zero_optimization" in ds_config
    assert ds_config["zero_optimization"]["stage"] == 3
    assert "bf16" in ds_config
    assert ds_config["bf16"]["enabled"] == "auto"
    assert "offload_optimizer" in ds_config["zero_optimization"]
    assert ds_config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"


def test_enum_serialization():
    """Test that enums serialize correctly with OmegaConf."""
    params = DeepSpeedParams(
        zero_stage=ZeRORuntimeStage.ZERO_3, precision=DeepSpeedPrecision.BF16
    )

    # Convert to OmegaConf and back to dict
    config = OmegaConf.structured(params)
    config_dict = OmegaConf.to_container(config)

    # Verify enum values are strings
    assert config_dict is not None
    assert isinstance(config_dict, dict)
    assert config_dict["zero_stage"] == "3"
    assert config_dict["precision"] == "bf16"


def test_default_factory_fields():
    """Test fields with default_factory work correctly."""
    params = DeepSpeedParams()

    # Convert to OmegaConf
    config = OmegaConf.structured(params)

    # Verify default factory field
    assert isinstance(config.activation_checkpointing, DictConfig)
    assert len(config.activation_checkpointing) == 0

    # Add values to the dict
    config.activation_checkpointing["partition_activations"] = True
    assert config.activation_checkpointing.partition_activations is True


def test_default_values_match_deepspeed():
    """Test that Oumi defaults match DeepSpeed defaults."""
    params = DeepSpeedParams()

    # Test the key defaults that were aligned with DeepSpeed
    assert params.zero_stage == ZeRORuntimeStage.ZERO_0  # DeepSpeed default
    assert params.overlap_comm is False  # DeepSpeed default
    assert (
        params.stage3_gather_16bit_weights_on_model_save is False
    )  # DeepSpeed default
