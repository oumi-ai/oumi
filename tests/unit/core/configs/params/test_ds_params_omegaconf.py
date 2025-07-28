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

"""Tests for DSParams OmegaConf integration."""

import sys
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

# Add root directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from oumi.core.configs.params.ds_params import (
    DeepSpeedOffloadDevice,
    DeepSpeedPrecision,
    DSParams,
    OffloadConfig,
    ZeRORuntimeStage,
)


class TestDSParamsOmegaConf:
    """Test suite for DSParams OmegaConf integration."""

    def test_basic_instantiation(self):
        """Test basic instantiation of DSParams with OmegaConf."""
        # Create DSParams instance
        params = DSParams()

        # Convert to OmegaConf DictConfig
        config = OmegaConf.structured(params)

        # Verify basic properties
        assert isinstance(config, DictConfig)
        assert config.enable_deepspeed is False
        assert config.zero_stage == ZeRORuntimeStage.ZERO_3

    def test_nested_dataclass_instantiation(self):
        """Test instantiation with nested dataclasses (OffloadConfig)."""
        # Create DSParams with offload config
        offload_config = OffloadConfig(
            device=DeepSpeedOffloadDevice.CPU, pin_memory=True, buffer_count=4
        )
        params = DSParams(enable_deepspeed=True, offload_optimizer=offload_config)

        # Convert to OmegaConf DictConfig
        config = OmegaConf.structured(params)

        # Verify nested structure
        assert config.enable_deepspeed is True
        assert config.offload_optimizer.device == DeepSpeedOffloadDevice.CPU
        assert config.offload_optimizer.pin_memory is True
        assert config.offload_optimizer.buffer_count == 4

    def test_validation_in_post_init(self):
        """Test that validation in __post_init__ works correctly."""
        # Test invalid parameter offloading configuration
        with pytest.raises(
            ValueError, match="Parameter offloading is only supported with ZeRO stage 3"
        ):
            DSParams(zero_stage=ZeRORuntimeStage.ZERO_2, offload_param=OffloadConfig())

        # Test invalid optimizer offloading configuration
        with pytest.raises(
            ValueError, match="Optimizer offloading requires ZeRO stage 2 or 3"
        ):
            DSParams(
                zero_stage=ZeRORuntimeStage.ZERO_1, offload_optimizer=OffloadConfig()
            )

    def test_omegaconf_merge(self):
        """Test merging configurations with OmegaConf."""
        # Base configuration
        base_params = DSParams(
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

    def test_to_deepspeed_config(self):
        """Test conversion to DeepSpeed configuration format."""
        params = DSParams(
            enable_deepspeed=True,
            zero_stage=ZeRORuntimeStage.ZERO_3,
            precision=DeepSpeedPrecision.BF16,
            offload_optimizer=OffloadConfig(device=DeepSpeedOffloadDevice.CPU),
        )

        # Convert to DeepSpeed config
        ds_config = params.to_deepspeed_config()

        # Verify structure
        assert isinstance(ds_config, dict)
        assert "zero_optimization" in ds_config
        assert ds_config["zero_optimization"]["stage"] == 3
        assert "bf16" in ds_config
        assert ds_config["bf16"]["enabled"] == "auto"
        assert "offload_optimizer" in ds_config["zero_optimization"]
        assert ds_config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    def test_enum_serialization(self):
        """Test that enums serialize correctly with OmegaConf."""
        params = DSParams(
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

    def test_default_factory_fields(self):
        """Test fields with default_factory work correctly."""
        params = DSParams()

        # Convert to OmegaConf
        config = OmegaConf.structured(params)

        # Verify default factory field
        assert isinstance(config.activation_checkpointing, DictConfig)
        assert len(config.activation_checkpointing) == 0

        # Add values to the dict
        config.activation_checkpointing["partition_activations"] = True
        assert config.activation_checkpointing.partition_activations is True


def run_tests():
    """Run the tests and print results."""
    print("Running DSParams OmegaConf integration tests...")
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
