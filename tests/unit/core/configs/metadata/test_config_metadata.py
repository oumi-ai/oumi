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

"""Tests for the ConfigMetadata dataclass."""

import pytest

from oumi.core.configs.metadata.config_metadata import (
    ConfigMetadata,
    ConfigType,
    FinetuningType,
    TrainingMethod,
)


class TestConfigMetadata:
    """Tests for ConfigMetadata dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metadata = ConfigMetadata(
            config_type=ConfigType.TRAINING,
            model_family="llama",
            model_size_billions=8.0,
            training_method=TrainingMethod.SFT,
            finetuning_type=FinetuningType.LORA,
            min_vram_gb=20.0,
            recommended_gpus=1,
            is_vision_model=False,
            tags=["beginner-friendly"],
            description="Test config",
        )

        result = metadata.to_dict()

        assert result["config_type"] == "training"
        assert result["model_family"] == "llama"
        assert result["model_size_billions"] == 8.0
        assert result["training_method"] == "sft"
        assert result["finetuning_type"] == "lora"
        assert result["min_vram_gb"] == 20.0
        assert result["recommended_gpus"] == 1
        assert result["is_vision_model"] is False
        assert result["tags"] == ["beginner-friendly"]
        assert result["description"] == "Test config"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "config_type": "training",
            "model_family": "qwen",
            "model_size_billions": 32.0,
            "training_method": "dpo",
            "finetuning_type": "qlora",
            "min_vram_gb": 48.0,
            "recommended_gpus": 2,
            "is_vision_model": True,
            "tags": ["advanced", "multi-gpu"],
            "description": "DPO training config",
        }

        metadata = ConfigMetadata.from_dict(data)

        assert metadata.config_type == ConfigType.TRAINING
        assert metadata.model_family == "qwen"
        assert metadata.model_size_billions == 32.0
        assert metadata.training_method == TrainingMethod.DPO
        assert metadata.finetuning_type == FinetuningType.QLORA
        assert metadata.min_vram_gb == 48.0
        assert metadata.recommended_gpus == 2
        assert metadata.is_vision_model is True
        assert metadata.tags == ["advanced", "multi-gpu"]
        assert metadata.description == "DPO training config"

    def test_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        original = ConfigMetadata(
            config_type=ConfigType.INFERENCE,
            model_family="gemma",
            model_size_billions=7.0,
            training_method=None,
            finetuning_type=None,
            min_vram_gb=16.0,
            recommended_gpus=1,
            is_vision_model=False,
            tags=["inference"],
            description="Inference config",
        )

        data = original.to_dict()
        restored = ConfigMetadata.from_dict(data)

        assert restored.config_type == original.config_type
        assert restored.model_family == original.model_family
        assert restored.model_size_billions == original.model_size_billions
        assert restored.min_vram_gb == original.min_vram_gb
        assert restored.tags == original.tags
        assert restored.description == original.description

    def test_default_values(self):
        """Test default values are correctly set."""
        metadata = ConfigMetadata(config_type=ConfigType.TRAINING)

        assert metadata.model_family is None
        assert metadata.model_size_billions is None
        assert metadata.training_method is None
        assert metadata.finetuning_type is None
        assert metadata.min_vram_gb is None
        assert metadata.recommended_gpus is None
        assert metadata.is_vision_model is False
        assert metadata.tags == []
        assert metadata.description is None

    def test_from_dict_with_missing_fields(self):
        """Test from_dict handles missing fields gracefully."""
        data = {
            "config_type": "evaluation",
            "model_family": "llama",
        }

        metadata = ConfigMetadata.from_dict(data)

        assert metadata.config_type == ConfigType.EVALUATION
        assert metadata.model_family == "llama"
        assert metadata.model_size_billions is None
        assert metadata.training_method is None
        assert metadata.tags == []


class TestEnums:
    """Tests for metadata enums."""

    def test_config_type_values(self):
        """Test ConfigType enum values."""
        assert ConfigType.TRAINING.value == "training"
        assert ConfigType.INFERENCE.value == "inference"
        assert ConfigType.EVALUATION.value == "evaluation"
        assert ConfigType.JOB.value == "job"

    def test_training_method_values(self):
        """Test TrainingMethod enum values."""
        assert TrainingMethod.SFT.value == "sft"
        assert TrainingMethod.DPO.value == "dpo"
        assert TrainingMethod.GRPO.value == "grpo"
        assert TrainingMethod.KTO.value == "kto"

    def test_finetuning_type_values(self):
        """Test FinetuningType enum values."""
        assert FinetuningType.FULL.value == "full"
        assert FinetuningType.LORA.value == "lora"
        assert FinetuningType.QLORA.value == "qlora"
