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

"""Tests for the VRAM estimator module."""

import pytest

from oumi.core.configs.metadata.config_metadata import FinetuningType
from oumi.core.configs.metadata.vram_estimator import (
    estimate_training_vram_gb,
    get_recommended_gpus,
    get_vram_tier,
)


class TestEstimateTrainingVramGb:
    """Tests for estimate_training_vram_gb function."""

    def test_qlora_small_model(self):
        """Test QLoRA VRAM estimate for a small model."""
        vram = estimate_training_vram_gb(
            model_size_billions=1.0,
            finetuning_type=FinetuningType.QLORA,
            dtype="bfloat16",
        )
        # QLoRA for 1B model should require relatively little VRAM
        assert 0.5 < vram < 10

    def test_qlora_vs_lora(self):
        """Test that QLoRA requires less VRAM than LoRA."""
        qlora_vram = estimate_training_vram_gb(
            model_size_billions=8.0,
            finetuning_type=FinetuningType.QLORA,
        )
        lora_vram = estimate_training_vram_gb(
            model_size_billions=8.0,
            finetuning_type=FinetuningType.LORA,
        )
        # QLoRA should require significantly less VRAM due to quantization
        assert qlora_vram < lora_vram

    def test_lora_vs_full(self):
        """Test that LoRA requires less VRAM than full fine-tuning."""
        lora_vram = estimate_training_vram_gb(
            model_size_billions=8.0,
            finetuning_type=FinetuningType.LORA,
        )
        full_vram = estimate_training_vram_gb(
            model_size_billions=8.0,
            finetuning_type=FinetuningType.FULL,
        )
        # LoRA should require much less VRAM than full fine-tuning
        assert lora_vram < full_vram

    def test_larger_model_more_vram(self):
        """Test that larger models require more VRAM."""
        vram_8b = estimate_training_vram_gb(
            model_size_billions=8.0,
            finetuning_type=FinetuningType.QLORA,
        )
        vram_70b = estimate_training_vram_gb(
            model_size_billions=70.0,
            finetuning_type=FinetuningType.QLORA,
        )
        # 70B should require more VRAM than 8B
        assert vram_70b > vram_8b

    def test_gradient_checkpointing_reduces_vram(self):
        """Test that gradient checkpointing reduces VRAM estimate."""
        vram_with_gc = estimate_training_vram_gb(
            model_size_billions=8.0,
            finetuning_type=FinetuningType.FULL,
            gradient_checkpointing=True,
        )
        vram_without_gc = estimate_training_vram_gb(
            model_size_billions=8.0,
            finetuning_type=FinetuningType.FULL,
            gradient_checkpointing=False,
        )
        # Gradient checkpointing should reduce VRAM
        assert vram_with_gc < vram_without_gc

    def test_larger_batch_size_more_vram(self):
        """Test that larger batch sizes require more VRAM."""
        vram_bs1 = estimate_training_vram_gb(
            model_size_billions=8.0,
            finetuning_type=FinetuningType.LORA,
            batch_size=1,
        )
        vram_bs4 = estimate_training_vram_gb(
            model_size_billions=8.0,
            finetuning_type=FinetuningType.LORA,
            batch_size=4,
        )
        # Larger batch size should require more VRAM
        assert vram_bs4 > vram_bs1


class TestGetRecommendedGpus:
    """Tests for get_recommended_gpus function."""

    def test_single_gpu_sufficient(self):
        """Test that single GPU is recommended for small VRAM."""
        assert get_recommended_gpus(20.0) == 1
        assert get_recommended_gpus(80.0) == 1

    def test_multi_gpu_needed(self):
        """Test that multiple GPUs are recommended for large VRAM."""
        assert get_recommended_gpus(100.0) == 2
        assert get_recommended_gpus(200.0) == 3
        assert get_recommended_gpus(500.0) == 7

    def test_custom_gpu_memory(self):
        """Test with custom GPU memory size."""
        # With 40GB GPUs
        assert get_recommended_gpus(60.0, gpu_memory_gb=40.0) == 2
        assert get_recommended_gpus(100.0, gpu_memory_gb=40.0) == 3


class TestGetVramTier:
    """Tests for get_vram_tier function."""

    def test_consumer_tier(self):
        """Test consumer GPU tier."""
        tier = get_vram_tier(6.0)
        assert "consumer" in tier.lower()

    def test_prosumer_tier(self):
        """Test prosumer GPU tier."""
        tier = get_vram_tier(12.0)
        assert "prosumer" in tier.lower() or "3090" in tier or "4090" in tier

    def test_workstation_tier(self):
        """Test workstation GPU tier."""
        tier = get_vram_tier(20.0)
        assert "workstation" in tier.lower() or "4090" in tier or "A5000" in tier

    def test_datacenter_tier(self):
        """Test datacenter GPU tier."""
        tier = get_vram_tier(60.0)
        assert "datacenter" in tier.lower() or "A100" in tier or "H100" in tier

    def test_multi_gpu_tier(self):
        """Test multi-GPU tier."""
        tier = get_vram_tier(200.0)
        assert "multi" in tier.lower()
