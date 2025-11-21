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

"""Integration tests for vLLM backend in Megatron GRPO trainer.

These tests verify:
- vLLM engine initialization
- Weight export to vLLM format
- Generation with vLLM backend
- Weight synchronization during training
"""

import tempfile
from pathlib import Path
from typing import Callable

import pytest
import torch
from datasets import Dataset

# Check for both Megatron and vLLM availability
try:
    import megatron.bridge  # noqa: F401
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False

try:
    import vllm  # noqa: F401
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

if MEGATRON_AVAILABLE:
    from oumi.core.configs import (
        DataParams,
        ModelParams,
        PeftParams,
        TrainerType,
        TrainingConfig,
        TrainingParams,
    )
    from oumi.core.configs.params.megatron_params import (
        MegatronGRPOConfig,
        MegatronParams,
        MegatronSamplingConfig,
    )
    from oumi.core.trainers.megatron.megatron_grpo_trainer import OumiMegatronGrpoTrainer

pytestmark = [
    pytest.mark.skipif(
        not MEGATRON_AVAILABLE or not VLLM_AVAILABLE,
        reason="Requires both Megatron-Bridge and vLLM"
    ),
    pytest.mark.integration,
    pytest.mark.slow,
]


@pytest.fixture
def simple_reward_function() -> Callable:
    """Simple reward function for testing."""
    def reward_fn(batch: dict, completions: list[str]) -> torch.Tensor:
        rewards = torch.tensor([len(c) / 100.0 for c in completions], dtype=torch.float32)
        return rewards
    return reward_fn


@pytest.fixture
def dummy_dataset() -> Dataset:
    """Create a small dummy dataset."""
    data = {
        "input_ids": [[1, 2, 3, 4, 5]] * 5,
        "attention_mask": [[1, 1, 1, 1, 1]] * 5,
    }
    return Dataset.from_dict(data)


@pytest.fixture
def vllm_config(tmp_path: Path) -> TrainingConfig:
    """Create configuration with vLLM backend."""
    return TrainingConfig(
        model=ModelParams(
            model_name="HuggingFaceTB/SmolLM-135M",
            trust_remote_code=False,
            torch_dtype_str="float32",
        ),
        training=TrainingParams(
            trainer_type=TrainerType.MEGATRON_GRPO,
            output_dir=str(tmp_path / "output"),
            max_steps=2,
            per_device_train_batch_size=1,
            learning_rate=1e-4,
            save_steps=1,
            eval_steps=1,
        ),
        data=DataParams(dataset_name="dummy"),
        peft=PeftParams(),
        megatron=MegatronParams(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            micro_batch_size=1,
            inference_backend="vllm",  # Use vLLM backend
            vllm_weight_sync_interval=1,  # Sync every step for testing
            sampling_config=MegatronSamplingConfig(
                temperature=0.7,
                top_p=0.9,
                max_tokens=10,
            ),
            grpo_config=MegatronGRPOConfig(
                kl_beta=0.001,
                max_prompt_length=64,
                max_completion_length=32,
            ),
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM requires CUDA")
class TestVLLMIntegration:
    """Test vLLM backend integration."""

    def test_vllm_engine_initialization(
        self,
        vllm_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that vLLM engine can be initialized."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(vllm_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=vllm_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # Verify vLLM weights were exported during init
        assert trainer._vllm_model_path is not None
        assert trainer._vllm_model_path.exists()

        # Verify vLLM engine starts as None (lazy init)
        assert trainer._vllm_engine is None

    def test_vllm_generation(
        self,
        vllm_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test generation using vLLM backend."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(vllm_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=vllm_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        # Generate completions (should initialize vLLM engine on first call)
        completions, generated_ids, logprobs = trainer._generate_with_vllm(batch)

        # Verify vLLM engine was initialized
        assert trainer._vllm_engine is not None

        # Verify generation outputs
        assert len(completions) == 1
        assert isinstance(completions[0], str)
        assert generated_ids.shape[0] == 1
        assert logprobs.shape[0] == 1

    def test_vllm_weight_export(
        self,
        vllm_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that model weights are exported to HF format for vLLM."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(vllm_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=vllm_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        vllm_model_path = trainer._vllm_model_path

        # Verify HF model files were exported
        assert (vllm_model_path / "config.json").exists()
        assert (vllm_model_path / "tokenizer_config.json").exists()

        # Should have either safetensors or pytorch_model.bin
        has_weights = (
            (vllm_model_path / "model.safetensors").exists() or
            (vllm_model_path / "pytorch_model.bin").exists()
        )
        assert has_weights

    def test_vllm_weight_sync(
        self,
        vllm_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that weights are synced to vLLM during training."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(vllm_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=vllm_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # Run one training step to initialize vLLM
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        trainer._training_step(batch, 0)

        initial_sync_step = trainer._last_vllm_sync_step

        # Run another step - should trigger sync (interval=1)
        trainer._global_step = 1
        trainer._training_step(batch, 1)

        # Verify sync occurred
        assert trainer._last_vllm_sync_step > initial_sync_step

    def test_vllm_with_tensor_parallelism(
        self,
        vllm_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test vLLM with tensor parallelism enabled."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires 2+ GPUs for TP=2")

        # Enable TP=2
        vllm_config.megatron.tensor_model_parallel_size = 2

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(vllm_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=vllm_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # vLLM engine should be initialized with TP=2
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        completions, _, _ = trainer._generate_with_vllm(batch)
        assert len(completions) == 1

    def test_vllm_fallback_on_error(
        self,
        vllm_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that training handles vLLM errors gracefully."""
        from transformers import AutoTokenizer

        # Set invalid vLLM config to trigger error
        vllm_config.megatron.sampling_config.max_tokens = -1  # Invalid

        tokenizer = AutoTokenizer.from_pretrained(vllm_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=vllm_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        # Should raise error with invalid config
        with pytest.raises(Exception):
            trainer._generate_with_vllm(batch)


class TestVLLMMemoryManagement:
    """Test memory management for vLLM integration."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_vllm_engine_cleanup(
        self,
        vllm_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that old vLLM engines are cleaned up during weight sync."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(vllm_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=vllm_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # Initialize vLLM engine
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        trainer._generate_with_vllm(batch)

        initial_engine = trainer._vllm_engine
        assert initial_engine is not None

        # Trigger weight sync
        trainer._sync_weights_to_vllm()

        # Old engine should be replaced
        assert trainer._vllm_engine is None or trainer._vllm_engine is not initial_engine

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_vllm_memory_leak_prevention(
        self,
        vllm_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that repeated weight syncs don't cause memory leaks."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(vllm_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=vllm_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        # Initialize vLLM
        trainer._generate_with_vllm(batch)

        # Record initial memory usage
        initial_memory = torch.cuda.memory_allocated()

        # Sync weights multiple times
        for _ in range(3):
            trainer._sync_weights_to_vllm()
            trainer._generate_with_vllm(batch)

        # Memory usage should not grow unboundedly
        final_memory = torch.cuda.memory_allocated()

        # Allow some growth but not proportional to number of syncs
        # (Should be < 2x initial, not > 4x)
        assert final_memory < initial_memory * 2.5
