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

"""Integration tests for Megatron GRPO trainer.

These tests verify end-to-end functionality including:
- Basic training loop execution
- Checkpoint save and load
- Sequence packing
- Evaluation
- Multi-GPU distributed training (when available)
"""

import os
import tempfile
from pathlib import Path
from typing import Callable

import pytest
import torch
from datasets import Dataset

# Skip all tests if Megatron-Bridge is not available
try:
    import megatron.bridge  # noqa: F401
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False

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
    from oumi.core.tokenizers import BaseTokenizer
    from oumi.core.trainers.megatron.megatron_grpo_trainer import OumiMegatronGrpoTrainer

pytestmark = [
    pytest.mark.skipif(not MEGATRON_AVAILABLE, reason="Megatron-Bridge not installed"),
    pytest.mark.integration,
]


@pytest.fixture
def simple_reward_function() -> Callable:
    """Simple reward function that returns length-based rewards."""
    def reward_fn(batch: dict, completions: list[str]) -> torch.Tensor:
        # Reward based on completion length (simple heuristic)
        rewards = torch.tensor([len(c) / 100.0 for c in completions], dtype=torch.float32)
        return rewards
    return reward_fn


@pytest.fixture
def dummy_dataset() -> Dataset:
    """Create a small dummy dataset for testing."""
    data = {
        "input_ids": [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3],
        ] * 10,  # Repeat to have more samples
        "attention_mask": [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1],
        ] * 10,
    }
    return Dataset.from_dict(data)


@pytest.fixture
def test_config(tmp_path: Path) -> TrainingConfig:
    """Create a minimal test configuration for Megatron GRPO."""
    return TrainingConfig(
        model=ModelParams(
            model_name="HuggingFaceTB/SmolLM-135M",  # Small model for testing
            trust_remote_code=False,
            torch_dtype_str="float32",
        ),
        training=TrainingParams(
            trainer_type=TrainerType.MEGATRON_GRPO,
            output_dir=str(tmp_path / "output"),
            max_steps=3,  # Very short for testing
            per_device_train_batch_size=2,
            learning_rate=1e-4,
            save_steps=2,
            eval_steps=2,
            logging_steps=1,
            save_final_model=True,
        ),
        data=DataParams(
            dataset_name="dummy",
        ),
        peft=PeftParams(),
        megatron=MegatronParams(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            micro_batch_size=1,
            global_batch_size=2,
            inference_backend="megatron",  # Use Megatron for CI (no vLLM)
            sampling_config=MegatronSamplingConfig(
                temperature=0.7,
                top_p=0.9,
                max_tokens=10,  # Short completions for testing
            ),
            grpo_config=MegatronGRPOConfig(
                kl_beta=0.001,
                max_prompt_length=64,
                max_completion_length=32,
            ),
        ),
    )


class TestMegatronGRPOBasicTraining:
    """Test basic training functionality."""

    def test_trainer_initialization(
        self,
        test_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that trainer can be initialized without errors."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(test_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        assert trainer is not None
        assert trainer._model is not None
        assert trainer._ref_model is not None
        assert trainer._optimizer is not None
        assert trainer._scheduler is not None

    def test_single_training_step(
        self,
        test_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that a single training step executes without errors."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(test_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # Get a batch
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        }

        # Run one training step
        metrics = trainer._training_step(batch, batch_idx=0)

        # Verify metrics are returned
        assert "loss" in metrics
        assert "kl" in metrics
        assert "rewards_mean" in metrics
        assert isinstance(metrics["loss"], float)

    def test_full_training_loop(
        self,
        test_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that full training loop completes successfully."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(test_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # Run training (should complete 3 steps)
        trainer.train()

        # Verify training completed
        assert trainer._global_step == 3


class TestMegatronGRPOCheckpointing:
    """Test checkpoint save and load functionality."""

    def test_checkpoint_save(
        self,
        test_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that checkpoints are saved correctly."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(test_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # Save checkpoint
        trainer.save_model(test_config, final=False)

        # Verify checkpoint directory exists
        output_dir = Path(test_config.training.output_dir)
        checkpoint_dir = output_dir / "checkpoints" / "step_0"
        assert checkpoint_dir.exists()

        # Verify checkpoint contains required files
        # Megatron checkpoints have model_base, optimizer, scheduler subdirectories
        assert (checkpoint_dir / "model_base").exists() or any(checkpoint_dir.glob("*"))

    def test_checkpoint_resume(
        self,
        test_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that training can resume from checkpoint."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(test_config.model.model_name)

        # First training session
        trainer1 = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # Train for 2 steps and save
        test_config.training.max_steps = 2
        trainer1.train()
        trainer1.save_model(test_config, final=False)

        initial_step = trainer1._global_step

        # Second training session - resume from checkpoint
        trainer2 = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # Resume training
        output_dir = Path(test_config.training.output_dir)
        checkpoint_path = output_dir / "checkpoints" / f"step_{initial_step}"

        trainer2._load_checkpoint(str(checkpoint_path))

        # Verify state was restored
        assert trainer2._global_step == initial_step


class TestMegatronGRPOSequencePacking:
    """Test sequence packing functionality."""

    def test_sequence_packing_enabled(
        self,
        test_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test training with sequence packing enabled."""
        from transformers import AutoTokenizer

        # Enable sequence packing
        test_config.megatron.enable_sequence_packing = True

        tokenizer = AutoTokenizer.from_pretrained(test_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # Verify packing is enabled
        assert trainer._use_sequence_packing is True

        # Create batch with variable-length sequences
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 0, 0],
                [1, 2, 3, 4, 5],
                [1, 2, 0, 0, 0],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0],
            ]),
        }

        # Run training step with packing
        metrics = trainer._training_step(batch, batch_idx=0)

        # Verify metrics include packing info
        assert "num_bins" in metrics
        assert metrics["num_bins"] >= 1

    def test_sequence_packing_correctness(
        self,
        test_config: TrainingConfig,
        simple_reward_function: Callable,
    ):
        """Test that sequence packing produces correct results."""
        from oumi.core.trainers.megatron.sequence_packing import (
            compute_packing_efficiency,
            pack_sequences,
        )

        # Create batch with known sequences
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 0, 0, 0],
                [4, 5, 6, 7, 0, 0],
                [8, 9, 0, 0, 0, 0],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0],
            ]),
        }

        # Pack sequences
        packed_batches = pack_sequences(batch, max_bin_size=10, pad_token_id=0)

        # Verify packing occurred
        assert len(packed_batches) >= 1

        # Check efficiency
        efficiency = compute_packing_efficiency(packed_batches, max_bin_size=10)
        assert efficiency["total_sequences"] == 3
        assert efficiency["utilization"] > 0.0


class TestMegatronGRPOEvaluation:
    """Test evaluation functionality."""

    def test_evaluation_execution(
        self,
        test_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that evaluation runs successfully."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(test_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # Run evaluation
        metrics = trainer._evaluate()

        # Verify eval metrics are returned
        assert "eval/reward_mean" in metrics
        assert "eval/reward_std" in metrics
        assert "eval/completion_length_mean" in metrics
        assert "eval/total_samples" in metrics
        assert metrics["eval/total_samples"] > 0

    def test_evaluation_without_dataset(
        self,
        test_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test that evaluation handles missing eval dataset gracefully."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(test_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=None,  # No eval dataset
        )

        # Run evaluation - should return empty dict
        metrics = trainer._evaluate()
        assert metrics == {}


class TestMegatronGRPORewardFunctions:
    """Test reward function handling."""

    def test_reward_function_validation(
        self,
        test_config: TrainingConfig,
        dummy_dataset: Dataset,
    ):
        """Test that reward function output is validated."""
        from transformers import AutoTokenizer

        def bad_reward_fn(batch: dict, completions: list[str]) -> list:
            # Returns list instead of tensor - should fail
            return [1.0] * len(completions)

        tokenizer = AutoTokenizer.from_pretrained(test_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[bad_reward_fn],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        # This should auto-convert list to tensor
        completions = ["test completion"]
        rewards = trainer._compute_rewards(batch, completions)
        assert isinstance(rewards, torch.Tensor)

    def test_invalid_reward_shape(
        self,
        test_config: TrainingConfig,
        dummy_dataset: Dataset,
    ):
        """Test that invalid reward shapes are caught."""
        from transformers import AutoTokenizer

        def bad_shape_reward_fn(batch: dict, completions: list[str]) -> torch.Tensor:
            # Returns wrong shape
            return torch.tensor([1.0])  # Should be len(completions)

        tokenizer = AutoTokenizer.from_pretrained(test_config.model.model_name)

        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[bad_shape_reward_fn],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        completions = ["completion 1", "completion 2"]

        # Should raise ValueError for shape mismatch
        with pytest.raises(ValueError, match="returned.*rewards but expected"):
            trainer._compute_rewards(batch, completions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestMegatronGRPODistributed:
    """Test distributed training functionality.

    Note: These tests require multiple GPUs and are skipped in single-GPU environments.
    """

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2+ GPUs")
    def test_tensor_parallelism_init(
        self,
        test_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test initialization with tensor parallelism."""
        # Modify config for TP=2
        test_config.megatron.tensor_model_parallel_size = 2

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(test_config.model.model_name)

        # This should initialize with TP=2
        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        assert trainer is not None

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2+ GPUs")
    def test_pipeline_parallelism_init(
        self,
        test_config: TrainingConfig,
        dummy_dataset: Dataset,
        simple_reward_function: Callable,
    ):
        """Test initialization with pipeline parallelism."""
        # Modify config for PP=2
        test_config.megatron.pipeline_model_parallel_size = 2

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(test_config.model.model_name)

        # This should initialize with PP=2
        trainer = OumiMegatronGrpoTrainer(
            processing_class=tokenizer,
            config=test_config,
            reward_funcs=[simple_reward_function],
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        assert trainer is not None
