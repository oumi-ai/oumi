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

from unittest import mock

import pytest

from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)
from oumi.training import Trainer
from oumi.training.trainer import _merge_params, _resolve_trainer_type


class TestResolveTrainerType:
    """Tests for the _resolve_trainer_type helper function."""

    def test_exact_match_lowercase(self):
        """Test exact match with lowercase enum value."""
        assert _resolve_trainer_type("trl_sft") == TrainerType.TRL_SFT
        assert _resolve_trainer_type("trl_dpo") == TrainerType.TRL_DPO
        assert _resolve_trainer_type("hf") == TrainerType.HF
        assert _resolve_trainer_type("oumi") == TrainerType.OUMI

    def test_exact_match_case_insensitive(self):
        """Test exact match is case-insensitive."""
        assert _resolve_trainer_type("TRL_SFT") == TrainerType.TRL_SFT
        assert _resolve_trainer_type("Trl_Dpo") == TrainerType.TRL_DPO
        assert _resolve_trainer_type("HF") == TrainerType.HF
        assert _resolve_trainer_type("OUMI") == TrainerType.OUMI

    def test_short_form_sft(self):
        """Test short form 'sft' resolves to TRL_SFT."""
        assert _resolve_trainer_type("sft") == TrainerType.TRL_SFT

    def test_short_form_dpo(self):
        """Test short form 'dpo' resolves to TRL_DPO."""
        assert _resolve_trainer_type("dpo") == TrainerType.TRL_DPO

    def test_short_form_kto(self):
        """Test short form 'kto' resolves to TRL_KTO."""
        assert _resolve_trainer_type("kto") == TrainerType.TRL_KTO

    def test_unknown_trainer_type_raises(self):
        """Test that unknown trainer type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown trainer_type"):
            _resolve_trainer_type("unknown")

    def test_ambiguous_grpo_raises(self):
        """Test that 'grpo' raises due to ambiguity (trl_grpo, verl_grpo)."""
        with pytest.raises(ValueError, match="Ambiguous trainer_type"):
            _resolve_trainer_type("grpo")


class TestMergeParams:
    """Tests for the _merge_params helper function."""

    def test_flat_overrides_take_precedence(self):
        """Test that flat overrides take precedence over config object."""
        base_params = TrainingParams(learning_rate=1e-4, num_train_epochs=3)
        overrides = {"learning_rate": 2e-4}
        result = _merge_params(TrainingParams, base_params, overrides)
        assert result.learning_rate == 2e-4
        assert result.num_train_epochs == 3

    def test_config_obj_values_used_when_no_override(self):
        """Test that config object values are used when no override."""
        base_params = TrainingParams(learning_rate=1e-4, max_steps=100)
        result = _merge_params(TrainingParams, base_params, {})
        assert result.learning_rate == 1e-4
        assert result.max_steps == 100

    def test_defaults_used_when_no_config_or_override(self):
        """Test that dataclass defaults are used when neither config nor override."""
        result = _merge_params(TrainingParams, None, {})
        # Check a default value
        assert result.num_train_epochs == 3  # Default from TrainingParams

    def test_none_values_in_overrides_ignored(self):
        """Test that None values in overrides are ignored."""
        base_params = TrainingParams(learning_rate=1e-4)
        overrides = {"learning_rate": None, "max_steps": 50}
        result = _merge_params(TrainingParams, base_params, overrides)
        assert result.learning_rate == 1e-4  # From base, not overridden
        assert result.max_steps == 50


class TestTrainerInit:
    """Tests for Trainer initialization."""

    def test_init_with_minimal_params(self):
        """Test initialization with minimal required parameters."""
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            dataset="test-dataset",
        )
        assert trainer.trainer_type == TrainerType.TRL_SFT
        assert trainer.config.model.model_name == "gpt2"
        assert len(trainer.config.data.train.datasets) == 1
        assert trainer.config.data.train.datasets[0].dataset_name == "test-dataset"

    def test_init_with_datasets_list(self):
        """Test initialization with multiple datasets."""
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            datasets=["dataset1", "dataset2"],
        )
        assert len(trainer.config.data.train.datasets) == 2
        assert trainer.config.data.train.datasets[0].dataset_name == "dataset1"
        assert trainer.config.data.train.datasets[1].dataset_name == "dataset2"

    def test_init_with_eval_dataset(self):
        """Test initialization with evaluation dataset."""
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            dataset="train-dataset",
            eval_dataset="eval-dataset",
        )
        assert len(trainer.config.data.validation.datasets) == 1
        assert trainer.config.data.validation.datasets[0].dataset_name == "eval-dataset"

    def test_init_both_dataset_and_datasets_raises(self):
        """Test that providing both dataset and datasets raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            Trainer(
                trainer_type="sft",
                model="gpt2",
                dataset="single",
                datasets=["list1", "list2"],
            )

    def test_init_no_model_raises(self):
        """Test that missing model raises ValueError."""
        with pytest.raises(ValueError, match="Either 'model' or 'model_params'"):
            Trainer(
                trainer_type="sft",
                dataset="test-dataset",
            )

    def test_init_no_dataset_raises(self):
        """Test that missing dataset raises ValueError."""
        with pytest.raises(ValueError, match="Either 'dataset', 'datasets'"):
            Trainer(
                trainer_type="sft",
                model="gpt2",
            )

    def test_init_with_training_params(self):
        """Test initialization with training parameters."""
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            dataset="test-dataset",
            learning_rate=2e-4,
            num_train_epochs=5,
            max_steps=100,
            per_device_train_batch_size=4,
            output_dir="/tmp/output",
        )
        assert trainer.config.training.learning_rate == 2e-4
        assert trainer.config.training.num_train_epochs == 5
        assert trainer.config.training.max_steps == 100
        assert trainer.config.training.per_device_train_batch_size == 4
        assert trainer.config.training.output_dir == "/tmp/output"

    def test_init_with_extended_params(self):
        """Test initialization with extended training parameters."""
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            dataset="test-dataset",
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            save_steps=100,
            logging_steps=10,
            eval_strategy="steps",
        )
        assert trainer.config.training.gradient_accumulation_steps == 4
        assert trainer.config.training.warmup_ratio == 0.1
        assert trainer.config.training.save_steps == 100
        assert trainer.config.training.logging_steps == 10
        assert trainer.config.training.eval_strategy == "steps"

    def test_init_use_peft(self):
        """Test initialization with use_peft parameter."""
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            dataset="test-dataset",
            use_peft=True,
        )
        assert trainer.config.training.use_peft is True


class TestTrainerWithConfigObjects:
    """Tests for Trainer with full config objects."""

    def test_init_with_model_params(self):
        """Test initialization with ModelParams object."""
        model_params = ModelParams(
            model_name="custom-model",
            model_max_length=1024,
        )
        trainer = Trainer(
            trainer_type="sft",
            model_params=model_params,
            dataset="test-dataset",
        )
        assert trainer.config.model.model_name == "custom-model"
        assert trainer.config.model.model_max_length == 1024

    def test_init_with_training_params_object(self):
        """Test initialization with TrainingParams object."""
        training_params = TrainingParams(
            learning_rate=3e-4,
            max_steps=200,
        )
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            training_params=training_params,
            dataset="test-dataset",
        )
        assert trainer.config.training.learning_rate == 3e-4
        assert trainer.config.training.max_steps == 200

    def test_init_with_data_params(self):
        """Test initialization with DataParams object."""
        data_params = DataParams(
            train=DatasetSplitParams(
                datasets=[
                    DatasetParams(dataset_name="custom-train"),
                ]
            ),
        )
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            data_params=data_params,
        )
        assert trainer.config.data.train.datasets[0].dataset_name == "custom-train"

    def test_flat_params_override_config_objects(self):
        """Test that flat params override config objects."""
        training_params = TrainingParams(learning_rate=1e-4)
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            training_params=training_params,
            dataset="test-dataset",
            learning_rate=5e-5,  # Should override
        )
        assert trainer.config.training.learning_rate == 5e-5

    def test_model_flat_param_overrides_model_params(self):
        """Test that model flat param overrides model_params."""
        model_params = ModelParams(model_name="base-model")
        trainer = Trainer(
            trainer_type="sft",
            model="override-model",  # Should override
            model_params=model_params,
            dataset="test-dataset",
        )
        assert trainer.config.model.model_name == "override-model"

    def test_init_with_training_config(self):
        """Test initialization with full TrainingConfig object."""
        training_config = TrainingConfig(
            model=ModelParams(model_name="config-model"),
            training=TrainingParams(learning_rate=1e-5),
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[DatasetParams(dataset_name="config-dataset")]
                )
            ),
        )
        trainer = Trainer(
            trainer_type="dpo",
            training_config=training_config,
        )
        assert trainer.config.model.model_name == "config-model"
        assert trainer.config.training.learning_rate == 1e-5
        assert trainer.config.data.train.datasets[0].dataset_name == "config-dataset"
        assert trainer.trainer_type == TrainerType.TRL_DPO

    def test_flat_params_override_training_config(self):
        """Test that flat params override training_config values."""
        training_config = TrainingConfig(
            model=ModelParams(model_name="config-model"),
            training=TrainingParams(learning_rate=1e-5),
            data=DataParams(
                train=DatasetSplitParams(
                    datasets=[DatasetParams(dataset_name="config-dataset")]
                )
            ),
        )
        trainer = Trainer(
            trainer_type="sft",
            training_config=training_config,
            model="override-model",  # Should override
            learning_rate=2e-4,  # Should override
        )
        assert trainer.config.model.model_name == "override-model"
        assert trainer.config.training.learning_rate == 2e-4


class TestTrainerTypes:
    """Tests for different trainer types."""

    @pytest.mark.parametrize(
        "trainer_type_str,expected_type",
        [
            ("sft", TrainerType.TRL_SFT),
            ("trl_sft", TrainerType.TRL_SFT),
            ("dpo", TrainerType.TRL_DPO),
            ("trl_dpo", TrainerType.TRL_DPO),
            ("kto", TrainerType.TRL_KTO),
            ("trl_kto", TrainerType.TRL_KTO),
            ("hf", TrainerType.HF),
            ("oumi", TrainerType.OUMI),
            ("trl_grpo", TrainerType.TRL_GRPO),
        ],
    )
    def test_trainer_type_resolution(self, trainer_type_str, expected_type):
        """Test that trainer types resolve correctly."""
        trainer = Trainer(
            trainer_type=trainer_type_str,
            model="gpt2",
            dataset="test-dataset",
        )
        assert trainer.trainer_type == expected_type
        assert trainer.config.training.trainer_type == expected_type

    def test_verl_grpo_trainer_type(self):
        """Test VERL_GRPO trainer type (requires eval dataset)."""
        trainer = Trainer(
            trainer_type="verl_grpo",
            model="gpt2",
            dataset="test-dataset",
            eval_dataset="eval-dataset",  # Required for VERL_GRPO
        )
        assert trainer.trainer_type == TrainerType.VERL_GRPO
        assert trainer.config.training.trainer_type == TrainerType.VERL_GRPO


class TestTrainerMethods:
    """Tests for Trainer methods."""

    def test_supported_trainer_types(self):
        """Test supported_trainer_types returns all trainer types."""
        supported = Trainer.supported_trainer_types()
        assert "trl_sft" in supported
        assert "trl_dpo" in supported
        assert "hf" in supported
        assert "oumi" in supported

    def test_config_property(self):
        """Test config property returns TrainingConfig."""
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            dataset="test-dataset",
        )
        assert isinstance(trainer.config, TrainingConfig)

    def test_repr(self):
        """Test __repr__ method."""
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            dataset="test-dataset",
        )
        repr_str = repr(trainer)
        assert "Trainer" in repr_str
        assert "trl_sft" in repr_str
        assert "gpt2" in repr_str

    def test_train_calls_underlying_function(self):
        """Test that train() calls the underlying train function."""
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            dataset="test-dataset",
            output_dir="/tmp/test",
        )

        with mock.patch("oumi.train.train") as mock_train:
            mock_train.return_value = {"loss": 0.5}
            result = trainer.train()

            mock_train.assert_called_once_with(trainer.config)
            assert result == {"loss": 0.5}

    def test_train_with_resume_from_checkpoint(self):
        """Test train() with resume_from_checkpoint parameter."""
        trainer = Trainer(
            trainer_type="sft",
            model="gpt2",
            dataset="test-dataset",
            output_dir="/tmp/test",
        )

        with mock.patch("oumi.train.train") as mock_train:
            trainer.train(resume_from_checkpoint="/path/to/checkpoint")

            # Verify the config was updated
            assert (
                trainer.config.training.resume_from_checkpoint == "/path/to/checkpoint"
            )
            mock_train.assert_called_once()
