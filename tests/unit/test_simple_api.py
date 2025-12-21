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

"""Tests for the simplified API functions (chat, train, evaluate, judge)."""

from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import (
    EvaluationConfig,
    InferenceEngineType,
    TrainerType,
    TrainingConfig,
)
from oumi.core.configs.params.judge_params import JudgeOutputType
from oumi.judge import CRITERIA_TEMPLATES


class TestCriteriaTemplates:
    """Tests for judge criteria templates."""

    def test_truthfulness_template_exists(self):
        """Test truthfulness criteria template."""
        assert "truthfulness" in CRITERIA_TEMPLATES
        template = CRITERIA_TEMPLATES["truthfulness"]
        assert "prompt_template" in template
        assert "{request}" in template["prompt_template"]
        assert "{response}" in template["prompt_template"]
        assert template["judgment_type"] == JudgeOutputType.BOOL

    def test_helpfulness_template_exists(self):
        """Test helpfulness criteria template."""
        assert "helpfulness" in CRITERIA_TEMPLATES
        template = CRITERIA_TEMPLATES["helpfulness"]
        assert "prompt_template" in template
        assert "{request}" in template["prompt_template"]
        assert "{response}" in template["prompt_template"]
        assert template["judgment_type"] == JudgeOutputType.INT

    def test_safety_template_exists(self):
        """Test safety criteria template."""
        assert "safety" in CRITERIA_TEMPLATES
        template = CRITERIA_TEMPLATES["safety"]
        assert "prompt_template" in template
        assert template["judgment_type"] == JudgeOutputType.BOOL

    def test_relevance_template_exists(self):
        """Test relevance criteria template."""
        assert "relevance" in CRITERIA_TEMPLATES
        template = CRITERIA_TEMPLATES["relevance"]
        assert "prompt_template" in template
        assert template["judgment_type"] == JudgeOutputType.BOOL

    def test_coherence_template_exists(self):
        """Test coherence criteria template."""
        assert "coherence" in CRITERIA_TEMPLATES
        template = CRITERIA_TEMPLATES["coherence"]
        assert "prompt_template" in template
        assert template["judgment_type"] == JudgeOutputType.BOOL

    def test_all_templates_have_include_explanation(self):
        """Test all templates have include_explanation field."""
        for name, template in CRITERIA_TEMPLATES.items():
            assert "include_explanation" in template, f"Missing include_explanation in {name}"
            assert template["include_explanation"] is True


class TestTrainingConfigFactoryMethods:
    """Tests for TrainingConfig factory methods."""

    def test_for_sft_creates_valid_config(self):
        """Test TrainingConfig.for_sft creates a valid config."""
        config = TrainingConfig.for_sft(
            model="meta-llama/Llama-3.1-8B",
            dataset="tatsu-lab/alpaca",
        )
        assert isinstance(config, TrainingConfig)
        assert config.model.model_name == "meta-llama/Llama-3.1-8B"
        assert config.training.trainer_type == TrainerType.TRL_SFT
        assert len(config.data.train.datasets) == 1
        assert config.data.train.datasets[0].dataset_name == "tatsu-lab/alpaca"

    def test_for_dpo_creates_valid_config(self):
        """Test TrainingConfig.for_dpo creates a valid config."""
        config = TrainingConfig.for_dpo(
            model="meta-llama/Llama-3.1-8B",
            dataset="my-preference-dataset",
        )
        assert isinstance(config, TrainingConfig)
        assert config.model.model_name == "meta-llama/Llama-3.1-8B"
        assert config.training.trainer_type == TrainerType.TRL_DPO

    def test_for_grpo_creates_valid_config(self):
        """Test TrainingConfig.for_grpo creates a valid config."""
        config = TrainingConfig.for_grpo(
            model="meta-llama/Llama-3.1-8B",
            dataset="my-grpo-dataset",
        )
        assert isinstance(config, TrainingConfig)
        assert config.training.trainer_type == TrainerType.TRL_GRPO

    def test_for_method_with_custom_params(self):
        """Test for_method with custom training parameters."""
        config = TrainingConfig.for_method(
            method="sft",
            model="meta-llama/Llama-3.1-8B",
            dataset="tatsu-lab/alpaca",
            epochs=5,
            batch_size=8,
            learning_rate=1e-5,
        )
        assert config.training.num_train_epochs == 5
        assert config.training.per_device_train_batch_size == 8
        assert config.training.learning_rate == 1e-5

    def test_for_method_with_peft_disabled(self):
        """Test for_method with PEFT disabled."""
        config = TrainingConfig.for_method(
            method="sft",
            model="meta-llama/Llama-3.1-8B",
            dataset="tatsu-lab/alpaca",
            use_peft=False,
        )
        assert config.training.use_peft is False

    def test_for_method_with_custom_lora_params(self):
        """Test for_method with custom LoRA parameters."""
        config = TrainingConfig.for_method(
            method="sft",
            model="meta-llama/Llama-3.1-8B",
            dataset="tatsu-lab/alpaca",
            use_peft=True,
            lora_r=32,
            lora_alpha=64,
        )
        assert config.peft.lora_r == 32
        assert config.peft.lora_alpha == 64

    def test_for_method_unknown_method_raises(self):
        """Test for_method raises ValueError for unknown method."""
        with pytest.raises(ValueError, match="Unknown training method"):
            TrainingConfig.for_method(
                method="unknown",
                model="meta-llama/Llama-3.1-8B",
                dataset="test",
            )


class TestJudgeFunctionValidation:
    """Tests for judge function input validation."""

    def test_judge_raises_without_criteria_or_prompt(self):
        """Test judge raises ValueError when neither criteria nor prompt is provided."""
        from oumi.judge import judge

        with pytest.raises(ValueError, match="Either 'criteria' or 'prompt_template'"):
            judge(
                "gpt-4o",
                [{"request": "test", "response": "test"}],
            )

    def test_judge_raises_for_unknown_criteria(self):
        """Test judge raises ValueError for unknown criteria."""
        from oumi.judge import judge

        with pytest.raises(ValueError, match="Unknown criteria"):
            judge(
                "gpt-4o",
                [{"request": "test", "response": "test"}],
                criteria="unknown_criteria",
            )


class TestChatFunctionValidation:
    """Tests for chat function input validation."""

    def test_chat_raises_without_message_or_conversation(self):
        """Test chat raises ValueError when no input is provided."""
        from oumi.infer import chat

        with pytest.raises(
            ValueError, match="At least one of 'message', 'messages', or 'conversation'"
        ):
            chat("gpt-4o")


class TestEvaluateFunctionValidation:
    """Tests for evaluate function input validation."""

    def test_evaluate_raises_without_tasks_in_simple_mode(self):
        """Test evaluate raises ValueError when tasks not provided in simple mode."""
        from oumi.evaluate import evaluate

        with pytest.raises(ValueError, match="'tasks' must be provided"):
            evaluate("meta-llama/Llama-3.1-8B")


class TestEngineCaching:
    """Tests for inference engine caching."""

    def test_clear_engine_cache(self):
        """Test clear_engine_cache clears the cache."""
        from oumi.infer import _ENGINE_CACHE, clear_engine_cache

        # Add something to cache
        _ENGINE_CACHE["test:model"] = MagicMock()
        assert len(_ENGINE_CACHE) > 0

        # Clear cache
        clear_engine_cache()
        assert len(_ENGINE_CACHE) == 0


class TestModuleExports:
    """Tests for module exports."""

    def test_chat_exported_from_oumi(self):
        """Test chat function is exported from oumi."""
        import oumi

        assert hasattr(oumi, "chat")
        assert callable(oumi.chat)

    def test_judge_function_importable(self):
        """Test judge function can be imported from oumi.judge module."""
        from oumi.judge import judge

        assert callable(judge)

    def test_judge_dataset_exported_from_oumi(self):
        """Test judge_dataset function is exported from oumi."""
        import oumi

        assert hasattr(oumi, "judge_dataset")
        assert callable(oumi.judge_dataset)

    def test_train_exported_from_oumi(self):
        """Test train function is exported from oumi."""
        import oumi

        assert hasattr(oumi, "train")
        # Note: oumi.train may be shadowed by the module, so we also test direct import
        from oumi.train import train

        assert callable(train)

    def test_evaluate_function_importable(self):
        """Test evaluate function can be imported from oumi.evaluate module."""
        from oumi.evaluate import evaluate

        assert callable(evaluate)

    def test_all_contains_new_functions(self):
        """Test __all__ contains new simplified API functions."""
        import oumi

        assert "chat" in oumi.__all__
        assert "judge" in oumi.__all__
        assert "judge_dataset" in oumi.__all__
