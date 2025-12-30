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

"""Integration tests for DSPy-based optimizers.

These tests actually run the optimizers with small models and datasets
to verify end-to-end functionality.
"""

import os
from pathlib import Path

import pytest

from oumi.core.configs import InferenceEngineType, ModelParams
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.prompt_optimization_params import (
    PromptOptimizationParams,
)
from oumi.core.configs.prompt_config import PromptOptimizationConfig
from oumi.core.prompt_optimization import (
    BootstrapFewShotOptimizer,
    MiproOptimizer,
    get_metric_fn,
)

# Mark all tests in this module as requiring DSPy
pytestmark = pytest.mark.skipif(
    os.environ.get("OUMI_RUN_DSPY_INTEGRATION_TESTS") != "1",
    reason=(
        "DSPy integration tests disabled by default (slow and resource-intensive). "
        "Set OUMI_RUN_DSPY_INTEGRATION_TESTS=1 to enable."
    ),
)


@pytest.fixture
def small_dataset():
    """Create a small test dataset."""
    train_data = [
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is 3+3?", "output": "6"},
        {"input": "What is 4+4?", "output": "8"},
        {"input": "What is 5+5?", "output": "10"},
        {"input": "What is 6+6?", "output": "12"},
        {"input": "What is 7+7?", "output": "14"},
        {"input": "What is 8+8?", "output": "16"},
        {"input": "What is 9+9?", "output": "18"},
        {"input": "What is 10+10?", "output": "20"},
        {"input": "What is 11+11?", "output": "22"},
        {"input": "What is 12+12?", "output": "24"},
        {"input": "What is 13+13?", "output": "26"},
        {"input": "What is 14+14?", "output": "28"},
        {"input": "What is 15+15?", "output": "30"},
        {"input": "What is 16+16?", "output": "32"},
        {"input": "What is 17+17?", "output": "34"},
        {"input": "What is 18+18?", "output": "36"},
        {"input": "What is 19+19?", "output": "38"},
        {"input": "What is 20+20?", "output": "40"},
        {"input": "What is 21+21?", "output": "42"},
    ]

    val_data = [
        {"input": "What is 1+1?", "output": "2"},
        {"input": "What is 22+22?", "output": "44"},
        {"input": "What is 23+23?", "output": "46"},
        {"input": "What is 24+24?", "output": "48"},
        {"input": "What is 25+25?", "output": "50"},
        {"input": "What is 26+26?", "output": "52"},
        {"input": "What is 27+27?", "output": "54"},
        {"input": "What is 28+28?", "output": "56"},
        {"input": "What is 29+29?", "output": "58"},
        {"input": "What is 30+30?", "output": "60"},
    ]

    return train_data, val_data


@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration using a small model."""
    return PromptOptimizationConfig(
        model=ModelParams(
            model_name="HuggingFaceTB/SmolLM2-135M",
            trust_remote_code=True,
            torch_dtype_str="float32",  # Use float32 for CPU testing
            tokenizer_kwargs={"pad_token": "<|endoftext|>"},
        ),
        generation=GenerationParams(
            max_new_tokens=10,
            temperature=0.7,
            use_sampling=False,  # Greedy decoding for determinism
        ),
        optimization=PromptOptimizationParams(
            optimizer="bootstrap",
            num_trials=3,  # Very few trials for quick testing
            max_bootstrapped_demos=2,
            max_labeled_demos=5,
            verbose=True,
            enable_checkpointing=False,  # Disable for tests
        ),
        train_dataset_path=str(tmp_path / "train.jsonl"),
        val_dataset_path=str(tmp_path / "val.jsonl"),
        output_dir=str(tmp_path / "output"),
        metric="accuracy",
        engine=InferenceEngineType.NATIVE,
    )


class TestBootstrapFewShotOptimizer:
    """Tests for BootstrapFewShot optimizer."""

    @pytest.mark.e2e
    def test_bootstrap_basic_optimization(self, small_dataset, test_config):
        """Test that Bootstrap optimizer runs without errors."""
        train_data, val_data = small_dataset

        metric_fn = get_metric_fn("accuracy")
        optimizer = BootstrapFewShotOptimizer(test_config, metric_fn)

        result = optimizer.optimize(
            train_data, val_data, initial_prompt="Answer the question."
        )

        # Basic assertions
        assert result is not None
        assert result.final_score >= 0.0
        assert result.final_score <= 1.0
        assert result.num_trials > 0
        assert result.metadata["optimizer"] == "bootstrapfewshot"
        assert result.metadata["status"] == "completed"

    @pytest.mark.e2e
    def test_bootstrap_with_checkpointing(self, small_dataset, tmp_path):
        """Test Bootstrap with checkpointing enabled."""
        train_data, val_data = small_dataset

        config = PromptOptimizationConfig(
            model=ModelParams(
                model_name="HuggingFaceTB/SmolLM2-135M",
                trust_remote_code=True,
                torch_dtype_str="float32",
                tokenizer_kwargs={"pad_token": "<|endoftext|>"},
            ),
            generation=GenerationParams(max_new_tokens=10, temperature=0.7),
            optimization=PromptOptimizationParams(
                optimizer="bootstrap",
                num_trials=3,
                verbose=True,
                enable_checkpointing=True,
                checkpoint_interval=1,  # Save very frequently
            ),
            train_dataset_path=str(tmp_path / "train.jsonl"),
            output_dir=str(tmp_path / "output"),
            metric="accuracy",
            engine=InferenceEngineType.NATIVE,
        )

        metric_fn = get_metric_fn("accuracy")
        optimizer = BootstrapFewShotOptimizer(config, metric_fn)

        result = optimizer.optimize(train_data, val_data)

        assert result is not None
        # Check that checkpoint file was created
        checkpoint_path = Path(config.output_dir) / "checkpoint.json"
        assert checkpoint_path.exists() or result.metadata["status"] == "completed"


class TestMiproOptimizer:
    """Tests for MIPRO optimizer."""

    @pytest.mark.e2e_eternal
    def test_mipro_basic_optimization(self, small_dataset, tmp_path):
        """Test that MIPRO optimizer runs without errors."""
        train_data, val_data = small_dataset

        config = PromptOptimizationConfig(
            model=ModelParams(
                model_name="HuggingFaceTB/SmolLM2-135M",
                trust_remote_code=True,
                torch_dtype_str="float32",
                tokenizer_kwargs={"pad_token": "<|endoftext|>"},
            ),
            generation=GenerationParams(max_new_tokens=10, temperature=0.7),
            optimization=PromptOptimizationParams(
                optimizer="mipro",
                num_trials=2,  # Very few for quick testing
                max_bootstrapped_demos=2,
                verbose=True,
                enable_checkpointing=False,
            ),
            train_dataset_path=str(tmp_path / "train.jsonl"),
            output_dir=str(tmp_path / "output"),
            metric="accuracy",
            engine=InferenceEngineType.NATIVE,
        )

        metric_fn = get_metric_fn("accuracy")
        optimizer = MiproOptimizer(config, metric_fn)

        result = optimizer.optimize(train_data, val_data)

        assert result is not None
        assert result.final_score >= 0.0
        assert result.final_score <= 1.0
        assert result.metadata["optimizer"] == "miprov2"


class TestMetricsIntegration:
    """Tests for different metrics with optimizers."""

    @pytest.mark.e2e
    def test_f1_metric(self, small_dataset, test_config):
        """Test optimization with F1 metric."""
        train_data, val_data = small_dataset
        test_config.metric = "f1"

        metric_fn = get_metric_fn("f1")
        optimizer = BootstrapFewShotOptimizer(test_config, metric_fn)

        result = optimizer.optimize(train_data, val_data)

        assert result is not None
        assert result.final_score >= 0.0
        assert result.final_score <= 1.0

    @pytest.mark.e2e
    def test_custom_metric(self, small_dataset, test_config, tmp_path):
        """Test optimization with custom metric."""
        # Create a custom metric file
        metric_file = tmp_path / "custom_metric.py"
        metric_file.write_text("""
def metric_fn(predictions, references):
    '''Simple exact match metric.'''
    correct = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return correct / len(predictions) if predictions else 0.0
""")

        train_data, val_data = small_dataset
        test_config.metric = "custom"
        test_config.custom_metric_path = str(metric_file)

        metric_fn = get_metric_fn("custom", custom_metric_path=str(metric_file))
        optimizer = BootstrapFewShotOptimizer(test_config, metric_fn)

        result = optimizer.optimize(train_data, val_data)

        assert result is not None
        assert result.final_score >= 0.0


class TestErrorHandling:
    """Tests for error handling in optimizers."""

    def test_empty_dataset_error(self, test_config):
        """Test that empty datasets raise appropriate errors."""
        metric_fn = get_metric_fn("accuracy")
        optimizer = BootstrapFewShotOptimizer(test_config, metric_fn)

        with pytest.raises(RuntimeError, match="No training examples"):
            optimizer.optimize([], [{"input": "test", "output": "test"}])

    def test_too_small_dataset_warning(self, test_config):
        """Test that very small datasets produce warnings."""
        small_train = [
            {"input": "What is 1+1?", "output": "2"},
            {"input": "What is 2+2?", "output": "4"},
        ]
        small_val = [{"input": "What is 3+3?", "output": "6"}]

        metric_fn = get_metric_fn("accuracy")
        optimizer = BootstrapFewShotOptimizer(test_config, metric_fn)

        result = optimizer.optimize(small_train, small_val)
        assert result is not None


class TestCostTracking:
    """Tests for cost tracking functionality."""

    def test_cost_estimation(self, test_config):
        """Test that cost estimation works."""
        from oumi.core.prompt_optimization.cost_tracking import (
            estimate_optimization_cost,
        )

        estimate = estimate_optimization_cost(
            model_name=test_config.model.model_name,
            num_train_examples=20,
            num_val_examples=10,
            num_trials=test_config.optimization.num_trials,
            optimizer=test_config.optimization.optimizer,
        )

        assert estimate is not None
        assert estimate.estimated_input_tokens > 0
        assert estimate.estimated_output_tokens > 0
        assert estimate.num_train_examples == 20
        assert estimate.num_val_examples == 10

    def test_cost_warning_threshold(self):
        """Test cost warning thresholds."""
        from oumi.core.prompt_optimization.cost_tracking import (
            CostEstimate,
            should_warn_about_cost,
        )

        # Low cost - no warning
        low_cost = CostEstimate(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            estimated_total_cost=2.0,
            num_train_examples=20,
            num_val_examples=10,
            num_trials=5,
            model_name="test-model",
            notes=[],
        )

        should_warn, msg = should_warn_about_cost(low_cost, threshold=10.0)
        assert not should_warn

        # High cost - warning
        high_cost = CostEstimate(
            estimated_input_tokens=1000000,
            estimated_output_tokens=500000,
            estimated_total_cost=50.0,
            num_train_examples=200,
            num_val_examples=100,
            num_trials=50,
            model_name="gpt-4",
            notes=[],
        )

        should_warn, msg = should_warn_about_cost(high_cost, threshold=10.0)
        assert should_warn
        assert msg is not None
        assert "$" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
