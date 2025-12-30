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

"""Unit tests for cost tracking functionality."""

import pytest

from oumi.core.prompt_optimization.cost_tracking import (
    CostEstimate,
    estimate_optimization_cost,
    get_model_costs,
    should_warn_about_cost,
)


class TestGetModelCosts:
    """Tests for get_model_costs function."""

    def test_exact_match(self):
        """Test exact model name match."""
        costs = get_model_costs("gpt-4")
        assert costs["input"] == 30.0
        assert costs["output"] == 60.0

    def test_unknown_model(self):
        """Test unknown model returns default (free)."""
        costs = get_model_costs("my-custom-local-model")
        assert costs["input"] == 0.0
        assert costs["output"] == 0.0

    def test_claude_models(self):
        """Test Claude model costs."""
        opus_costs = get_model_costs("claude-3-opus")
        assert opus_costs["input"] > 0
        assert opus_costs["output"] > opus_costs["input"]

        haiku_costs = get_model_costs("claude-3-haiku")
        assert haiku_costs["input"] > 0
        assert haiku_costs["input"] < opus_costs["input"]


class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_cost_estimate_creation(self):
        """Test creating a cost estimate."""
        estimate = CostEstimate(
            estimated_input_tokens=100000,
            estimated_output_tokens=50000,
            estimated_total_cost=10.50,
            num_train_examples=100,
            num_val_examples=20,
            num_trials=30,
            model_name="gpt-3.5-turbo",
            notes=["Test note"],
        )

        assert estimate.estimated_input_tokens == 100000
        assert estimate.estimated_output_tokens == 50000
        assert estimate.estimated_total_cost == 10.50
        assert len(estimate.notes) == 1

    def test_cost_estimate_to_dict(self):
        """Test converting estimate to dictionary."""
        estimate = CostEstimate(
            estimated_input_tokens=100000,
            estimated_output_tokens=50000,
            estimated_total_cost=10.50,
            num_train_examples=100,
            num_val_examples=20,
            num_trials=30,
            model_name="gpt-3.5-turbo",
            notes=[],
        )

        data = estimate.to_dict()
        assert data["estimated_input_tokens"] == 100000
        assert data["estimated_total_tokens"] == 150000
        assert data["estimated_total_cost_usd"] == 10.50
        assert data["model_name"] == "gpt-3.5-turbo"


class TestEstimateOptimizationCost:
    """Tests for estimate_optimization_cost function."""

    def test_mipro_cost_estimation(self):
        """Test cost estimation for MIPRO optimizer."""
        estimate = estimate_optimization_cost(
            model_name="gpt-3.5-turbo",
            num_train_examples=100,
            num_val_examples=20,
            num_trials=30,
            optimizer="mipro",
        )

        assert estimate is not None
        assert estimate.estimated_input_tokens > 0
        assert estimate.estimated_output_tokens > 0
        assert estimate.estimated_total_cost > 0
        assert estimate.model_name == "gpt-3.5-turbo"
        assert len(estimate.notes) > 0

    def test_bootstrap_cost_estimation(self):
        """Test cost estimation for Bootstrap optimizer."""
        estimate = estimate_optimization_cost(
            model_name="gpt-3.5-turbo",
            num_train_examples=100,
            num_val_examples=20,
            num_trials=10,
            optimizer="bootstrap",
        )

        assert estimate is not None
        assert estimate.estimated_input_tokens > 0
        # Bootstrap should be cheaper than MIPRO
        mipro_estimate = estimate_optimization_cost(
            model_name="gpt-3.5-turbo",
            num_train_examples=100,
            num_val_examples=20,
            num_trials=10,
            optimizer="mipro",
        )
        assert estimate.estimated_total_cost <= mipro_estimate.estimated_total_cost

    def test_gepa_cost_estimation(self):
        """Test cost estimation for GEPA optimizer."""
        estimate = estimate_optimization_cost(
            model_name="gpt-4",
            num_train_examples=50,
            num_val_examples=10,
            num_trials=20,
            optimizer="gepa",
        )

        assert estimate is not None
        assert estimate.estimated_input_tokens > 0
        assert estimate.estimated_total_cost > 0

    def test_local_model_cost(self):
        """Test cost estimation for local model (should be $0)."""
        estimate = estimate_optimization_cost(
            model_name="HuggingFaceTB/SmolLM2-135M",
            num_train_examples=100,
            num_val_examples=20,
            num_trials=30,
            optimizer="mipro",
        )

        assert estimate.estimated_total_cost == 0.0

    def test_small_dataset_warning(self):
        """Test that small datasets generate warnings."""
        estimate = estimate_optimization_cost(
            model_name="gpt-3.5-turbo",
            num_train_examples=30,  # Small dataset
            num_val_examples=10,
            num_trials=10,
            optimizer="bootstrap",
        )

        # Should have a warning about small dataset
        assert any("small" in note.lower() for note in estimate.notes)

    def test_different_num_trials(self):
        """Test that more trials increases cost."""
        estimate_10 = estimate_optimization_cost(
            model_name="gpt-3.5-turbo",
            num_train_examples=100,
            num_val_examples=20,
            num_trials=10,
            optimizer="mipro",
        )

        estimate_50 = estimate_optimization_cost(
            model_name="gpt-3.5-turbo",
            num_train_examples=100,
            num_val_examples=20,
            num_trials=50,
            optimizer="mipro",
        )

        assert estimate_50.estimated_total_cost > estimate_10.estimated_total_cost


class TestShouldWarnAboutCost:
    """Tests for should_warn_about_cost function."""

    def test_no_warning_below_threshold(self):
        """Test no warning for costs below threshold."""
        estimate = CostEstimate(
            estimated_input_tokens=10000,
            estimated_output_tokens=5000,
            estimated_total_cost=2.0,
            num_train_examples=50,
            num_val_examples=10,
            num_trials=10,
            model_name="gpt-3.5-turbo",
            notes=[],
        )

        should_warn, msg = should_warn_about_cost(estimate, threshold=10.0)
        assert not should_warn
        assert msg is None

    def test_warning_above_threshold(self):
        """Test warning for costs above threshold."""
        estimate = CostEstimate(
            estimated_input_tokens=1000000,
            estimated_output_tokens=500000,
            estimated_total_cost=50.0,
            num_train_examples=200,
            num_val_examples=50,
            num_trials=100,
            model_name="gpt-4",
            notes=[],
        )

        should_warn, msg = should_warn_about_cost(estimate, threshold=10.0)
        assert should_warn
        assert msg is not None
        assert "$50.00" in msg
        assert "$10.00" in msg

    def test_no_warning_free_model(self):
        """Test no warning for free/local models."""
        estimate = CostEstimate(
            estimated_input_tokens=1000000,
            estimated_output_tokens=500000,
            estimated_total_cost=0.0,
            num_train_examples=200,
            num_val_examples=50,
            num_trials=100,
            model_name="local-model",
            notes=[],
        )

        should_warn, msg = should_warn_about_cost(estimate, threshold=10.0)
        assert not should_warn

    def test_custom_threshold(self):
        """Test custom warning threshold."""
        estimate = CostEstimate(
            estimated_input_tokens=100000,
            estimated_output_tokens=50000,
            estimated_total_cost=15.0,
            num_train_examples=100,
            num_val_examples=20,
            num_trials=50,
            model_name="gpt-3.5-turbo",
            notes=[],
        )

        # Should warn with low threshold
        should_warn_low, _ = should_warn_about_cost(estimate, threshold=5.0)
        assert should_warn_low

        # Should not warn with high threshold
        should_warn_high, _ = should_warn_about_cost(estimate, threshold=20.0)
        assert not should_warn_high


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
