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

"""Tests for DSPy integration enhancements."""

import pytest

from oumi.core.configs.params.prompt_optimization_params import PromptOptimizationParams
from oumi.core.prompt_optimization.base import OptimizationResult


class TestMaxErrorsParameter:
    """Tests for max_errors parameter."""

    def test_max_errors_validation_positive(self):
        """Test that positive max_errors is valid."""
        params = PromptOptimizationParams(max_errors=10)
        params.__finalize_and_validate__()
        assert params.max_errors == 10

    def test_max_errors_validation_none(self):
        """Test that None max_errors is valid."""
        params = PromptOptimizationParams(max_errors=None)
        params.__finalize_and_validate__()
        assert params.max_errors is None

    def test_max_errors_validation_zero(self):
        """Test that zero max_errors is valid."""
        params = PromptOptimizationParams(max_errors=0)
        params.__finalize_and_validate__()
        assert params.max_errors == 0

    def test_max_errors_validation_negative_fails(self):
        """Test that negative max_errors raises ValueError."""
        params = PromptOptimizationParams(max_errors=-1)
        with pytest.raises(ValueError, match="max_errors must be non-negative"):
            params.__finalize_and_validate__()


class TestOptunaOptimizer:
    """Tests for Optuna optimizer configuration."""

    def test_optuna_optimizer_valid(self):
        """Test that 'optuna' is a valid optimizer."""
        params = PromptOptimizationParams(optimizer="optuna")
        params.__finalize_and_validate__()
        assert params.optimizer == "optuna"

    def test_optuna_with_num_trials(self):
        """Test Optuna with custom num_trials."""
        params = PromptOptimizationParams(optimizer="optuna", num_trials=100)
        params.__finalize_and_validate__()
        assert params.optimizer == "optuna"
        assert params.num_trials == 100


class TestOptimizationResultEnhancements:
    """Tests for enhanced OptimizationResult fields."""

    def test_optimization_result_with_candidate_programs(self):
        """Test OptimizationResult with candidate_programs field."""
        candidate_programs = [
            {"index": 0, "program": "instruction 1", "score": 0.85},
            {"index": 1, "program": "instruction 2", "score": 0.90},
        ]

        result = OptimizationResult(
            optimized_prompt="best prompt",
            optimized_demos=[],
            optimized_hyperparameters={},
            final_score=0.90,
            training_history=[],
            num_trials=10,
            metadata={"optimizer": "mipro"},
            candidate_programs=candidate_programs,
        )

        assert result.candidate_programs is not None
        assert len(result.candidate_programs) == 2
        assert result.candidate_programs[0]["score"] == 0.85  # type: ignore[index]
        assert result.candidate_programs[1]["score"] == 0.90  # type: ignore[index]

    def test_optimization_result_with_detailed_results(self):
        """Test OptimizationResult with detailed_results field."""
        detailed_results = {
            "candidates": ["prog1", "prog2"],
            "val_aggregate_scores": [0.8, 0.9],
            "best_idx": 1,
        }

        result = OptimizationResult(
            optimized_prompt="best prompt",
            optimized_demos=[],
            optimized_hyperparameters={},
            final_score=0.90,
            training_history=[],
            num_trials=10,
            metadata={"optimizer": "gepa"},
            detailed_results=detailed_results,
        )

        assert result.detailed_results is not None
        assert result.detailed_results["best_idx"] == 1  # type: ignore[index]
        assert len(result.detailed_results["candidates"]) == 2  # type: ignore[arg-type]

    def test_optimization_result_with_both_enhancements(self):
        """Test OptimizationResult with both candidate_programs and detailed_results."""
        result = OptimizationResult(
            optimized_prompt="best prompt",
            optimized_demos=[],
            optimized_hyperparameters={},
            final_score=0.90,
            training_history=[],
            num_trials=10,
            metadata={"optimizer": "mipro"},
            candidate_programs=[{"index": 0, "program": "p1", "score": 0.9}],
            detailed_results={"extra_info": "data"},
        )

        assert result.candidate_programs is not None
        assert result.detailed_results is not None
        assert len(result.candidate_programs) == 1

    def test_optimization_result_without_enhancements(self):
        """Test OptimizationResult without new fields (backward compatibility)."""
        result = OptimizationResult(
            optimized_prompt="best prompt",
            optimized_demos=[],
            optimized_hyperparameters={},
            final_score=0.90,
            training_history=[],
            num_trials=10,
            metadata={"optimizer": "bootstrap"},
        )

        assert result.candidate_programs is None
        assert result.detailed_results is None


class TestGEPAFeedbackMetricSupport:
    """Tests for GEPA feedback metric support."""

    def test_gepa_metric_signature_detection(self):
        """Test detection of GEPA-compatible metric signatures."""
        from oumi.core.configs.params.generation_params import GenerationParams
        from oumi.core.configs.params.model_params import ModelParams
        from oumi.core.configs.prompt_config import PromptOptimizationConfig
        from oumi.core.prompt_optimization.dspy_integration import OumiDSPyBridge

        # Mock config (simplified for testing)
        config = PromptOptimizationConfig(
            train_dataset_path="dummy.jsonl",
            model=ModelParams(model_name="test-model"),
            generation=GenerationParams(),
            optimization=PromptOptimizationParams(optimizer="gepa"),
        )

        def standard_metric(predictions, references):
            """Standard metric with 2 params."""
            return 1.0

        def gepa_metric(
            predictions, references, trace=None, pred_name=None, pred_trace=None
        ):
            """GEPA-compatible metric with 5 params."""
            if pred_name:
                return {"score": 1.0, "feedback": f"Good job {pred_name}"}
            return 1.0

        bridge = OumiDSPyBridge(config)

        # Test standard metric (should work)
        wrapped_standard = bridge.create_metric(standard_metric)
        assert wrapped_standard is not None

        # Test GEPA metric (should detect extended signature)
        wrapped_gepa = bridge.create_metric(
            gepa_metric,  # type: ignore[arg-type]
            support_gepa_feedback=True,
        )
        assert wrapped_gepa is not None

        # Test with disabled GEPA support
        wrapped_no_gepa = bridge.create_metric(
            gepa_metric,  # type: ignore[arg-type]
            support_gepa_feedback=False,
        )
        assert wrapped_no_gepa is not None


class TestOptimizerFactory:
    """Tests for optimizer factory with new optimizers."""

    def test_create_optuna_optimizer(self):
        """Test that optuna optimizer can be instantiated directly."""
        from oumi.core.configs.params.generation_params import GenerationParams
        from oumi.core.configs.params.model_params import ModelParams
        from oumi.core.configs.prompt_config import PromptOptimizationConfig
        from oumi.core.prompt_optimization import BootstrapFewShotWithOptunaOptimizer

        config = PromptOptimizationConfig(
            train_dataset_path="dummy.jsonl",
            model=ModelParams(model_name="test-model"),
            generation=GenerationParams(),
            optimization=PromptOptimizationParams(optimizer="optuna", num_trials=20),
        )

        optimizer = BootstrapFewShotWithOptunaOptimizer(config, lambda p, r: 1.0)
        assert optimizer is not None
        assert optimizer.get_optimizer_name() == "BootstrapFewShotWithOptuna"
        assert config.optimization.num_trials == 20

    def test_optimizer_factory_supports_all_optimizers(self):
        """Test that all optimizers can be instantiated."""
        from oumi.core.configs.params.generation_params import GenerationParams
        from oumi.core.configs.params.model_params import ModelParams
        from oumi.core.configs.prompt_config import PromptOptimizationConfig
        from oumi.core.prompt_optimization import (
            BootstrapFewShotOptimizer,
            BootstrapFewShotWithOptunaOptimizer,
            GepaOptimizer,
            MiproOptimizer,
        )

        optimizer_map = {
            "mipro": MiproOptimizer,
            "gepa": GepaOptimizer,
            "bootstrap": BootstrapFewShotOptimizer,
            "optuna": BootstrapFewShotWithOptunaOptimizer,
        }

        for optimizer_name, optimizer_class in optimizer_map.items():
            config = PromptOptimizationConfig(
                train_dataset_path="dummy.jsonl",
                model=ModelParams(model_name="test-model"),
                generation=GenerationParams(),
                optimization=PromptOptimizationParams(optimizer=optimizer_name),
            )

            optimizer = optimizer_class(config, lambda p, r: 1.0)
            assert optimizer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
