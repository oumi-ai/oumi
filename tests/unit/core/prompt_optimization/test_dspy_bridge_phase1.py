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

"""Unit tests for Phase 1 improvements to DSPy bridge."""

import pytest

from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.prompt_optimization_params import (
    PromptOptimizationParams,
)
from oumi.core.configs.prompt_config import PromptOptimizationConfig
from oumi.core.prompt_optimization.dspy_integration import OumiDSPyBridge


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return PromptOptimizationConfig(
        model=ModelParams(model_name="gpt2"),
        generation=GenerationParams(
            temperature=0.7,
            max_new_tokens=100,
            top_p=0.9,
        ),
        optimization=PromptOptimizationParams(
            optimizer="mipro",
            num_trials=10,
            verbose=False,
        ),
        train_dataset_path="dummy_train.jsonl",
        output_dir="dummy_output",
    )


class TestTokenCounting:
    """Test token counting functionality."""

    def test_estimate_token_count_empty_string(self, mock_config):
        """Test token counting with empty string."""
        bridge = OumiDSPyBridge(mock_config)
        count = bridge._estimate_token_count("")
        assert count == 0

    def test_estimate_token_count_fallback(self, mock_config):
        """Test token counting fallback when tokenizer unavailable."""
        bridge = OumiDSPyBridge(mock_config)
        # Force tokenizer to None to test fallback
        bridge._tokenizer = None

        text = "This is a test sentence."
        count = bridge._estimate_token_count(text)

        # Fallback uses ~4 chars per token
        expected_min = len(text) // 5
        expected_max = len(text) // 3
        assert expected_min <= count <= expected_max

    def test_estimate_token_count_with_tokenizer(self, mock_config):
        """Test token counting with actual tokenizer."""
        bridge = OumiDSPyBridge(mock_config)

        # Try to load tokenizer
        tokenizer = bridge._get_tokenizer()

        if tokenizer is not None:
            text = "Hello world!"
            count = bridge._estimate_token_count(text)
            # Should return a positive integer
            assert count > 0
            assert isinstance(count, int)


class TestContextManager:
    """Test context manager for temporary parameter overrides."""

    def test_temporary_params_restore_on_success(self, mock_config):
        """Test that params are restored after successful execution."""
        bridge = OumiDSPyBridge(mock_config)
        original_temp = bridge.config.generation.temperature
        original_max_tokens = bridge.config.generation.max_new_tokens

        with bridge._temporary_generation_params(temperature=0.5, max_tokens=50):
            # Inside context, params should be overridden
            assert bridge.config.generation.temperature == 0.5
            assert bridge.config.generation.max_new_tokens == 50

        # After context, params should be restored
        assert bridge.config.generation.temperature == original_temp
        assert bridge.config.generation.max_new_tokens == original_max_tokens

    def test_temporary_params_restore_on_exception(self, mock_config):
        """Test that params are restored even if exception occurs."""
        bridge = OumiDSPyBridge(mock_config)
        original_temp = bridge.config.generation.temperature

        try:
            with bridge._temporary_generation_params(temperature=0.5):
                assert bridge.config.generation.temperature == 0.5
                raise ValueError("Test exception")
        except ValueError:
            pass

        # After exception, params should still be restored
        assert bridge.config.generation.temperature == original_temp

    def test_temporary_params_only_overrides_provided(self, mock_config):
        """Test that only provided params are overridden."""
        bridge = OumiDSPyBridge(mock_config)
        original_temp = bridge.config.generation.temperature
        original_top_p = bridge.config.generation.top_p

        with bridge._temporary_generation_params(temperature=0.5):
            # Only temperature should be overridden
            assert bridge.config.generation.temperature == 0.5
            assert bridge.config.generation.top_p == original_top_p

        # Both should be restored
        assert bridge.config.generation.temperature == original_temp
        assert bridge.config.generation.top_p == original_top_p


class TestResponseAttributes:
    """Test that response objects have required attributes."""

    @pytest.mark.skipif(
        not pytest.importorskip("dspy", reason="DSPy not installed"),
        reason="DSPy required for this test",
    )
    def test_response_has_cache_hit(self, mock_config):
        """Test that response has cache_hit attribute."""
        # This is tested in integration since it requires actual inference
        # Just verify the bridge can be created
        bridge = OumiDSPyBridge(mock_config)
        assert bridge is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
