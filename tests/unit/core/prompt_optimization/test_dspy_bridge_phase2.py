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

"""Unit tests for Phase 2 improvements to DSPy bridge."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from oumi.core.configs import InferenceEngineType
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
        engine=InferenceEngineType.NATIVE,
    )


class TestAsyncSupport:
    """Test async support functionality."""

    @pytest.mark.asyncio
    async def test_aforward_method_exists(self, mock_config):
        """Test that aforward method exists and is async."""
        bridge = OumiDSPyBridge(mock_config)
        lm = bridge.create_dspy_lm()

        # Check that aforward exists
        assert hasattr(lm, "aforward")
        assert asyncio.iscoroutinefunction(lm.aforward)

    @pytest.mark.asyncio
    async def test_aforward_calls_forward(self, mock_config):
        """Test that aforward properly delegates to forward."""
        bridge = OumiDSPyBridge(mock_config)
        lm = bridge.create_dspy_lm()

        # Mock the forward method
        mock_response = Mock()
        mock_response.choices = [{"text": "test response"}]
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20}
        mock_response.model = "gpt2"
        mock_response.cache_hit = False
        mock_response._hidden_params = {"response_cost": None}

        with patch.object(lm, "forward", return_value=mock_response) as mock_forward:
            # Call aforward
            result = await lm.aforward(prompt="test prompt")

            # Verify forward was called
            mock_forward.assert_called_once_with(prompt="test prompt", messages=None)
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_aforward_with_kwargs(self, mock_config):
        """Test that aforward passes kwargs to forward."""
        bridge = OumiDSPyBridge(mock_config)
        lm = bridge.create_dspy_lm()

        mock_response = Mock()
        mock_response.choices = [{"text": "test response"}]
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20}
        mock_response.model = "gpt2"
        mock_response.cache_hit = False
        mock_response._hidden_params = {"response_cost": None}

        with patch.object(lm, "forward", return_value=mock_response) as mock_forward:
            # Call aforward with kwargs
            await lm.aforward(prompt="test prompt", temperature=0.5, max_tokens=50)

            # Verify forward was called with kwargs
            mock_forward.assert_called_once_with(
                prompt="test prompt",
                messages=None,
                temperature=0.5,
                max_tokens=50,
            )


class TestStateSerialization:
    """Test state serialization functionality."""

    def test_dump_state_structure(self, mock_config):
        """Test that dump_state returns correct structure."""
        bridge = OumiDSPyBridge(mock_config)
        lm = bridge.create_dspy_lm()

        # Set some state
        lm.num_calls = 10
        lm.failed_calls = 2

        state = lm.dump_state()

        # Verify state structure
        assert isinstance(state, dict)
        assert "model" in state
        assert "generation_config" in state
        assert "num_calls" in state
        assert "failed_calls" in state
        assert "history_length" in state
        assert "stats" in state

    def test_dump_state_values(self, mock_config):
        """Test that dump_state returns correct values."""
        bridge = OumiDSPyBridge(mock_config)
        lm = bridge.create_dspy_lm()

        # Set some state
        lm.num_calls = 10
        lm.failed_calls = 2
        lm.history = [{"test": "entry"}, {"test": "entry2"}]

        state = lm.dump_state()

        # Verify values
        assert state["model"] == "gpt2"
        assert state["num_calls"] == 10
        assert state["failed_calls"] == 2
        assert state["history_length"] == 2
        assert state["generation_config"]["temperature"] == 0.7
        assert state["generation_config"]["max_new_tokens"] == 100

    def test_load_state_restores_counts(self, mock_config):
        """Test that load_state restores call counts."""
        bridge = OumiDSPyBridge(mock_config)
        lm = bridge.create_dspy_lm()

        # Create state to load
        state = {
            "model": "gpt2",
            "num_calls": 15,
            "failed_calls": 3,
            "history_length": 12,
        }

        # Load state
        result = lm.load_state(state)

        # Verify state was restored
        assert lm.num_calls == 15
        assert lm.failed_calls == 3
        assert result is lm  # Should return self

    def test_load_state_warns_on_model_mismatch(self, mock_config):
        """Test that load_state warns when models don't match."""
        bridge = OumiDSPyBridge(mock_config)
        lm = bridge.create_dspy_lm()

        # Create state with different model
        state = {
            "model": "different-model",
            "num_calls": 10,
            "failed_calls": 2,
        }

        # Load state and verify warning is logged
        with patch(
            "oumi.core.prompt_optimization.dspy_integration.logger"
        ) as mock_logger:
            lm.load_state(state)
            mock_logger.warning.assert_called_once()
            assert "different-model" in mock_logger.warning.call_args[0][0]

    def test_dump_and_load_roundtrip(self, mock_config):
        """Test that dump_state and load_state work together."""
        bridge = OumiDSPyBridge(mock_config)
        lm1 = bridge.create_dspy_lm()

        # Set state
        lm1.num_calls = 20
        lm1.failed_calls = 5

        # Dump state
        state = lm1.dump_state()

        # Create new LM and load state
        lm2 = bridge.create_dspy_lm()
        lm2.load_state(state)

        # Verify state was transferred
        assert lm2.num_calls == 20
        assert lm2.failed_calls == 5


class TestCallbackSupport:
    """Test callback support functionality."""

    def test_callbacks_passed_to_bridge(self, mock_config):
        """Test that callbacks are passed to bridge."""
        mock_callback = Mock()
        callbacks = [mock_callback]

        bridge = OumiDSPyBridge(mock_config, callbacks=callbacks)

        assert bridge.callbacks == callbacks
        assert len(bridge.callbacks) == 1

    def test_callbacks_default_to_empty_list(self, mock_config):
        """Test that callbacks default to empty list."""
        bridge = OumiDSPyBridge(mock_config)

        assert bridge.callbacks == []
        assert isinstance(bridge.callbacks, list)

    def test_callbacks_passed_to_lm(self, mock_config):
        """Test that callbacks are passed to LM on creation."""
        mock_callback = Mock()
        callbacks = [mock_callback]

        bridge = OumiDSPyBridge(mock_config, callbacks=callbacks)

        # Mock dspy.LM to verify callbacks are passed
        with patch("dspy.LM.__init__", return_value=None) as mock_init:
            _ = bridge.create_dspy_lm()

            # Verify LM was initialized with callbacks
            # Note: This checks the super().__init__ call
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert "callbacks" in call_kwargs
            assert call_kwargs["callbacks"] == callbacks

    def test_multiple_callbacks(self, mock_config):
        """Test that multiple callbacks can be registered."""
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()
        callbacks = [callback1, callback2, callback3]

        bridge = OumiDSPyBridge(mock_config, callbacks=callbacks)

        assert len(bridge.callbacks) == 3
        assert callback1 in bridge.callbacks
        assert callback2 in bridge.callbacks
        assert callback3 in bridge.callbacks


class TestIntegration:
    """Integration tests for Phase 2 features."""

    @pytest.mark.asyncio
    async def test_async_with_state_serialization(self, mock_config):
        """Test that async calls work with state serialization."""
        bridge = OumiDSPyBridge(mock_config)
        lm = bridge.create_dspy_lm()

        # Mock forward to avoid actual inference
        mock_response = Mock()
        mock_response.choices = [{"text": "async test"}]
        mock_response.usage = {"prompt_tokens": 5, "completion_tokens": 10}
        mock_response.model = "gpt2"
        mock_response.cache_hit = False
        mock_response._hidden_params = {"response_cost": None}

        def mock_forward_with_count(*args, **kwargs):
            """Mock forward that increments call count."""
            lm.num_calls += 1
            return mock_response

        with patch.object(lm, "forward", side_effect=mock_forward_with_count):
            # Make async call
            await lm.aforward(prompt="test")

            # Dump state
            state = lm.dump_state()

            # Verify state includes call count
            assert state["num_calls"] == 1
            assert state["failed_calls"] == 0

            # Load state into new LM
            lm2 = bridge.create_dspy_lm()
            lm2.load_state(state)

            # Verify state was restored
            assert lm2.num_calls == 1
            assert lm2.failed_calls == 0

    def test_callbacks_with_state_serialization(self, mock_config):
        """Test that callbacks and state serialization work together."""
        mock_callback = Mock()
        bridge = OumiDSPyBridge(mock_config, callbacks=[mock_callback])
        lm = bridge.create_dspy_lm()

        # Set some state
        lm.num_calls = 5
        lm.failed_calls = 1

        # Dump and load state
        state = lm.dump_state()
        lm2 = bridge.create_dspy_lm()
        lm2.load_state(state)

        # Verify state was restored
        assert lm2.num_calls == 5
        assert lm2.failed_calls == 1


class TestBackwardCompatibility:
    """Test that Phase 2 changes don't break existing functionality."""

    def test_bridge_creation_without_callbacks(self, mock_config):
        """Test that bridge can be created without callbacks."""
        bridge = OumiDSPyBridge(mock_config)
        lm = bridge.create_dspy_lm()

        assert lm is not None
        assert hasattr(lm, "forward")

    def test_existing_methods_still_work(self, mock_config):
        """Test that existing methods still work."""
        bridge = OumiDSPyBridge(mock_config)
        lm = bridge.create_dspy_lm()

        # Test that old methods still exist
        assert hasattr(lm, "forward")
        assert hasattr(lm, "inspect_history")
        assert hasattr(lm, "get_stats")

        # Test get_stats
        stats = lm.get_stats()
        assert "total_calls" in stats
        assert "failed_calls" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
