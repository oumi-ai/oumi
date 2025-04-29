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

import pytest
from unittest.mock import MagicMock, patch

from oumi.inference.meta_inference_engine import MetaInferenceEngine
from oumi.core.configs import InferenceEngineType
from oumi.core.types.conversation import Conversation


@pytest.fixture
def mock_build_inference_engine():
    with patch("oumi.inference.meta_inference_engine.build_inference_engine") as mock:
        engine_mock = MagicMock()
        engine_mock.infer_online.return_value = [Conversation()]
        mock.return_value = engine_mock
        yield mock


class TestMetaInferenceEngine:
    def test_initialization(self):
        """Test that the MetaInferenceEngine initializes correctly with generation params."""
        engine = MetaInferenceEngine(temperature=0.7, max_tokens=1000)
        assert engine.generation_params.temperature == 0.7
        assert engine.generation_params.max_tokens == 1000
        assert not engine._engines  # Should start with empty engines cache

    def test_select_engine_type(self):
        """Test that the correct engine types are selected based on model names."""
        engine = MetaInferenceEngine()
        
        # Test OpenAI models
        assert engine._select_engine_type("gpt-4o") == InferenceEngineType.OPENAI
        assert engine._select_engine_type("text-davinci-003") == InferenceEngineType.OPENAI
        
        # Test Anthropic models
        assert engine._select_engine_type("claude-3-sonnet") == InferenceEngineType.ANTHROPIC
        
        # Test Google models
        assert engine._select_engine_type("gemini-pro") == InferenceEngineType.GOOGLE_GEMINI
        
        # Test LLaMA models
        assert engine._select_engine_type("meta-llama/Llama-3-70b") in [
            InferenceEngineType.VLLM, 
            InferenceEngineType.NATIVE
        ]

    def test_infer_caching(self, mock_build_inference_engine):
        """Test that engines are cached and reused for the same model."""
        engine = MetaInferenceEngine(temperature=0.7)
        conversations = [Conversation()]
        
        # First call should create a new engine
        engine.infer(conversations, model_name="gpt-4")
        assert mock_build_inference_engine.call_count == 1
        
        # Second call with the same model should reuse the engine
        engine.infer(conversations, model_name="gpt-4")
        assert mock_build_inference_engine.call_count == 1
        
        # Different model should create a new engine
        engine.infer(conversations, model_name="claude-3-sonnet")
        assert mock_build_inference_engine.call_count == 2

    def test_infer_with_config(self, mock_build_inference_engine):
        """Test inference with a custom config."""
        engine = MetaInferenceEngine()
        conversations = [Conversation()]
        
        # Create a mock inference config
        inference_config = MagicMock()
        inference_config.copy.return_value = inference_config
        inference_config.engine = InferenceEngineType.OPENAI
        
        # Run inference with the config
        engine.infer(conversations, model_name="gpt-4", inference_config=inference_config)
        
        # Verify that the config was used correctly
        mock_build_inference_engine.assert_called_once()
        _, kwargs = mock_build_inference_engine.call_args
        assert kwargs["engine_type"] == InferenceEngineType.OPENAI