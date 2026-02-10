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

"""Tests for the unified InferenceEngine class."""

from unittest import mock
from unittest.mock import patch

import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.inference import BaseInferenceEngine
from oumi.inference import InferenceEngine
from oumi.inference.anthropic_inference_engine import AnthropicInferenceEngine
from oumi.inference.fireworks_inference_engine import FireworksInferenceEngine
from oumi.inference.openai_inference_engine import OpenAIInferenceEngine
from oumi.inference.openrouter_inference_engine import OpenRouterInferenceEngine
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


@pytest.fixture
def mock_aiohttp():
    """Mock aiohttp to avoid actual HTTP calls."""
    with patch("aiohttp.ClientSession"):
        yield


class TestInferenceEngineInit:
    """Tests for InferenceEngine initialization."""

    def test_init_with_provider_and_model(self, mock_aiohttp):
        """Test basic initialization with provider and model."""
        engine = InferenceEngine(
            provider="openrouter",
            model="meta-llama/llama-3-70b",
        )

        assert engine.provider == "openrouter"
        assert engine._model_params.model_name == "meta-llama/llama-3-70b"
        assert isinstance(engine.engine, OpenRouterInferenceEngine)
        assert isinstance(engine, BaseInferenceEngine)

    def test_init_with_fireworks(self, mock_aiohttp):
        """Test initialization with Fireworks provider."""
        engine = InferenceEngine(
            provider="fireworks",
            model="accounts/fireworks/models/llama-v3-70b",
            temperature=0.7,
        )

        assert engine.provider == "fireworks"
        assert engine._model_params.model_name == "accounts/fireworks/models/llama-v3-70b"
        assert engine._generation_params.temperature == 0.7
        assert isinstance(engine.engine, FireworksInferenceEngine)

    def test_init_with_openai(self, mock_aiohttp):
        """Test initialization with OpenAI provider."""
        engine = InferenceEngine(
            provider="openai",
            model="gpt-4",
            api_key="sk-test-key",
        )

        assert engine.provider == "openai"
        assert engine._model_params.model_name == "gpt-4"
        assert isinstance(engine.engine, OpenAIInferenceEngine)

    def test_init_with_anthropic(self, mock_aiohttp):
        """Test initialization with Anthropic provider."""
        engine = InferenceEngine(
            provider="anthropic",
            model="claude-3-opus-20240229",
        )

        assert engine.provider == "anthropic"
        assert isinstance(engine.engine, AnthropicInferenceEngine)

    def test_init_provider_case_insensitive(self, mock_aiohttp):
        """Test that provider name is case insensitive."""
        engine1 = InferenceEngine(provider="OpenRouter", model="test-model")
        engine2 = InferenceEngine(provider="OPENROUTER", model="test-model")
        engine3 = InferenceEngine(provider="openrouter", model="test-model")

        assert engine1.provider == "openrouter"
        assert engine2.provider == "openrouter"
        assert engine3.provider == "openrouter"

    def test_init_unknown_provider_raises(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            InferenceEngine(provider="unknown_provider", model="test-model")

    def test_init_missing_model_raises(self, mock_aiohttp):
        """Test that missing model raises ValueError."""
        with pytest.raises(ValueError, match="Either 'model' or 'model_params'"):
            InferenceEngine(provider="openrouter")

    def test_supported_providers_method(self):
        """Test that supported_providers() returns available providers."""
        providers = InferenceEngine.supported_providers()
        assert "openrouter" in providers
        assert "fireworks" in providers
        assert "anthropic" in providers
        assert "openai" in providers


class TestInferenceEngineFlatParams:
    """Tests for flat parameter handling."""

    def test_generation_params_from_flat(self, mock_aiohttp):
        """Test that flat generation params are properly set."""
        engine = InferenceEngine(
            provider="openrouter",
            model="test-model",
            temperature=0.8,
            max_new_tokens=500,
            top_p=0.95,
            seed=42,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            stop_strings=["STOP", "END"],
        )

        assert engine._generation_params.temperature == 0.8
        assert engine._generation_params.max_new_tokens == 500
        assert engine._generation_params.top_p == 0.95
        assert engine._generation_params.seed == 42
        assert engine._generation_params.frequency_penalty == 0.5
        assert engine._generation_params.presence_penalty == 0.3
        assert engine._generation_params.stop_strings == ["STOP", "END"]

    def test_remote_params_from_flat(self, mock_aiohttp):
        """Test that flat remote params are properly set."""
        engine = InferenceEngine(
            provider="openrouter",
            model="test-model",
            api_key="test-api-key",
            api_url="https://custom.api.com/v1",
            num_workers=10,
            max_retries=5,
        )

        assert engine.engine._remote_params.api_key == "test-api-key"
        assert engine.engine._remote_params.api_url == "https://custom.api.com/v1"
        assert engine.engine._remote_params.num_workers == 10
        assert engine.engine._remote_params.max_retries == 5

    def test_model_params_from_flat(self, mock_aiohttp):
        """Test that flat model params are properly set."""
        engine = InferenceEngine(
            provider="openrouter",
            model="test-model",
            trust_remote_code=True,
            chat_template="llama3-instruct",
        )

        assert engine._model_params.trust_remote_code is True
        assert engine._model_params.chat_template == "llama3-instruct"


class TestInferenceEngineConfigObjects:
    """Tests for config object handling."""

    def test_init_with_model_params_object(self, mock_aiohttp):
        """Test initialization with ModelParams object."""
        model_params = ModelParams(
            model_name="custom-model",
            trust_remote_code=True,
            model_max_length=4096,
        )
        engine = InferenceEngine(
            provider="openrouter",
            model_params=model_params,
        )

        assert engine._model_params.model_name == "custom-model"
        assert engine._model_params.trust_remote_code is True
        assert engine._model_params.model_max_length == 4096

    def test_init_with_generation_params_object(self, mock_aiohttp):
        """Test initialization with GenerationParams object."""
        generation_params = GenerationParams(
            temperature=0.9,
            max_new_tokens=2000,
            top_p=0.8,
            min_p=0.1,
        )
        engine = InferenceEngine(
            provider="openrouter",
            model="test-model",
            generation_params=generation_params,
        )

        assert engine._generation_params.temperature == 0.9
        assert engine._generation_params.max_new_tokens == 2000
        assert engine._generation_params.top_p == 0.8
        assert engine._generation_params.min_p == 0.1

    def test_init_with_remote_params_object(self, mock_aiohttp):
        """Test initialization with RemoteParams object."""
        remote_params = RemoteParams(
            api_key="object-api-key",
            api_url="https://object.api.com",
            num_workers=20,
            politeness_policy=0.5,
        )
        engine = InferenceEngine(
            provider="openrouter",
            model="test-model",
            remote_params=remote_params,
        )

        assert engine.engine._remote_params.api_key == "object-api-key"
        assert engine.engine._remote_params.api_url == "https://object.api.com"
        assert engine.engine._remote_params.num_workers == 20
        assert engine.engine._remote_params.politeness_policy == 0.5


class TestInferenceEngineFlatParamsOverride:
    """Tests for flat params overriding config objects."""

    def test_flat_model_overrides_model_params(self, mock_aiohttp):
        """Test that flat model param overrides model_params.model_name."""
        model_params = ModelParams(model_name="original-model")
        engine = InferenceEngine(
            provider="openrouter",
            model="override-model",
            model_params=model_params,
        )

        assert engine._model_params.model_name == "override-model"

    def test_flat_temperature_overrides_generation_params(self, mock_aiohttp):
        """Test that flat temperature overrides generation_params.temperature."""
        generation_params = GenerationParams(temperature=0.5, max_new_tokens=100)
        engine = InferenceEngine(
            provider="openrouter",
            model="test-model",
            temperature=0.9,
            generation_params=generation_params,
        )

        assert engine._generation_params.temperature == 0.9
        # Non-overridden params should be preserved
        assert engine._generation_params.max_new_tokens == 100

    def test_flat_api_key_overrides_remote_params(self, mock_aiohttp):
        """Test that flat api_key overrides remote_params.api_key."""
        remote_params = RemoteParams(api_key="original-key", num_workers=5)
        engine = InferenceEngine(
            provider="openrouter",
            model="test-model",
            api_key="override-key",
            remote_params=remote_params,
        )

        assert engine.engine._remote_params.api_key == "override-key"
        # Non-overridden params should be preserved
        assert engine.engine._remote_params.num_workers == 5

    def test_multiple_flat_overrides(self, mock_aiohttp):
        """Test multiple flat params overriding config objects."""
        model_params = ModelParams(
            model_name="original-model",
            trust_remote_code=False,
        )
        generation_params = GenerationParams(
            temperature=0.5,
            max_new_tokens=100,
            top_p=0.8,
        )
        remote_params = RemoteParams(
            api_key="original-key",
            num_workers=5,
        )

        engine = InferenceEngine(
            provider="openrouter",
            model="override-model",
            temperature=0.9,
            max_new_tokens=500,
            api_key="override-key",
            trust_remote_code=True,
            model_params=model_params,
            generation_params=generation_params,
            remote_params=remote_params,
        )

        # Overridden values
        assert engine._model_params.model_name == "override-model"
        assert engine._model_params.trust_remote_code is True
        assert engine._generation_params.temperature == 0.9
        assert engine._generation_params.max_new_tokens == 500
        assert engine.engine._remote_params.api_key == "override-key"

        # Preserved values from config objects
        assert engine._generation_params.top_p == 0.8
        assert engine.engine._remote_params.num_workers == 5

    def test_none_flat_params_do_not_override(self, mock_aiohttp):
        """Test that None flat params don't override config objects."""
        generation_params = GenerationParams(temperature=0.7, top_p=0.9)
        engine = InferenceEngine(
            provider="openrouter",
            model="test-model",
            temperature=None,  # Explicitly None
            generation_params=generation_params,
        )

        # Should use value from generation_params since flat is None
        assert engine._generation_params.temperature == 0.7
        assert engine._generation_params.top_p == 0.9


class TestInferenceEngineDelegation:
    """Tests for method delegation to underlying engine."""

    def test_get_supported_params_delegates(self, mock_aiohttp):
        """Test that get_supported_params delegates to underlying engine."""
        engine = InferenceEngine(provider="openrouter", model="test-model")

        supported = engine.get_supported_params()

        # Should have common params
        assert "temperature" in supported
        assert "max_new_tokens" in supported

    def test_getattr_delegates_to_engine(self, mock_aiohttp):
        """Test that unknown attributes are delegated to underlying engine."""
        engine = InferenceEngine(provider="fireworks", model="test-model")

        # base_url is a property on the underlying engine
        assert engine.base_url == "https://api.fireworks.ai/inference/v1/chat/completions"

    def test_infer_batch_accessible(self, mock_aiohttp):
        """Test that infer_batch is accessible via delegation."""
        engine = InferenceEngine(provider="fireworks", model="test-model")

        # infer_batch should be accessible (even if we don't call it)
        assert hasattr(engine, "infer_batch")
        assert callable(engine.infer_batch)

    def test_repr(self, mock_aiohttp):
        """Test string representation."""
        engine = InferenceEngine(
            provider="openrouter",
            model="meta-llama/llama-3-70b",
        )

        repr_str = repr(engine)
        assert "InferenceEngine" in repr_str
        assert "openrouter" in repr_str
        assert "meta-llama/llama-3-70b" in repr_str


class TestInferenceEngineIsBaseInferenceEngine:
    """Tests to verify InferenceEngine is a proper BaseInferenceEngine subclass."""

    def test_isinstance_base_inference_engine(self, mock_aiohttp):
        """Test that InferenceEngine is instance of BaseInferenceEngine."""
        engine = InferenceEngine(provider="openrouter", model="test-model")
        assert isinstance(engine, BaseInferenceEngine)

    def test_engine_property_returns_underlying(self, mock_aiohttp):
        """Test that engine property returns the underlying engine."""
        engine = InferenceEngine(provider="openrouter", model="test-model")
        assert isinstance(engine.engine, RemoteInferenceEngine)
        assert isinstance(engine.engine, OpenRouterInferenceEngine)


class TestInferenceEngineAllProviders:
    """Tests for all supported providers."""

    @pytest.mark.parametrize(
        "provider,canonical",
        [
            ("anthropic", "anthropic"),
            ("deepseek", "deepseek"),
            ("fireworks", "fireworks"),
            ("gemini", "gemini"),  # gemini is enum value of GOOGLE_GEMINI
            ("google_gemini", "google_gemini"),  # google_gemini is enum name
            ("google_vertex", "google_vertex"),
            ("lambda", "lambda"),
            ("openai", "openai"),
            ("openrouter", "openrouter"),
            ("parasail", "parasail"),
            ("sambanova", "sambanova"),
            ("together", "together"),
            ("remote", "remote"),
            ("remote_vllm", "remote_vllm"),
            # Note: sglang is tested separately as it requires additional mocking
        ],
    )
    def test_remote_providers_init(self, mock_aiohttp, provider, canonical):
        """Test that all remote providers can be initialized."""
        engine = InferenceEngine(
            provider=provider,
            model="test-model",
            api_key="test-key",
            api_url="https://test.api.com",
        )

        assert engine.provider == canonical
        assert engine._model_params.model_name == "test-model"
        assert isinstance(engine, BaseInferenceEngine)

    def test_sglang_provider_init(self, mock_aiohttp):
        """Test SGLang provider initialization with proper mocking."""
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.eos_token = "<eos>"

        with patch.multiple(
            "oumi.inference.sglang_inference_engine",
            build_tokenizer=mock.MagicMock(return_value=mock_tokenizer),
            build_processor=mock.MagicMock(return_value=None),
            is_image_text_llm=mock.MagicMock(return_value=False),
        ):
            engine = InferenceEngine(
                provider="sglang",
                model="test-model",
                api_key="test-key",
                api_url="https://test.api.com",
            )

            assert engine.provider == "sglang"
            assert engine._model_params.model_name == "test-model"
            assert isinstance(engine, BaseInferenceEngine)


class TestInferenceEngineUsageExamples:
    """Tests that match the documented usage examples."""

    def test_example_fireworks(self, mock_aiohttp):
        """Test the Fireworks usage example from docstring."""
        engine = InferenceEngine(
            provider="fireworks",
            model="accounts/fireworks/models/llama-v3-70b",
        )

        assert engine.provider == "fireworks"
        assert (
            engine._model_params.model_name
            == "accounts/fireworks/models/llama-v3-70b"
        )

    def test_example_openrouter(self, mock_aiohttp):
        """Test the OpenRouter usage example from docstring."""
        engine = InferenceEngine(
            provider="openrouter",
            model="meta-llama/llama-3-70b",
        )

        assert engine.provider == "openrouter"
        assert engine._model_params.model_name == "meta-llama/llama-3-70b"

    def test_example_with_temperature(self, mock_aiohttp):
        """Test usage with temperature parameter."""
        engine = InferenceEngine(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_new_tokens=1000,
        )

        assert engine._generation_params.temperature == 0.7
        assert engine._generation_params.max_new_tokens == 1000
