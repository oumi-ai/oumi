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

"""Unit tests for JAX inference engine."""

from unittest.mock import MagicMock

import pytest

from oumi.core.configs import GenerationParams, ModelParams

pytestmark = pytest.mark.jax


@pytest.fixture
def llama3_model_params():
    return ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        load_pretrained_weights=False,
        trust_remote_code=True,
    )


@pytest.fixture
def generation_params():
    return GenerationParams(max_new_tokens=10)


class TestJAXInferenceEngineUnit:
    """Unit tests for JAXInferenceEngine."""

    def test_unsupported_model_raises(self):
        """Unsupported model names should raise ValueError."""
        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        with pytest.raises(ValueError, match="Cannot determine JAX architecture"):
            JAXInferenceEngine._resolve_architecture("totally-unknown-model")

    def test_architecture_resolution(self):
        """Test that model names resolve to correct architectures."""
        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        # Test resolution without instantiation (staticmethod)
        _, arch = JAXInferenceEngine._resolve_architecture("meta-llama/Llama-3.1-8B")
        assert arch == "llama3"

        _, arch = JAXInferenceEngine._resolve_architecture("deepseek-ai/DeepSeek-R1")
        assert arch == "deepseek_r1"

        _, arch = JAXInferenceEngine._resolve_architecture("Qwen/Qwen3-32B")
        assert arch == "qwen3"

        _, arch = JAXInferenceEngine._resolve_architecture("meta-llama/Llama-4-Scout")
        assert arch == "llama4"

        _, arch = JAXInferenceEngine._resolve_architecture(
            "moonshotai/Kimi-K2-Instruct"
        )
        assert arch == "kimi_k2"

        _, arch = JAXInferenceEngine._resolve_architecture("openai/gpt-oss-20b")
        assert arch == "gpt_oss"

        _, arch = JAXInferenceEngine._resolve_architecture("nvidia/Nemotron-3-Nano")
        assert arch == "nemotron3"

    def test_unsupported_architecture_raises(self):
        """Unknown architecture should raise ValueError with helpful message."""
        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        with pytest.raises(ValueError, match="Cannot determine JAX architecture"):
            JAXInferenceEngine._resolve_architecture("some/random-model")

    def test_get_supported_params(self, llama3_model_params):
        """Test supported params returns expected set."""
        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        engine = JAXInferenceEngine.__new__(JAXInferenceEngine)
        params = engine.get_supported_params()
        assert "max_new_tokens" in params
        assert "temperature" in params
        assert "top_p" in params

    def test_cleanup(self, llama3_model_params):
        """Test resource cleanup."""
        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        engine = JAXInferenceEngine.__new__(JAXInferenceEngine)
        engine._model_module = MagicMock()
        engine._weights = MagicMock()
        engine._config = MagicMock()
        engine._tokenizer = MagicMock()

        engine.cleanup()

        assert engine._model_module is None
        assert engine._weights is None
        assert engine._config is None
        assert engine._tokenizer is None


class TestJAXRegistry:
    """Tests for the JAX model registry."""

    def test_registry_has_models(self):
        from oumi.models.experimental.jax_models.registry import SUPPORTED_MODELS

        assert len(SUPPORTED_MODELS) > 0

    def test_all_architectures_have_modules(self):
        from oumi.models.experimental.jax_models.registry import (
            get_implementation_module,
            list_supported_architectures,
        )

        for arch in list_supported_architectures():
            module_path = get_implementation_module(arch)
            assert module_path, f"No module path for architecture: {arch}"

    def test_get_recommended_model(self):
        from oumi.models.experimental.jax_models.registry import get_recommended_model

        # Should return a small model that doesn't require auth
        model = get_recommended_model(max_size_gb=10.0, requires_no_auth=True)
        assert model is not None

    def test_models_by_architecture(self):
        from oumi.models.experimental.jax_models.registry import (
            get_models_by_architecture,
        )

        arch_models = get_models_by_architecture()
        assert "llama3_jax" in arch_models
        assert len(arch_models["llama3_jax"]) > 0


class TestJAXUtils:
    """Tests for JAX utility functions."""

    def test_check_jax_devices(self):
        from oumi.utils.jax_utils import check_jax_devices

        info = check_jax_devices()
        # Should work even if JAX not installed (returns error dict)
        assert isinstance(info, dict)

    def test_setup_jax_for_performance(self):
        from oumi.utils.jax_utils import setup_jax_for_performance

        # Should not raise
        setup_jax_for_performance()

    def test_memory_usage_mb(self):
        from oumi.utils.jax_utils import memory_usage_mb

        result = memory_usage_mb()
        assert isinstance(result, float)
        assert result >= 0.0
