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

"""Tests for provider detection utilities."""

import pytest

from oumi.core.configs import InferenceEngineType
from oumi.utils.provider_detection import detect_provider, is_yaml_path


class TestDetectProvider:
    """Tests for detect_provider function."""

    def test_explicit_openai_prefix(self):
        """Test explicit openai/ prefix."""
        engine, model = detect_provider("openai/gpt-4o")
        assert engine == InferenceEngineType.OPENAI
        assert model == "gpt-4o"

    def test_explicit_anthropic_prefix(self):
        """Test explicit anthropic/ prefix."""
        engine, model = detect_provider("anthropic/claude-3-opus")
        assert engine == InferenceEngineType.ANTHROPIC
        assert model == "claude-3-opus"

    def test_explicit_google_prefix(self):
        """Test explicit google/ prefix."""
        engine, model = detect_provider("google/gemini-pro")
        assert engine == InferenceEngineType.GOOGLE_GEMINI
        assert model == "gemini-pro"

    def test_explicit_together_prefix(self):
        """Test explicit together/ prefix."""
        engine, model = detect_provider("together/meta-llama/Llama-3.1-8B")
        assert engine == InferenceEngineType.TOGETHER
        assert model == "meta-llama/Llama-3.1-8B"

    def test_explicit_vllm_prefix(self):
        """Test explicit vllm/ prefix."""
        engine, model = detect_provider("vllm/meta-llama/Llama-3.1-8B")
        assert engine == InferenceEngineType.VLLM
        assert model == "meta-llama/Llama-3.1-8B"

    def test_explicit_sglang_prefix(self):
        """Test explicit sglang/ prefix."""
        engine, model = detect_provider("sglang/meta-llama/Llama-3.1-8B")
        assert engine == InferenceEngineType.SGLANG
        assert model == "meta-llama/Llama-3.1-8B"

    def test_auto_detect_openai_gpt(self):
        """Test auto-detection of OpenAI GPT models."""
        engine, model = detect_provider("gpt-4o")
        assert engine == InferenceEngineType.OPENAI
        assert model == "gpt-4o"

        engine, model = detect_provider("gpt-4-turbo")
        assert engine == InferenceEngineType.OPENAI
        assert model == "gpt-4-turbo"

        engine, model = detect_provider("gpt-3.5-turbo")
        assert engine == InferenceEngineType.OPENAI
        assert model == "gpt-3.5-turbo"

    def test_auto_detect_openai_o1(self):
        """Test auto-detection of OpenAI o1 models."""
        engine, model = detect_provider("o1-preview")
        assert engine == InferenceEngineType.OPENAI
        assert model == "o1-preview"

        engine, model = detect_provider("o1-mini")
        assert engine == InferenceEngineType.OPENAI
        assert model == "o1-mini"

    def test_auto_detect_anthropic_claude(self):
        """Test auto-detection of Anthropic Claude models."""
        engine, model = detect_provider("claude-3-opus")
        assert engine == InferenceEngineType.ANTHROPIC
        assert model == "claude-3-opus"

        engine, model = detect_provider("claude-3.5-sonnet")
        assert engine == InferenceEngineType.ANTHROPIC
        assert model == "claude-3.5-sonnet"

        engine, model = detect_provider("claude-2")
        assert engine == InferenceEngineType.ANTHROPIC
        assert model == "claude-2"

    def test_auto_detect_google_gemini(self):
        """Test auto-detection of Google Gemini models."""
        engine, model = detect_provider("gemini-pro")
        assert engine == InferenceEngineType.GOOGLE_GEMINI
        assert model == "gemini-pro"

        engine, model = detect_provider("gemini-1.5-flash")
        assert engine == InferenceEngineType.GOOGLE_GEMINI
        assert model == "gemini-1.5-flash"

    def test_huggingface_model_defaults_to_vllm(self):
        """Test that HuggingFace-style models default to vLLM."""
        engine, model = detect_provider("meta-llama/Llama-3.1-8B")
        assert engine == InferenceEngineType.VLLM
        assert model == "meta-llama/Llama-3.1-8B"

        engine, model = detect_provider("mistralai/Mistral-7B-v0.1")
        assert engine == InferenceEngineType.VLLM
        assert model == "mistralai/Mistral-7B-v0.1"

    def test_unknown_model_defaults_to_vllm(self):
        """Test that unknown models default to vLLM."""
        engine, model = detect_provider("some-unknown-model")
        assert engine == InferenceEngineType.VLLM
        assert model == "some-unknown-model"

    def test_case_insensitive_prefix(self):
        """Test that prefixes are case-insensitive."""
        engine, model = detect_provider("OpenAI/gpt-4o")
        assert engine == InferenceEngineType.OPENAI
        assert model == "gpt-4o"

        engine, model = detect_provider("ANTHROPIC/claude-3")
        assert engine == InferenceEngineType.ANTHROPIC
        assert model == "claude-3"


class TestIsYamlPath:
    """Tests for is_yaml_path function."""

    def test_yaml_extension(self):
        """Test .yaml extension."""
        assert is_yaml_path("config.yaml") is True
        assert is_yaml_path("path/to/config.yaml") is True
        assert is_yaml_path("/absolute/path/config.yaml") is True

    def test_yml_extension(self):
        """Test .yml extension."""
        assert is_yaml_path("config.yml") is True
        assert is_yaml_path("path/to/config.yml") is True
        assert is_yaml_path("/absolute/path/config.yml") is True

    def test_non_yaml_paths(self):
        """Test non-YAML paths."""
        assert is_yaml_path("gpt-4o") is False
        assert is_yaml_path("meta-llama/Llama-3.1-8B") is False
        assert is_yaml_path("config.json") is False
        assert is_yaml_path("config.txt") is False
        assert is_yaml_path("yaml") is False
        assert is_yaml_path("") is False

    def test_yaml_in_model_name(self):
        """Test that 'yaml' in model name doesn't trigger false positive."""
        assert is_yaml_path("my-yaml-model") is False
        assert is_yaml_path("yaml-config-loader") is False
