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

"""Unit tests for HuggingFaceInferenceEngine."""

from unittest.mock import patch

import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.huggingface_inference_engine import (
    HuggingFaceInferenceEngine,
    _split_model_and_provider,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_engine(**kwargs) -> HuggingFaceInferenceEngine:
    model_params = kwargs.pop("model_params", ModelParams(model_name="owner/model"))
    with patch("aiohttp.ClientSession"):
        return HuggingFaceInferenceEngine(model_params=model_params, **kwargs)


def _simple_conversation() -> Conversation:
    return Conversation(
        messages=[Message(role=Role.USER, content="Hello")]
    )


# ---------------------------------------------------------------------------
# _split_model_and_provider
# ---------------------------------------------------------------------------

class TestSplitModelAndProvider:
    def test_no_provider_suffix(self):
        model_id, provider = _split_model_and_provider("meta-llama/Llama-3.1-8B-Instruct")
        assert model_id == "meta-llama/Llama-3.1-8B-Instruct"
        assert provider is None

    def test_fastest_suffix(self):
        model_id, provider = _split_model_and_provider(
            "meta-llama/Llama-3.1-8B-Instruct:fastest"
        )
        assert model_id == "meta-llama/Llama-3.1-8B-Instruct"
        assert provider == "fastest"

    def test_cheapest_suffix(self):
        model_id, provider = _split_model_and_provider(
            "meta-llama/Llama-3.1-8B-Instruct:cheapest"
        )
        assert model_id == "meta-llama/Llama-3.1-8B-Instruct"
        assert provider == "cheapest"

    def test_named_provider_suffix(self):
        model_id, provider = _split_model_and_provider(
            "Qwen/Qwen2.5-7B-Instruct:together"
        )
        assert model_id == "Qwen/Qwen2.5-7B-Instruct"
        assert provider == "together"

    def test_no_slash_no_suffix(self):
        model_id, provider = _split_model_and_provider("some-model")
        assert model_id == "some-model"
        assert provider is None

    def test_empty_suffix_treated_as_no_provider(self):
        # "owner/model:" has an empty suffix — treated as no provider.
        model_id, provider = _split_model_and_provider("owner/model:")
        assert model_id == "owner/model"
        assert provider is None


# ---------------------------------------------------------------------------
# Engine initialisation
# ---------------------------------------------------------------------------

class TestHuggingFaceInferenceEngineInit:
    def test_default_base_url(self):
        engine = _make_engine()
        assert engine.base_url == "https://router.huggingface.co/v1/chat/completions"

    def test_default_api_key_env_varname(self):
        engine = _make_engine()
        assert engine.api_key_env_varname == "HF_TOKEN"

    def test_model_params_stored(self):
        model_params = ModelParams(model_name="owner/my-model")
        engine = _make_engine(model_params=model_params)
        assert engine._model_params.model_name == "owner/my-model"

    def test_custom_remote_params(self):
        remote_params = RemoteParams(api_key="hf_test", num_workers=5)
        engine = _make_engine(remote_params=remote_params)
        assert engine._remote_params.api_key == "hf_test"
        assert engine._remote_params.num_workers == 5

    def test_default_remote_params_num_workers(self):
        engine = _make_engine()
        assert engine._remote_params.num_workers == 20

    def test_missing_model_params_raises(self):
        with pytest.raises(TypeError):
            HuggingFaceInferenceEngine()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# _convert_conversation_to_api_input
# ---------------------------------------------------------------------------

class TestConvertConversationToApiInput:
    def _convert(
        self,
        model_name: str,
        generation_params: GenerationParams | None = None,
    ) -> dict:
        engine = _make_engine(model_params=ModelParams(model_name=model_name))
        gen_params = generation_params or GenerationParams()
        return engine._convert_conversation_to_api_input(
            conversation=_simple_conversation(),
            generation_params=gen_params,
            model_params=ModelParams(model_name=model_name),
        )

    def test_model_name_without_provider(self):
        result = self._convert("owner/plain-model")
        assert result["model"] == "owner/plain-model"
        assert "provider" not in result

    def test_model_name_with_provider_suffix_stripped(self):
        result = self._convert("owner/model:together")
        assert result["model"] == "owner/model"
        assert result["provider"] == "together"

    def test_fastest_provider(self):
        result = self._convert("owner/model:fastest")
        assert result["model"] == "owner/model"
        assert result["provider"] == "fastest"

    def test_cheapest_provider(self):
        result = self._convert("owner/model:cheapest")
        assert result["model"] == "owner/model"
        assert result["provider"] == "cheapest"

    def test_messages_included(self):
        result = self._convert("owner/model")
        assert "messages" in result
        assert any(m.get("role") == "user" for m in result["messages"])

    def test_max_new_tokens_forwarded(self):
        gen = GenerationParams(max_new_tokens=512)
        result = self._convert("owner/model", generation_params=gen)
        # RemoteInferenceEngine maps max_new_tokens -> max_completion_tokens.
        assert result.get("max_completion_tokens") == 512
