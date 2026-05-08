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

"""Unit tests for HuggingFaceRouterInferenceEngine."""

import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.huggingface_inference_engine import (
    HuggingFaceRouterInferenceEngine,
)


def _make_engine(**kwargs) -> HuggingFaceRouterInferenceEngine:
    model_params = kwargs.pop("model_params", ModelParams(model_name="owner/model"))
    return HuggingFaceRouterInferenceEngine(model_params=model_params, **kwargs)


def _simple_conversation() -> Conversation:
    return Conversation(messages=[Message(role=Role.USER, content="Hello")])


class TestHuggingFaceRouterInferenceEngineInit:
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
            HuggingFaceRouterInferenceEngine()  # type: ignore[call-arg]


class TestProviderSuffixPassThrough:
    """The HF router parses the ``:provider`` suffix server-side; the engine must
    pass ``model_name`` through to the request body unchanged."""

    def _convert(self, model_name: str) -> dict:
        engine = _make_engine(model_params=ModelParams(model_name=model_name))
        return engine._convert_conversation_to_api_input(
            conversation=_simple_conversation(),
            generation_params=GenerationParams(),
            model_params=ModelParams(model_name=model_name),
        )

    def test_plain_model_name(self):
        result = self._convert("owner/plain-model")
        assert result["model"] == "owner/plain-model"
        assert "provider" not in result

    def test_fastest_suffix_preserved(self):
        result = self._convert("owner/model:fastest")
        assert result["model"] == "owner/model:fastest"
        assert "provider" not in result

    def test_cheapest_suffix_preserved(self):
        result = self._convert("owner/model:cheapest")
        assert result["model"] == "owner/model:cheapest"

    def test_named_provider_suffix_preserved(self):
        result = self._convert("Qwen/Qwen2.5-7B-Instruct:together")
        assert result["model"] == "Qwen/Qwen2.5-7B-Instruct:together"

    def test_messages_included(self):
        result = self._convert("owner/model")
        assert "messages" in result
        assert any(m.get("role") == "user" for m in result["messages"])

    def test_max_new_tokens_forwarded(self):
        engine = _make_engine(model_params=ModelParams(model_name="owner/model"))
        result = engine._convert_conversation_to_api_input(
            conversation=_simple_conversation(),
            generation_params=GenerationParams(max_new_tokens=512),
            model_params=ModelParams(model_name="owner/model"),
        )
        # RemoteInferenceEngine maps max_new_tokens -> max_completion_tokens.
        assert result.get("max_completion_tokens") == 512


class TestUnsupportedOperations:
    """HF Inference Providers does not expose OpenAI-style /v1/batches, /v1/files,
    or /v1/models; calls to the inherited methods must surface clearly."""

    def test_infer_batch_raises(self):
        engine = _make_engine()
        with pytest.raises(NotImplementedError, match="Batch inference"):
            engine.infer_batch([_simple_conversation()])

    def test_list_models_raises(self):
        engine = _make_engine()
        with pytest.raises(NotImplementedError, match="Listing models"):
            engine.list_models()
