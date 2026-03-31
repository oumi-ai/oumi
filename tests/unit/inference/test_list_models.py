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

"""Tests for list_models() across inference engines."""

from unittest.mock import patch

import pytest
from aioresponses import aioresponses

from oumi.core.configs import ModelParams, RemoteParams
from oumi.inference import (
    FireworksInferenceEngine,
    OpenAIInferenceEngine,
    RemoteInferenceEngine,
    TogetherInferenceEngine,
)


@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m


def _make_engine(cls, api_url="http://fake/v1/chat/completions", **kwargs):
    return cls(
        model_params=ModelParams(model_name="test-model"),
        remote_params=RemoteParams(api_key="fake", api_url=api_url),
        **kwargs,
    )


def test_list_models_parses_openai_format(mock_aioresponse):
    """OpenAI-style {"data": [...]} response is parsed correctly."""
    mock_aioresponse.get(
        "http://fake/v1/models",
        payload={"data": [{"id": "m-b"}, {"id": "m-a"}, {"id": "m-c"}]},
    )
    engine = _make_engine(RemoteInferenceEngine)
    assert engine.list_models(chat_only=False) == ["m-a", "m-b", "m-c"]


def test_list_models_http_error_raises(mock_aioresponse):
    """Non-200 responses raise RuntimeError."""
    mock_aioresponse.get("http://fake/v1/models", status=403, body="forbidden")
    engine = _make_engine(RemoteInferenceEngine)
    with pytest.raises(RuntimeError, match="403"):
        engine.list_models()


def test_openai_chat_only_filters_non_chat(mock_aioresponse):
    """OpenAI filter excludes embedding, TTS, and image models."""
    mock_aioresponse.get(
        "http://fake/v1/models",
        payload={
            "data": [
                {"id": "gpt-4o"},
                {"id": "text-embedding-3-large"},
                {"id": "tts-1"},
                {"id": "dall-e-3"},
                {"id": "o3-mini"},
            ]
        },
        repeat=True,
    )
    engine = _make_engine(OpenAIInferenceEngine)
    assert engine.list_models(chat_only=True) == ["gpt-4o", "o3-mini"]
    assert engine.list_models(chat_only=False) == [
        "dall-e-3",
        "gpt-4o",
        "o3-mini",
        "text-embedding-3-large",
        "tts-1",
    ]


def test_together_chat_only_filters_by_type(mock_aioresponse):
    """Together filter uses the 'type' field."""
    mock_aioresponse.get(
        "http://fake/v1/models",
        payload=[
            {"id": "llama-3", "type": "chat"},
            {"id": "flux", "type": "image"},
            {"id": "whisper", "type": "audio"},
        ],
        repeat=True,
    )
    engine = _make_engine(TogetherInferenceEngine)
    assert engine.list_models(chat_only=True) == ["llama-3"]
    assert engine.list_models(chat_only=False) == ["flux", "llama-3", "whisper"]


def test_fireworks_chat_only_filters_by_supports_chat(mock_aioresponse):
    """Fireworks filter uses the 'supports_chat' field."""
    mock_aioresponse.get(
        "https://api.fireworks.ai/inference/v1/models",
        payload={
            "data": [
                {"id": "llama", "supports_chat": True},
                {"id": "embed", "supports_chat": False},
            ]
        },
        repeat=True,
    )
    engine = _make_engine(
        FireworksInferenceEngine,
        api_url="https://api.fireworks.ai/inference/v1/chat/completions",
    )
    assert engine.list_models(chat_only=True) == ["llama"]
    assert engine.list_models(chat_only=False) == ["embed", "llama"]


def test_no_filter_returns_all(mock_aioresponse):
    """Providers without a filter return all models for both chat_only values."""
    mock_aioresponse.get(
        "http://fake/v1/models",
        payload={"data": [{"id": "a"}, {"id": "b"}]},
        repeat=True,
    )
    engine = _make_engine(RemoteInferenceEngine)
    assert engine.list_models(chat_only=True) == ["a", "b"]
    assert engine.list_models(chat_only=False) == ["a", "b"]


def test_models_url_overrides():
    """Engines with non-standard API paths override get_models_api_url."""
    fw = _make_engine(
        FireworksInferenceEngine,
        api_url="https://api.fireworks.ai/inference/v1/chat/completions",
    )
    assert fw.get_models_api_url() == "https://api.fireworks.ai/inference/v1/models"

    base = _make_engine(
        RemoteInferenceEngine, api_url="https://api.example.com/v1/chat/completions"
    )
    assert base.get_models_api_url() == "https://api.example.com/v1/models"


def test_bedrock_list_models_filters_by_modality():
    """Bedrock filters to TEXT-in/TEXT-out models when chat_only=True."""
    mock_response = {
        "modelSummaries": [
            {
                "modelId": "claude-3",
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"],
            },
            {
                "modelId": "titan-embed",
                "inputModalities": ["TEXT"],
                "outputModalities": ["EMBEDDING"],
            },
            {
                "modelId": "nova-canvas",
                "inputModalities": ["TEXT", "IMAGE"],
                "outputModalities": ["IMAGE"],
            },
            {
                "modelId": "claude-vision",
                "inputModalities": ["TEXT", "IMAGE"],
                "outputModalities": ["TEXT"],
            },
        ]
    }

    with (
        patch.dict("os.environ", {"AWS_REGION": "us-east-1"}),
        patch("oumi.inference.bedrock_inference_engine.boto3") as mock_boto3,
    ):
        mock_client = mock_boto3.client.return_value
        mock_client.list_foundation_models.return_value = mock_response

        from oumi.inference import BedrockInferenceEngine

        engine = BedrockInferenceEngine(
            model_params=ModelParams(model_name="test"),
        )
        chat = engine.list_models(chat_only=True)
        all_models = engine.list_models(chat_only=False)

    assert chat == ["claude-3", "claude-vision"]
    assert all_models == ["claude-3", "claude-vision", "nova-canvas", "titan-embed"]
