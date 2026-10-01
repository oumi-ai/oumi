# Copyright 2026 - Oumi
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

"""Tests for the OrcaRouter model catalog and capability filtering."""

import json

import pytest
from aioresponses import aioresponses

from oumi.orcarouter.catalog import (
    CATALOG_MAX_BYTES,
    OrcaRouterCapability,
    OrcaRouterCatalogError,
    discover_models,
    fetch_models,
    filter_models,
    models_endpoint_url,
    parse_model,
    verified_seed_models,
)

_API_BASE = "https://api.orcarouter.ai"
_MODELS_URL = "https://api.orcarouter.ai/v1/models"

# A catalog that exercises every capability and modality the selectors must
# distinguish: text-only chat, image-input chat, embedding, image generation,
# video generation, and rerank.
_CATALOG = {
    "data": [
        {
            "id": "openai/gpt-5.5",
            "object": "model",
            "context_length": 400000,
            "architecture": {"input_modalities": ["text", "image"]},
            "supported_endpoint_types": ["openai", "openai-response", "anthropic"],
        },
        {
            "id": "deepseek/deepseek-v4-pro",
            "object": "model",
            "context_length": 1048576,
            "architecture": {"input_modalities": ["text"]},
            "supported_endpoint_types": ["openai", "openai-response"],
        },
        {
            "id": "google/gemini-3.5-flash",
            "object": "model",
            "architecture": {"input_modalities": ["text", "image", "audio", "video"]},
            "supported_endpoint_types": ["openai", "gemini"],
        },
        # No architecture block: capabilities are undeclared and must fail closed.
        {
            "id": "orcarouter/auto",
            "object": "model",
            "supported_endpoint_types": ["openai", "anthropic", "gemini"],
        },
        {
            "id": "vendor/text-embedding-3",
            "object": "model",
            "supported_endpoint_types": ["embeddings"],
        },
        {
            "id": "vendor/flux-image",
            "object": "model",
            "supported_endpoint_types": ["image-generation"],
        },
        {
            "id": "vendor/veo-video",
            "object": "model",
            "supported_endpoint_types": ["openai-video"],
        },
        {
            "id": "vendor/jina-reranker",
            "object": "model",
            "supported_endpoint_types": ["jina-rerank"],
        },
    ]
}


def _models_for(capability, **kwargs):
    models = [parse_model(raw) for raw in _CATALOG["data"]]
    return filter_models([m for m in models if m], capability, **kwargs)


@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m


def test_catalog_url_is_on_the_api_origin_with_capability():
    """The catalog lives on the inference origin, under /v1, on the API host."""
    assert models_endpoint_url(_API_BASE, OrcaRouterCapability.CHAT) == (
        f"{_MODELS_URL}?capability=chat"
    )
    assert models_endpoint_url(_API_BASE, OrcaRouterCapability.EMBEDDING) == (
        f"{_MODELS_URL}?capability=embedding"
    )
    # No capability means the unfiltered catalog.
    assert models_endpoint_url(_API_BASE, None) == _MODELS_URL


def test_multimodal_capability_keys_parse():
    assert OrcaRouterCapability.parse("multimodal:image") is OrcaRouterCapability.CHAT
    assert OrcaRouterCapability.required_modality("multimodal:video") == "video"
    assert OrcaRouterCapability.required_modality("chat") is None
    with pytest.raises(ValueError):
        OrcaRouterCapability.parse("multimodal:telepathy")
    with pytest.raises(ValueError):
        OrcaRouterCapability.parse("unknown")


def test_parse_model_rejects_records_without_a_usable_id():
    assert parse_model(None) is None
    assert parse_model({"id": ""}) is None
    assert parse_model({"id": 42}) is None
    assert parse_model(["not", "a", "record"]) is None


def test_chat_filter_excludes_non_text_endpoints():
    """Non-text chat-capable records and image/video/rerank-only models are out."""
    chat_ids = [m.id for m in _models_for(OrcaRouterCapability.CHAT)]
    assert chat_ids == [
        "deepseek/deepseek-v4-pro",
        "google/gemini-3.5-flash",
        "openai/gpt-5.5",
        "orcarouter/auto",
    ]
    for excluded in (
        "vendor/text-embedding-3",
        "vendor/flux-image",
        "vendor/veo-video",
        "vendor/jina-reranker",
    ):
        assert excluded not in chat_ids


def test_multimodal_filter_requires_declared_input_modality():
    """Undeclared modalities fail closed; declared ones are admitted."""
    image_ids = [m.id for m in _models_for("multimodal:image")]
    assert image_ids == ["google/gemini-3.5-flash", "openai/gpt-5.5"]
    # `orcarouter/auto` has no architecture block, so it must not appear.
    assert "orcarouter/auto" not in image_ids

    audio_ids = [m.id for m in _models_for("multimodal:audio")]
    assert audio_ids == ["google/gemini-3.5-flash"]

    video_ids = [m.id for m in _models_for("multimodal:video")]
    assert video_ids == ["google/gemini-3.5-flash"]


def test_embedding_image_video_and_rerank_filters_match_endpoints_only():
    assert [m.id for m in _models_for(OrcaRouterCapability.EMBEDDING)] == [
        "vendor/text-embedding-3"
    ]
    assert [m.id for m in _models_for(OrcaRouterCapability.IMAGE)] == [
        "vendor/flux-image"
    ]
    assert [m.id for m in _models_for(OrcaRouterCapability.VIDEO)] == [
        "vendor/veo-video"
    ]
    assert [m.id for m in _models_for(OrcaRouterCapability.RERANK)] == [
        "vendor/jina-reranker"
    ]


def test_verified_seed_keeps_reasoning_and_modality_metadata():
    """The fallback catalog keeps the verified effort ladder and modalities."""
    seed = {model.id: model for model in verified_seed_models()}
    gpt = seed["openai/gpt-5.5"]
    assert gpt.supports_reasoning
    assert set(gpt.reasoning_efforts) == {"low", "medium", "high", "xhigh"}
    assert gpt.supports_input("image")
    assert gpt.supports_input("text")

    gemini = seed["google/gemini-3.5-flash"]
    assert {"image", "audio", "video"} <= set(gemini.input_modalities)

    deepseek = seed["deepseek/deepseek-v4-pro"]
    assert deepseek.supports_input("text")
    assert not deepseek.supports_input("image")
    assert deepseek.context_length == 1048576

    assert set(seed) == {
        "openai/gpt-5.5",
        "anthropic/claude-opus-4.8",
        "google/gemini-3.5-flash",
        "deepseek/deepseek-v4-pro",
        "orcarouter/auto",
    }


@pytest.mark.asyncio
async def test_fetch_models_sends_bearer_and_parses_live_catalog(mock_aioresponse):
    seen = {}

    def callback(url, **kwargs):
        seen["url"] = str(url)
        seen["authorization"] = kwargs["headers"]["Authorization"]
        from aioresponses import CallbackResult

        return CallbackResult(status=200, payload=_CATALOG)

    mock_aioresponse.get(
        f"{_MODELS_URL}?capability=chat", callback=callback, repeat=True
    )

    models = await fetch_models(_API_BASE, "sk-orca-test-key")

    assert seen["url"] == f"{_MODELS_URL}?capability=chat"
    assert seen["authorization"] == "Bearer sk-orca-test-key"
    assert [m.id for m in models] == [
        "deepseek/deepseek-v4-pro",
        "google/gemini-3.5-flash",
        "openai/gpt-5.5",
        "orcarouter/auto",
    ]


@pytest.mark.asyncio
async def test_fetch_models_rejects_oversized_catalog(mock_aioresponse):
    oversized = {"data": [{"id": f"vendor/model-{i}"} for i in range(50)]}
    mock_aioresponse.get(
        f"{_MODELS_URL}?capability=chat",
        body=json.dumps(oversized).encode(),
        status=200,
        repeat=True,
        headers={"Content-Length": str(CATALOG_MAX_BYTES + 1)},
    )
    with pytest.raises(OrcaRouterCatalogError):
        await fetch_models(_API_BASE, "sk-orca-test-key")


@pytest.mark.asyncio
async def test_fetch_models_rejects_non_json_body(mock_aioresponse):
    mock_aioresponse.get(
        f"{_MODELS_URL}?capability=chat", body=b"<html>nope</html>", status=200
    )
    with pytest.raises(OrcaRouterCatalogError):
        await fetch_models(_API_BASE, "sk-orca-test-key")


def test_discover_models_uses_live_result_and_not_the_seed(mock_aioresponse):
    """A successful live catalog is authoritative; seed entries never merge in."""
    mock_aioresponse.get(
        f"{_MODELS_URL}?capability=chat",
        payload={
            "data": [
                {
                    "id": "vendor/live-only-model",
                    "architecture": {"input_modalities": ["text"]},
                    "supported_endpoint_types": ["openai"],
                }
            ]
        },
        repeat=True,
    )

    result = discover_models(_API_BASE, "sk-orca-test-key")

    assert result.degraded is False
    assert result.source == "live"
    assert [m.id for m in result.models] == ["vendor/live-only-model"]
    assert "openai/gpt-5.5" not in [m.id for m in result.models]


def test_discover_models_falls_back_to_verified_seed_on_outage(mock_aioresponse):
    mock_aioresponse.get(
        f"{_MODELS_URL}?capability=chat", status=503, body="unavailable", repeat=True
    )

    result = discover_models(_API_BASE, "sk-orca-test-key")

    assert result.degraded is True
    assert result.source == "fallback"
    assert result.catalog_source == "seed"
    assert result.detail
    assert "openai/gpt-5.5" in [m.id for m in result.models]
    # The seed is filtered by the same capability rules as the live catalog.
    image_only = discover_models(
        _API_BASE, "sk-orca-test-key", capability="multimodal:image"
    )
    assert "deepseek/deepseek-v4-pro" not in [m.id for m in image_only.models]


def test_discover_models_without_a_key_is_degraded_not_free_text():
    result = discover_models(_API_BASE, None)
    assert result.degraded is True
    assert result.source == "fallback"
    assert result.models

    with pytest.raises(OrcaRouterCatalogError):
        discover_models(_API_BASE, None, allow_fallback=False)


def test_discover_models_without_fallback_raises_on_outage(mock_aioresponse):
    mock_aioresponse.get(
        f"{_MODELS_URL}?capability=chat", status=503, body="nope", repeat=True
    )
    with pytest.raises(OrcaRouterCatalogError):
        discover_models(_API_BASE, "sk-orca-test-key", allow_fallback=False)


def test_catalog_result_validates_a_stale_selection():
    result = discover_models(_API_BASE, None)
    assert result.validate_selection("openai/gpt-5.5")
    assert not result.validate_selection("vendor/model-that-vanished")
    assert not result.validate_selection(None)
