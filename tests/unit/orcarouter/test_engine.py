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

"""Tests for the OrcaRouter inference engine and its model selector."""

import pytest
from aioresponses import aioresponses

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.types.conversation import ContentItem, Conversation, Message, Role, Type
from oumi.inference import OrcaRouterInferenceEngine
from oumi.orcarouter.catalog import (
    OrcaRouterCapability,
    resolve_model_selector,
)
from oumi.orcarouter.credentials import CredentialStore, OrcaRouterSession

_API_BASE = "https://api.orcarouter.ai"
_CHAT_URL = f"{_API_BASE}/v1/models?capability=chat"
# A multimodal selector first satisfies chat on the wire, then requires the
# modality to be declared in architecture.input_modalities client-side, so the
# request URL is the same chat catalog.
_MULTIMODAL_CHAT_URL = f"{_API_BASE}/v1/models?capability=chat"

_LIVE_CATALOG = {
    "data": [
        {
            "id": "openai/gpt-5.5",
            "architecture": {"input_modalities": ["text", "image"]},
            "supported_endpoint_types": ["openai", "openai-response"],
        },
        {
            "id": "deepseek/deepseek-v4-pro",
            "architecture": {"input_modalities": ["text"]},
            "supported_endpoint_types": ["openai"],
            "context_length": 1048576,
        },
        {
            "id": "vendor/flux-image",
            "supported_endpoint_types": ["image-generation"],
        },
    ]
}


def test_engine_defaults_to_the_api_origin_and_the_documented_env_var():
    engine = OrcaRouterInferenceEngine(ModelParams(model_name="openai/gpt-5.5"))
    assert engine._remote_params.api_url == (
        "https://api.orcarouter.ai/v1/chat/completions"
    )
    assert engine._remote_params.api_key_env_varname == "ORCAROUTER_API_KEY"
    assert engine.get_models_api_url() == "https://api.orcarouter.ai/v1/models"
    assert engine.base_url == "https://api.orcarouter.ai/v1/chat/completions"


def test_engine_honors_explicit_overrides():
    engine = OrcaRouterInferenceEngine(
        ModelParams(model_name="orcarouter/auto"),
        generation_params=GenerationParams(max_new_tokens=32),
        remote_params=RemoteParams(
            api_url="https://gateway.internal.example/v1/chat/completions",
            api_key="explicit-key",
        ),
    )
    assert engine._model_params.model_name == "orcarouter/auto"
    assert engine._remote_params.api_url == (
        "https://gateway.internal.example/v1/chat/completions"
    )
    assert engine._get_api_key(engine._remote_params) == "explicit-key"
    # The catalog follows the same origin the inference call uses.
    assert engine.get_models_api_url() == "https://gateway.internal.example/v1/models"


def test_engine_uses_a_custom_inference_origin(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORCA_API_BASE_URL", "https://selfhosted.example")
    engine = OrcaRouterInferenceEngine(ModelParams(model_name="orcarouter/auto"))
    assert engine._remote_params.api_url == (
        "https://selfhosted.example/v1/chat/completions"
    )


def test_engine_supported_params_are_inherited():
    engine = OrcaRouterInferenceEngine(ModelParams(model_name="openai/gpt-5.5"))
    params = engine.get_supported_params()
    assert {"max_new_tokens", "temperature", "top_p", "stop_strings"} <= params


def test_engine_reuses_the_stored_pkce_credential(
    store: CredentialStore, monkeypatch: pytest.MonkeyPatch
):
    """A key minted by the browser login drives the same engine."""
    OrcaRouterSession(store=store).connect_with_api_key("sk-orca-44444444444444444444")
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "oumi.inference.orcarouter_inference_engine.OrcaRouterSession",
        lambda: OrcaRouterSession(store=store),
    )

    engine = OrcaRouterInferenceEngine(ModelParams(model_name="openai/gpt-5.5"))
    assert engine._get_api_key(engine._remote_params) == "sk-orca-44444444444444444444"


def test_selector_options_come_from_the_api_and_exclude_non_chat_models(
    monkeypatch: pytest.MonkeyPatch,
):
    """The option list is fetched, filtered, and ordered by the API catalog."""
    seen = {}

    def callback(url, **kwargs):
        from aioresponses import CallbackResult

        seen["url"] = str(url)
        return CallbackResult(status=200, payload=_LIVE_CATALOG)

    with aioresponses() as m:
        m.get(_CHAT_URL, callback=callback, repeat=True)
        selector = resolve_model_selector(_API_BASE, "sk-orca-test-key")

    assert seen["url"] == _CHAT_URL
    assert selector.source == "live"
    assert selector.degraded is False
    assert [model.id for model in selector.options] == [
        "deepseek/deepseek-v4-pro",
        "openai/gpt-5.5",
    ]
    # An image-generation-only model never reaches a chat selector.
    assert "vendor/flux-image" not in [m.id for m in selector.options]


def test_adding_an_image_attachment_removes_the_text_only_model(
    monkeypatch: pytest.MonkeyPatch,
):
    """The selector options are recomputed when the attachment type changes."""
    with aioresponses() as m:
        m.get(_CHAT_URL, payload=_LIVE_CATALOG, repeat=True)
        text_only = resolve_model_selector(_API_BASE, "sk-orca-test-key")

    with aioresponses() as m:
        m.get(_CHAT_URL, payload=_LIVE_CATALOG, repeat=True)
        with_image = resolve_model_selector(
            _API_BASE,
            "sk-orca-test-key",
            capability="multimodal:image",
            current_model="deepseek/deepseek-v4-pro",
        )

    assert [m.id for m in text_only.options] == [
        "deepseek/deepseek-v4-pro",
        "openai/gpt-5.5",
    ]
    # After an image is attached only the image-capable chat model remains, and
    # the previously selected text-only model is cleared.
    assert [m.id for m in with_image.options] == ["openai/gpt-5.5"]
    assert with_image.selected is None


def test_selector_clears_a_model_that_vanished_from_the_catalog():
    with aioresponses() as m:
        m.get(_CHAT_URL, payload=_LIVE_CATALOG, repeat=True)
        selector = resolve_model_selector(
            _API_BASE,
            "sk-orca-test-key",
            current_model="vendor/model-removed-upstream",
        )

    assert selector.selected is None
    assert "openai/gpt-5.5" in [m.id for m in selector.options]


def test_selector_keeps_a_still_compatible_selection():
    with aioresponses() as m:
        m.get(_CHAT_URL, payload=_LIVE_CATALOG, repeat=True)
        selector = resolve_model_selector(
            _API_BASE, "sk-orca-test-key", current_model="openai/gpt-5.5"
        )
    assert selector.selected == "openai/gpt-5.5"


def test_selector_falls_back_only_to_the_labelled_verified_catalog():
    """An outage yields verified seed options, never free text or a raw example."""
    with aioresponses() as m:
        m.get(_CHAT_URL, status=503, body="down", repeat=True)
        selector = resolve_model_selector(_API_BASE, "sk-orca-test-key")

    assert selector.degraded is True
    assert selector.source == "fallback"
    ids = [model.id for model in selector.options]
    assert ids == sorted(ids)
    assert "openai/gpt-5.5" in ids and "orcarouter/auto" in ids


def test_selector_without_a_credential_is_degraded_rather_than_unusable():
    selector = resolve_model_selector(_API_BASE, None)
    assert selector.degraded is True
    assert selector.options


def test_mark_needs_reauth_after_a_401_from_discovery(
    store: CredentialStore, monkeypatch: pytest.MonkeyPatch
):
    """A rejected key marks exactly that generation; no refresh is attempted."""
    session = OrcaRouterSession(store=store)
    credential = session.connect_with_api_key("sk-orca-55555555555555555555")

    with aioresponses() as m:
        m.get(
            _CHAT_URL,
            status=401,
            body="unauthorized",
            repeat=True,
        )
        engine = OrcaRouterInferenceEngine(
            ModelParams(model_name="openai/gpt-5.5"),
            remote_params=RemoteParams(
                api_key=credential.key, api_url=f"{_API_BASE}/v1/chat/completions"
            ),
        )
        from oumi.orcarouter.catalog import OrcaRouterAuthError

        with pytest.raises(OrcaRouterAuthError):
            engine.list_models()

    assert session.mark_needs_reauth(credential) is True
    assert session.active_credential() is None
    # The secret is still on disk: a failed credential is never silently deleted.
    assert session.current_credential() is not None


def _image_conversation() -> Conversation:
    return Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.TEXT, content="what is this?"),
                    ContentItem(type=Type.IMAGE_PATH, content="fixture.png"),
                ],
            )
        ]
    )


def test_infer_refuses_an_image_for_a_text_only_model(monkeypatch: pytest.MonkeyPatch):
    """Second-layer guard: a stale text-only selection cannot receive an image."""

    def handler(url, **kwargs):
        from aioresponses import CallbackResult

        payload = {
            "data": [
                {
                    "id": "openai/gpt-5.5",
                    "architecture": {"input_modalities": ["text", "image"]},
                    "supported_endpoint_types": ["openai"],
                },
                {
                    "id": "deepseek/deepseek-v4-pro",
                    "architecture": {"input_modalities": ["text"]},
                    "supported_endpoint_types": ["openai"],
                },
            ]
        }
        return CallbackResult(status=200, payload=payload)

    with aioresponses() as m:
        m.get(
            f"{_API_BASE}/v1/models",
            callback=handler,
            repeat=True,
        )
        engine = OrcaRouterInferenceEngine(
            ModelParams(model_name="deepseek/deepseek-v4-pro"),
            remote_params=RemoteParams(
                api_key="sk-orca-test-key", api_url=f"{_API_BASE}/v1/chat/completions"
            ),
        )
        with pytest.raises(ValueError, match="does not declare image input"):
            engine.infer([_image_conversation()])


def test_infer_allows_an_image_for_a_declared_image_model(
    monkeypatch: pytest.MonkeyPatch,
):
    """The guard passes when the catalog declares image input."""

    def handler(url, **kwargs):
        from aioresponses import CallbackResult

        payload = {
            "data": [
                {
                    "id": "openai/gpt-5.5",
                    "architecture": {"input_modalities": ["text", "image"]},
                    "supported_endpoint_types": ["openai"],
                }
            ]
        }
        return CallbackResult(status=200, payload=payload)

    with aioresponses() as m:
        m.get(f"{_API_BASE}/v1/models", callback=handler, repeat=True)
        # Only the guard runs here; no chat request is issued.
        from oumi.inference.orcarouter_inference_engine import (
            ensure_model_accepts_input,
        )

        ensure_model_accepts_input(
            _API_BASE, "sk-orca-test-key", "openai/gpt-5.5", [_image_conversation()]
        )


def test_infer_does_not_probe_the_catalog_for_text_only_input():
    """A text-only request must not pay for a catalog round trip."""
    conversation = Conversation(messages=[Message(role=Role.USER, content="hello")])
    from oumi.inference.orcarouter_inference_engine import _required_input_modalities

    assert _required_input_modalities([conversation]) == ()
    assert _required_input_modalities([_image_conversation()]) == ("image",)


def test_engine_constructs_from_an_inference_config():
    config = InferenceConfig(
        model=ModelParams(model_name="orcarouter/auto"),
        engine=None,
    )
    engine = OrcaRouterInferenceEngine(config.model)
    assert engine._model_params.model_name == "orcarouter/auto"
    assert OrcaRouterCapability.parse("chat") is OrcaRouterCapability.CHAT
