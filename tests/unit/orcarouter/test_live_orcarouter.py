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

"""Live OrcaRouter checks.

These run only when ``ORCAROUTER_API_KEY`` is set, so the default unit suite
stays offline and free of credentials. They exercise the real provider code
path: the engine's own ``list_models``, the capability-filtered selector, and a
real ``/v1/chat/completions`` call.
"""

import os

import pytest

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    InferenceEngineType,
    ModelParams,
    RemoteParams,
)
from oumi.core.types.conversation import Role
from oumi.inference import OrcaRouterInferenceEngine
from oumi.orcarouter.catalog import (
    OrcaRouterCapability,
    resolve_model_selector,
    verified_seed_models,
)
from oumi.orcarouter.credentials import resolve_api_base_url, resolve_auth_base_url

pytestmark = pytest.mark.skipif(
    not os.environ.get("ORCAROUTER_API_KEY"),
    reason="live OrcaRouter checks require ORCAROUTER_API_KEY",
)

_CHAT_MODEL = "deepseek/deepseek-v4-flash"


def _engine(model_name: str = _CHAT_MODEL) -> OrcaRouterInferenceEngine:
    return OrcaRouterInferenceEngine(
        ModelParams(model_name=model_name),
        remote_params=RemoteParams(api_key=os.environ["ORCAROUTER_API_KEY"]),
    )


def test_live_origins_are_not_derived_from_one_another():
    assert resolve_auth_base_url() == "https://www.orcarouter.ai"
    assert resolve_api_base_url() == "https://api.orcarouter.ai"
    assert "api.orcarouter.ai" not in resolve_auth_base_url()
    assert resolve_auth_base_url() not in resolve_api_base_url()


def test_live_model_catalog_is_reachable_through_the_engine():
    """The engine's discovery path returns the live catalog for this account."""
    models = _engine().list_models()
    assert models, "the live OrcaRouter catalog returned no chat models"
    assert models == sorted(models)
    # Namespaces are preserved verbatim.
    assert all("/" in model_id for model_id in models)


def test_live_selector_filters_by_capability():
    """A chat selector and a multimodal selector draw from the live catalog."""
    chat = resolve_model_selector(
        resolve_api_base_url(),
        os.environ["ORCAROUTER_API_KEY"],
        capability=OrcaRouterCapability.CHAT,
    )
    assert chat.source == "live"
    assert chat.degraded is False
    assert chat.options

    multimodal = resolve_model_selector(
        resolve_api_base_url(),
        os.environ["ORCAROUTER_API_KEY"],
        capability="multimodal:image",
        current_model=_CHAT_MODEL,
    )
    if multimodal.options:
        # Every option must explicitly declare image input; nothing is guessed.
        assert all(model.supports_input("image") for model in multimodal.options)
    # A text-only selection is cleared as soon as image input is required.
    assert multimodal.selected is None


def test_live_chat_completion_runs_through_the_engine():
    """A real request through the implemented provider path.

    The upstream gateway occasionally drops a routed reply and returns an empty
    completion, so the request is retried a bounded number of times. A non-empty
    assistant reply is still required -- this is not a soft assertion.
    """
    from oumi import infer

    config = InferenceConfig(
        model=ModelParams(model_name=_CHAT_MODEL),
        engine=InferenceEngineType.ORCAROUTER,
        generation=GenerationParams(max_new_tokens=32, temperature=0.0),
    )

    attempts = 3
    for attempt in range(1, attempts + 1):
        conversations = infer(config, ["Reply with exactly: ORCAROUTER OK"])
        replies = [
            message.content
            for conversation in conversations
            for message in conversation.messages
            if message.role == Role.ASSISTANT
        ]
        if any(isinstance(reply, str) and reply.strip() for reply in replies):
            return
    pytest.fail(f"the live gateway returned no assistant text in {attempts} attempts")


def test_live_seed_models_are_a_subset_of_something_real():
    """The verified seed stays small, labelled, and metadata-complete."""
    seed = verified_seed_models()
    assert 3 <= len(seed) <= 8
    gpt = next(model for model in seed if model.id == "openai/gpt-5.5")
    assert set(gpt.reasoning_efforts) == {"low", "medium", "high", "xhigh"}
