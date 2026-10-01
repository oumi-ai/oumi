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

"""Inference engine for OrcaRouter.

`OrcaRouter <https://www.orcarouter.ai>`_ is an OpenAI-compatible AI gateway
that routes many providers behind one endpoint. Model names keep their
``vendor/model`` namespace (for example ``openai/gpt-5.5``).

Inference and model discovery use the API origin (``https://api.orcarouter.ai``
by default, overridable through ``ORCA_API_BASE_URL`` or the shared
``ORCA_BASE_URL``). Login uses a different origin, ``ORCA_AUTH_BASE_URL``; the
two are never derived from one another.
"""

from __future__ import annotations

import os
import urllib.parse

import aiohttp
from typing_extensions import override

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.orcarouter.catalog import (
    OrcaRouterCapability,
    discover_models,
    models_endpoint_url,
)
from oumi.orcarouter.credentials import (
    ORCAROUTER_API_KEY_ENV_VAR,
    OrcaRouterSession,
    api_v1_base_url,
    resolve_api_base_url,
)
from oumi.utils.logging import logger


def _required_input_modalities(conversations: list[Conversation]) -> tuple[str, ...]:
    """Returns the non-text modalities these conversations actually send."""
    needs_image = any(
        message.image_content_items
        for conversation in conversations
        for message in conversation.messages
    )
    return ("image",) if needs_image else ()


def ensure_model_accepts_input(
    api_base_url: str,
    api_key: str | None,
    model_name: str,
    conversations: list[Conversation],
) -> None:
    """Fails closed when ``model_name`` cannot accept the content being sent.

    The primary defence is the capability-filtered model selector; this is the
    second layer, so a stale selection can never send an image to a text-only
    model.

    Args:
        api_base_url: The OrcaRouter inference origin.
        api_key: A plain OrcaRouter API key, or None.
        model_name: The selected model ID.
        conversations: The conversations about to be sent.

    Raises:
        ValueError: If the content requires a modality the model does not
            declare in the OrcaRouter catalog.
    """
    required = _required_input_modalities(conversations)
    if not required:
        return

    try:
        result = discover_models(
            api_base_url,
            api_key,
            capability=OrcaRouterCapability.CHAT,
            required_input_modalities=required,
        )
    except Exception as e:
        raise ValueError(
            f"Could not verify that the OrcaRouter model '{model_name}' accepts "
            f"{', '.join(required)} input, so the request was refused "
            f"({type(e).__name__})."
        ) from None

    if not result.validate_selection(model_name):
        compatible = [model.id for model in result.models]
        preview = ", ".join(compatible[:5]) if compatible else "none available"
        raise ValueError(
            f"The OrcaRouter model '{model_name}' does not declare "
            f"{', '.join(required)} input, so it cannot accept this request. "
            f"Pick a model that does, for example: {preview}."
        )


class OrcaRouterInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the OrcaRouter gateway.

    Example:
        >>> from oumi.core.configs import ModelParams
        >>> from oumi.inference import OrcaRouterInferenceEngine
        >>> engine = OrcaRouterInferenceEngine(  # doctest: +SKIP
        ...     model_params=ModelParams(model_name="openai/gpt-5.5")
        ... )

    Documentation: https://www.orcarouter.ai
    """

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: GenerationParams | None = None,
        remote_params: RemoteParams | None = None,
        api_base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Initializes the OrcaRouter engine.

        Args:
            model_params: Model parameters; ``model_name`` keeps the
                ``vendor/model`` namespace.
            generation_params: Generation parameters.
            remote_params: Remote parameters. When ``api_url`` is unset it
                defaults to the OrcaRouter chat-completions endpoint derived from
                the resolved API origin.
            api_base_url: Explicit inference-origin override.
            http_session: A caller-owned aiohttp session to share across
                operations, matching every other remote engine.
        """
        resolved = resolve_api_base_url(api_base_url)
        default_chat_url = f"{api_v1_base_url(resolved)}/chat/completions"
        if remote_params is None:
            remote_params = RemoteParams(api_url=default_chat_url)
        elif not remote_params.api_url:
            remote_params.api_url = default_chat_url
        if not remote_params.api_key_env_varname:
            remote_params.api_key_env_varname = ORCAROUTER_API_KEY_ENV_VAR
        super().__init__(
            model_params,
            generation_params=generation_params,
            remote_params=remote_params,
            http_session=http_session,
        )
        self._api_base_url = resolved

        # A key obtained by `oumi orcarouter login` is stored by the project's
        # own credential store. Use it when neither an explicit key nor the
        # documented environment variable is set, so both sign-in paths feed the
        # same engine. A credential marked as needing reauthentication is never
        # used.
        if not self._remote_params.api_key and not os.environ.get(
            self._remote_params.api_key_env_varname or ""
        ):
            stored = OrcaRouterSession().active_credential()
            if stored is not None:
                self._remote_params.api_key = stored.key

    @property
    @override
    def base_url(self) -> str | None:
        """Returns the default chat-completions URL for the OrcaRouter API."""
        return f"{api_v1_base_url(resolve_api_base_url())}/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """Returns the default environment variable name for the OrcaRouter key."""
        return ORCAROUTER_API_KEY_ENV_VAR

    @override
    def get_models_api_url(self) -> str:
        """Returns the OrcaRouter model-catalog URL on the configured API origin."""
        return models_endpoint_url(self._resolved_api_base_url(), None)

    def _resolved_api_base_url(self) -> str:
        """Returns the API origin backing this engine instance.

        An explicit ``api_url`` on the engine's own remote params wins, so a
        per-config override or a self-hosted base is honored.
        """
        remote_params = getattr(self, "_remote_params", None)
        api_url = getattr(remote_params, "api_url", None)
        if api_url:
            parsed = urllib.parse.urlparse(api_url)
            if parsed.scheme and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}"
        return self._api_base_url

    @override
    def list_models(self, chat_only: bool = True) -> list[str]:
        """Returns the model IDs OrcaRouter advertises for this account.

        Args:
            chat_only: When True (default), only models that can serve a chat
                completion are returned. When False, the unfiltered catalog is
                returned.

        Returns:
            list[str]: Sorted model IDs, namespaces preserved.
        """
        result = discover_models(
            self._resolved_api_base_url(),
            self._get_api_key(self._remote_params),
            capability=OrcaRouterCapability.CHAT if chat_only else None,
        )
        if result.degraded:
            logger.warning(
                "OrcaRouter model discovery fell back to the verified catalog "
                f"({result.detail}). The catalog may be incomplete until the "
                "catalog endpoint is reachable again."
            )
        return [model.id for model in result.models]

    @override
    def infer(
        self,
        input: list[Conversation] | None = None,  # noqa: A002 - matches base signature
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        """Runs inference, refusing image content the selected model cannot accept.

        Args:
            input: Conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            list[Conversation]: Inference output.

        Raises:
            ValueError: If an image is present and the selected model does not
                declare image input in the OrcaRouter catalog.
        """
        conversations = input or []
        model_name = self._model_params.model_name
        if inference_config is not None and inference_config.model:
            model_name = inference_config.model.model_name
        ensure_model_accepts_input(
            self._resolved_api_base_url(),
            self._get_api_key(self._remote_params),
            model_name,
            conversations,
        )
        return super().infer(conversations, inference_config)
