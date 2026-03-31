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

"""HuggingFace Inference Providers engine implementation."""

from typing import Any

from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine

# Model name suffix separator used by HuggingFace Inference Providers.
# Models can be specified as "owner/model:provider" or "owner/model:cheapest" etc.
_HF_PROVIDER_SUFFIX_SEP = ":"


def _split_model_and_provider(
    model_name: str,
) -> tuple[str, str | None]:
    """Split a HuggingFace model name into model ID and optional provider suffix.

    HuggingFace Inference Providers support an optional provider/policy suffix
    appended to the model name, e.g.:
      - "meta-llama/Llama-3.1-8B-Instruct"           -> model only
      - "meta-llama/Llama-3.1-8B-Instruct:fastest"   -> prefer fastest provider
      - "meta-llama/Llama-3.1-8B-Instruct:cheapest"  -> prefer cheapest provider
      - "meta-llama/Llama-3.1-8B-Instruct:together"  -> use Together AI

    Args:
        model_name: Full model name, optionally with a provider suffix.

    Returns:
        Tuple of (model_id, provider_or_none).
    """
    # HuggingFace model names always contain at least one "/" (e.g. "owner/model").
    # A trailing ":xxx" after the last segment is the provider suffix.
    last_slash = model_name.rfind("/")
    if last_slash == -1:
        # No slash — treat the whole string as a model ID.
        return model_name, None

    after_slash = model_name[last_slash + 1 :]
    if _HF_PROVIDER_SUFFIX_SEP in after_slash:
        sep_pos = after_slash.index(_HF_PROVIDER_SUFFIX_SEP)
        model_id = model_name[: last_slash + 1 + sep_pos]
        provider = after_slash[sep_pos + 1 :]
        return model_id, provider or None

    return model_name, None


class HuggingFaceInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference via the HuggingFace Inference Providers API.

    HuggingFace Inference Providers offer serverless, OpenAI-compatible access to
    hundreds of models hosted by HuggingFace and partner providers (Together AI,
    Fireworks, SambaNova, Cerebras, etc.).

    Authentication:
        Set the ``HF_TOKEN`` environment variable to a HuggingFace token with the
        ``Make calls to Inference Providers`` permission, or supply the token via
        ``RemoteParams.api_key`` / ``RemoteParams.api_key_env_varname``.

    Provider selection:
        Append a provider suffix to ``model_name`` in ``ModelParams``:

        * ``"meta-llama/Llama-3.1-8B-Instruct"``          — auto-select provider
        * ``"meta-llama/Llama-3.1-8B-Instruct:fastest"``  — prefer fastest
        * ``"meta-llama/Llama-3.1-8B-Instruct:cheapest"`` — prefer cheapest
        * ``"meta-llama/Llama-3.1-8B-Instruct:together"`` — use Together AI

        The suffix is forwarded as the ``provider`` field in the API request and
        stripped from the model name sent to the API.

    References:
        https://huggingface.co/docs/inference-providers/index
        https://huggingface.co/docs/huggingface_hub/guides/inference
    """

    @property
    @override
    def base_url(self) -> str | None:
        """Return the default base URL for the HuggingFace Inference Providers API."""
        return "https://router.huggingface.co/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """Return the default environment variable name for the HuggingFace token."""
        return "HF_TOKEN"

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to a HuggingFace Inference Providers API request.

        This method extends the standard OpenAI-compatible conversion by optionally
        forwarding a ``provider`` field extracted from the model name suffix.

        Args:
            conversation: The conversation to convert.
            generation_params: Parameters for generation during inference.
            model_params: Model parameters.  ``model_name`` may carry an optional
                provider suffix (e.g. ``"owner/model:together"``).

        Returns:
            A dictionary representing the HuggingFace API request body.
        """
        model_id, provider = _split_model_and_provider(model_params.model_name)

        if provider is not None:
            # Build a temporary ModelParams with the clean model ID so that the
            # parent class sends the correct model name to the API.
            import dataclasses

            model_params = dataclasses.replace(model_params, model_name=model_id)

        api_input = super()._convert_conversation_to_api_input(
            conversation=conversation,
            generation_params=generation_params,
            model_params=model_params,
        )

        if provider is not None:
            api_input["provider"] = provider

        return api_input

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters for the HuggingFace API."""
        return RemoteParams(num_workers=20, politeness_policy=0.0)
