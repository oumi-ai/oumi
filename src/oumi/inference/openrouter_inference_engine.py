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

from typing import Any

from typing_extensions import override

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class OpenRouterInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the OpenRouter API.

    OpenRouter provides a unified API that gives access to hundreds of AI models
    through a single endpoint. It supports models from OpenAI, Anthropic, Google,
    Meta, and many other providers.

    Model names should use the provider/model format (e.g., "openai/gpt-4",
    "anthropic/claude-3-opus", "meta-llama/llama-3-70b-instruct").

    Documentation: https://openrouter.ai/docs
    """

    @property
    @override
    def base_url(self) -> str | None:
        """Return the default base URL for the OpenRouter API."""
        return "https://openrouter.ai/api/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """Return the default environment variable name for the OpenRouter API key."""
        return "OPENROUTER_API_KEY"

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to an OpenRouter input.

        Omits sampling parameters left at their defaults: an unset ``seed`` and
        zero ``frequency_penalty`` / ``presence_penalty``. OpenRouter's
        ``provider.require_parameters`` counts every parameter present in the
        body, whatever its value, so sending them would rule out providers that
        do not support them even though the request does not use them.

        Documentation: https://openrouter.ai/docs/guides/routing/provider-selection
        """
        api_input = super()._convert_conversation_to_api_input(
            conversation, generation_params, model_params
        )
        if api_input.get("seed") is None:
            api_input.pop("seed", None)
        for penalty in ("frequency_penalty", "presence_penalty"):
            if api_input.get(penalty) == 0:
                api_input.pop(penalty, None)
        return api_input

    @override
    def get_models_api_url(self) -> str:
        """Returns the URL for the OpenRouter models API."""
        return "https://openrouter.ai/api/v1/models"

    @override
    def infer_batch(
        self,
        _conversations: list[Conversation],
        _inference_config: InferenceConfig | None = None,
    ) -> str:
        """Batch inference is not implemented for OpenRouter."""
        raise NotImplementedError(
            "Batch inference is not implemented for OpenRouter. "
            "Please open an issue on GitHub if you'd like this feature."
        )
