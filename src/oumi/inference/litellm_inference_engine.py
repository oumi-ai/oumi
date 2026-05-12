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

"""Inference engine for LiteLLM.

LiteLLM provides a unified interface to 100+ LLM providers (Anthropic, Bedrock,
Vertex AI, Cohere, Mistral, etc.) through a single ``completion()`` call.
The provider is specified via the model string in ``ModelParams.model_name``
(e.g. ``anthropic/claude-sonnet-4-5``, ``bedrock/anthropic.claude-v2``).
"""

import aiohttp
from typing_extensions import override

from oumi.core.configs import InferenceConfig, RemoteParams
from oumi.core.types.conversation import Conversation
from oumi.inference.adaptive_semaphore import PoliteAdaptiveSemaphore
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.logging import logger

_LITELLM_PLACEHOLDER_URL = "https://litellm.sdk.local/v1/chat/completions"


class LiteLLMInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference via the LiteLLM SDK.

    LiteLLM routes requests to the correct provider based on the model string.
    For example, ``anthropic/claude-sonnet-4-5`` routes to Anthropic,
    ``bedrock/anthropic.claude-v2`` routes to AWS Bedrock, and
    ``vertex_ai/gemini-pro`` routes to Google Vertex AI.

    Authentication is handled via provider-specific environment variables
    (e.g. ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``AWS_ACCESS_KEY_ID``).
    LiteLLM reads these automatically based on the model prefix.

    For a full list of supported providers, see:
    https://docs.litellm.ai/docs/providers

    Example:
        >>> from oumi.core.configs import ModelParams, GenerationParams
        >>> engine = LiteLLMInferenceEngine(
        ...     model_params=ModelParams(model_name="anthropic/claude-sonnet-4-5"),
        ...     generation_params=GenerationParams(max_new_tokens=500),
        ... )
    """

    @property
    @override
    def base_url(self) -> str | None:
        """Placeholder URL to satisfy parent validation.

        The actual routing is handled by the LiteLLM SDK, not via HTTP.
        """
        return _LITELLM_PLACEHOLDER_URL

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """LiteLLM reads provider-specific env vars automatically."""
        return None

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters for LiteLLM."""
        return RemoteParams(num_workers=50, politeness_policy=60.0)

    @override
    def _set_required_fields_for_inference(self, remote_params: RemoteParams):
        """Override to skip the api_key requirement.

        LiteLLM manages authentication internally via provider-specific
        environment variables, so no single api_key is needed.
        """
        if not remote_params.api_url:
            remote_params.api_url = _LITELLM_PLACEHOLDER_URL

    @override
    async def _query_api(
        self,
        conversation: Conversation,
        semaphore: PoliteAdaptiveSemaphore,
        session: aiohttp.ClientSession,
        inference_config: InferenceConfig | None = None,
    ) -> Conversation:
        """Queries a provider via the LiteLLM SDK instead of raw HTTP.

        Args:
            conversation: The conversation to run inference on.
            semaphore: Semaphore to limit concurrent requests.
            session: The aiohttp session (unused; kept for interface compat).
            inference_config: Parameters for inference.

        Returns:
            Conversation: Inference output with the model's response appended.
        """
        import litellm

        if inference_config is None:
            generation_params = self._generation_params
            model_params = self._model_params
        else:
            generation_params = inference_config.generation or self._generation_params
            model_params = inference_config.model or self._model_params

        api_input = self._convert_conversation_to_api_input(
            conversation, generation_params, model_params
        )

        model = api_input.pop('model', model_params.model_name)
        messages = api_input.pop('messages', [])

        semaphore_or_controller = (
            self._adaptive_concurrency_controller
            if self._remote_params.use_adaptive_concurrency
            else semaphore
        )

        async with semaphore_or_controller:
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    drop_params=True,
                    **api_input,
                )
                response_json = response.model_dump(mode='json')

                result = self._convert_api_output_to_conversation(
                    response_json, conversation
                )
                await self._try_record_success()
                return result
            except Exception as e:
                await self._try_record_error()
                logger.error(f'LiteLLMInferenceEngine - inference error: {e}')
                raise
