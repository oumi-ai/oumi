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

try:
    import litellm  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    litellm = None  # type: ignore[assignment]

from typing import Any

from typing_extensions import override

from oumi.core.async_utils import safe_asyncio_run
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import (
    Conversation,
    FinishReason,
    Message,
    Role,
)
from oumi.utils.conversation_utils import create_list_of_message_json_dicts
from oumi.utils.logging import logger

_FINISH_REASON_MAP = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class LiteLLMInferenceEngine(BaseInferenceEngine):
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

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: GenerationParams | None = None,
    ):
        """Initializes the LiteLLM inference engine.

        Args:
            model_params: The model parameters. ``model_name`` should use the
                LiteLLM model string format (e.g. ``anthropic/claude-sonnet-4-5``).
            generation_params: The generation parameters.

        Raises:
            RuntimeError: If the ``litellm`` package is not installed.
        """
        if litellm is None:
            raise RuntimeError(
                "litellm is not installed. "
                "Install it with `pip install oumi[litellm]`."
            )
        super().__init__(model_params=model_params, generation_params=generation_params)

    @override
    def get_supported_params(self) -> set[str]:
        """Returns supported generation parameters."""
        return {
            "frequency_penalty",
            "max_new_tokens",
            "presence_penalty",
            "seed",
            "stop_strings",
            "temperature",
            "top_p",
        }

    def _build_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to a litellm.completion() kwargs dict."""
        messages = create_list_of_message_json_dicts(
            conversation.messages,
            group_adjacent_same_role_turns=False,
        )

        api_input: dict[str, Any] = {
            "model": model_params.model_name,
            "messages": messages,
            "temperature": generation_params.temperature,
            "max_completion_tokens": generation_params.max_new_tokens,
        }

        if generation_params.seed is not None:
            api_input["seed"] = generation_params.seed
        if generation_params.top_p is not None:
            api_input["top_p"] = generation_params.top_p
        if generation_params.frequency_penalty:
            api_input["frequency_penalty"] = generation_params.frequency_penalty
        if generation_params.presence_penalty:
            api_input["presence_penalty"] = generation_params.presence_penalty
        if generation_params.stop_strings:
            api_input["stop"] = generation_params.stop_strings

        return api_input

    def _parse_response(
        self, response_json: dict[str, Any], original: Conversation
    ) -> Conversation:
        """Converts a litellm response dict back into a Conversation."""
        if "error" in response_json:
            raise RuntimeError(
                f"API error: "
                f"{response_json['error'].get('message', response_json['error'])}"
            )
        choices = response_json.get("choices")
        if not choices:
            raise RuntimeError(f"No choices in response: {response_json}")

        message = choices[0].get("message", {})
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if content is None and not tool_calls:
            content = ""

        metadata = dict(original.metadata)
        usage = response_json.get("usage")
        if usage:
            metadata["usage"] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        raw_reason = choices[0].get("finish_reason")
        if raw_reason:
            finish = _FINISH_REASON_MAP.get(raw_reason.lower(), FinishReason.UNKNOWN)
            metadata["finish_reason"] = finish.value

        return Conversation(
            messages=[
                *original.messages,
                Message(
                    content=content,
                    role=Role(message.get("role", "assistant")),
                ),
            ],
            metadata=metadata,
            conversation_id=original.conversation_id,
            tools=original.tools,
        )

    @override
    def _infer_online(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        """Runs inference via litellm.acompletion().

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        if inference_config is not None:
            generation_params = inference_config.generation or self._generation_params
            model_params = inference_config.model or self._model_params
        else:
            generation_params = self._generation_params
            model_params = self._model_params

        async def _run_all() -> list[Conversation]:
            results: list[Conversation] = []
            for conversation in input:
                api_input = self._build_api_input(
                    conversation, generation_params, model_params
                )
                try:
                    response = await litellm.acompletion(**api_input)
                    response_json = response.model_dump(mode="json")
                    results.append(
                        self._parse_response(response_json, conversation)
                    )
                except Exception as e:
                    logger.error(
                        f"LiteLLMInferenceEngine - inference error: {e}"
                    )
                    raise
            return results

        return safe_asyncio_run(_run_all())
