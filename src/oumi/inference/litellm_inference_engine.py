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

import asyncio
from typing import Any

from tqdm.asyncio import tqdm
from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import (
    Conversation,
    FinishReason,
    Message,
    Role,
)
from oumi.inference.adaptive_semaphore import PoliteAdaptiveSemaphore
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.conversation_utils import create_list_of_message_json_dicts
from oumi.utils.logging import logger

try:
    import litellm  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    litellm = None  # type: ignore[assignment]

_FINISH_REASON_MAP = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class LiteLLMInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference via the LiteLLM SDK.

    This class extends RemoteInferenceEngine to provide specific functionality
    for interacting with LLM providers through LiteLLM's unified interface.
    It handles the conversion of Oumi's Conversation objects to LiteLLM's
    expected input format, as well as parsing the API responses back into
    Conversation objects.

    LiteLLM routes requests to the correct provider based on the model string.
    For example, ``anthropic/claude-sonnet-4-5`` routes to Anthropic,
    ``bedrock/anthropic.claude-v2`` routes to AWS Bedrock, and
    ``vertex_ai/gemini-pro`` routes to Google Vertex AI.

    Authentication is handled via provider-specific environment variables
    (e.g. ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``AWS_ACCESS_KEY_ID``).
    LiteLLM reads these automatically based on the model prefix.

    For a full list of supported providers, see:
    https://docs.litellm.ai/docs/providers

    Note:
        This engine requires the litellm package to be installed.
        If not installed, it will raise a RuntimeError.

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
        remote_params: RemoteParams | None = None,
    ):
        """Initializes the LiteLLM inference engine.

        Args:
            model_params: The model parameters. ``model_name`` should use the
                LiteLLM model string format (e.g. ``anthropic/claude-sonnet-4-5``).
            generation_params: The generation parameters.
            remote_params: Parameters for remote inference.

        Raises:
            RuntimeError: If the ``litellm`` package is not installed.
        """
        if litellm is None:
            raise RuntimeError(
                "litellm is not installed. Install it with `pip install oumi[litellm]`."
            )

        super().__init__(
            model_params=model_params,
            generation_params=generation_params,
            remote_params=remote_params,
        )

    @property
    @override
    def base_url(self) -> str | None:
        """Return the default base URL for the LiteLLM engine."""
        return None

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """Return the default environment variable name for the API key."""
        return None

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams()

    @override
    def _set_required_fields_for_inference(self, remote_params: RemoteParams):
        """Override to skip API key validation.

        LiteLLM reads provider-specific environment variables automatically
        based on the model prefix (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY).
        """
        pass

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to a litellm.completion() kwargs dict.

        Args:
            conversation: The conversation to convert.
            generation_params: Parameters for text generation.
            model_params: Model parameters to use during inference.

        Returns:
            Dict[str, Any]: A dictionary containing the formatted input for
                the LiteLLM completion call.
        """
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

    def _call_litellm_completion(
        self,
        api_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Synchronously invokes litellm.completion.

        Args:
            api_input: The keyword arguments for litellm.completion().

        Returns:
            Dict[str, Any]: The response as a JSON-serializable dictionary.
        """
        response = litellm.completion(**api_input)
        return response.model_dump(mode="json")

    @override
    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts a LiteLLM response dict back into a Conversation.

        Args:
            response: The API response to convert.
            original_conversation: The original conversation.

        Returns:
            Conversation: The conversation including the generated response.
        """
        if "error" in response:
            raise RuntimeError(
                f"API error: {response['error'].get('message', response['error'])}"
            )
        choices = response.get("choices")
        if not choices:
            raise RuntimeError(f"No choices in response: {response}")

        message = choices[0].get("message", {})
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if content is None and not tool_calls:
            content = ""

        metadata = dict(original_conversation.metadata)
        usage = response.get("usage")
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
                *original_conversation.messages,
                Message(
                    content=content,
                    role=Role(message.get("role", "assistant")),
                    tool_calls=tool_calls,
                ),
            ],
            metadata=metadata,
            conversation_id=original_conversation.conversation_id,
            tools=original_conversation.tools,
        )

    @override
    async def _infer(
        self,
        input: list[Conversation],
        inference_config: Any | None = None,
    ) -> list[Conversation]:
        """Async inference implementation that doesn't use HTTP sessions."""
        semaphore = PoliteAdaptiveSemaphore(
            capacity=self._remote_params.num_workers,
            politeness_policy=self._remote_params.politeness_policy,
        )

        # Create tasks for all conversations
        tasks = [
            self._query_api(
                conversation,
                semaphore,
                None,  # No HTTP session needed for LiteLLM SDK
                inference_config=inference_config,
            )
            for conversation in input
        ]

        disable_tqdm = len(tasks) < 2
        results = await tqdm.gather(*tasks, disable=disable_tqdm)
        return results

    @override
    async def _query_api(
        self,
        conversation: Conversation,
        semaphore: PoliteAdaptiveSemaphore,
        session: Any,
        inference_config: Any | None = None,
    ) -> Conversation:
        """Queries LiteLLM using the SDK instead of HTTP.

        Args:
            conversation: The conversation to run inference on.
            semaphore: Semaphore to limit concurrent requests.
            session: Unused (LiteLLM manages its own connections).
            inference_config: Parameters for inference.

        Returns:
            Conversation: Inference output.
        """
        if inference_config is None:
            remote_params = self._remote_params
            generation_params = self._generation_params
            model_params = self._model_params
            output_path = None
        else:
            remote_params = inference_config.remote_params or self._remote_params
            generation_params = inference_config.generation or self._generation_params
            model_params = inference_config.model or self._model_params
            output_path = inference_config.output_path

        if self._rate_limiter is not None:
            await self._rate_limiter.wait_if_needed()

        semaphore_or_controller = (
            self._adaptive_concurrency_controller
            if self._remote_params.use_adaptive_concurrency
            else semaphore
        )
        async with semaphore_or_controller:
            api_input = self._convert_conversation_to_api_input(
                conversation, generation_params, model_params
            )
            failure_reason: str | None = None
            for attempt in range(remote_params.max_retries + 1):
                try:
                    if attempt > 0:
                        delay = min(
                            remote_params.retry_backoff_base * (2 ** (attempt - 1)),
                            remote_params.retry_backoff_max,
                        )
                        await asyncio.sleep(delay)

                    response = await asyncio.to_thread(
                        self._call_litellm_completion,
                        api_input,
                    )
                    if self._rate_limiter is not None:
                        usage = self._extract_usage_from_response(response)
                        if usage:
                            await self._rate_limiter.record_usage(
                                input_tokens=usage.get("prompt_tokens", 0),
                                output_tokens=usage.get("completion_tokens", 0),
                            )
                    result = self._convert_api_output_to_conversation(
                        response, conversation
                    )
                    if output_path:
                        self._save_conversation_to_scratch(result, output_path)
                    await self._try_record_success()
                    return result
                except RuntimeError:
                    raise
                except Exception as e:
                    failure_reason = f"LiteLLM error: {str(e)}"
                    logger.warning(
                        f"LiteLLMInferenceEngine attempt {attempt + 1}/"
                        f"{remote_params.max_retries + 1} failed: {e}"
                    )
                    await self._try_record_error()
                    if attempt >= remote_params.max_retries:
                        raise RuntimeError(failure_reason) from e
                    continue

            raise RuntimeError(
                f"Failed to query LiteLLM after {remote_params.max_retries} retries. "
                + (f"Reason: {failure_reason}" if failure_reason else "")
            )

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "max_new_tokens",
            "presence_penalty",
            "seed",
            "stop_strings",
            "temperature",
            "top_p",
        }

    @override
    def list_models(self, chat_only: bool = True) -> list[str]:
        """Returns model IDs available through LiteLLM.

        Args:
            chat_only: If True (default), only return chat models.

        Returns:
            list[str]: A sorted list of model ID strings.
        """
        try:
            models = litellm.model_list or []
        except Exception:
            models = []
        return sorted(models)

    # Override batch inference methods to indicate they're not supported
    @override
    def infer_batch(
        self,
        conversations: list[Conversation],
        inference_config: Any | None = None,
    ) -> str:
        """LiteLLM does not support batch inference via OpenAI-style batch API."""
        raise NotImplementedError(
            "Batch inference is not supported for LiteLLM engine."
        )

    @override
    def get_batch_status(self, batch_id: str) -> Any:
        """LiteLLM does not support batch inference via OpenAI-style batch API."""
        raise NotImplementedError(
            "Batch inference is not supported for LiteLLM engine."
        )

    @override
    def list_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
    ) -> Any:
        """LiteLLM does not support batch inference via OpenAI-style batch API."""
        raise NotImplementedError(
            "Batch inference is not supported for LiteLLM engine."
        )

    @override
    def get_batch_results(
        self,
        batch_id: str,
        conversations: list[Conversation],
    ) -> list[Conversation]:
        """LiteLLM does not support batch inference via OpenAI-style batch API."""
        raise NotImplementedError(
            "Batch inference is not supported for LiteLLM engine."
        )
