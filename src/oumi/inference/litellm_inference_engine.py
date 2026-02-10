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

"""LiteLLM inference engine for unified access to 100+ LLM providers."""

from collections.abc import AsyncIterator
from typing import Any

from tqdm.asyncio import tqdm
from typing_extensions import override

from oumi.core.async_utils import safe_asyncio_run
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.adaptive_semaphore import PoliteAdaptiveSemaphore
from oumi.utils.conversation_utils import create_list_of_message_json_dicts
from oumi.utils.logging import logger


class LiteLLMInferenceEngine(BaseInferenceEngine):
    """Engine for running inference via LiteLLM's unified SDK.

    LiteLLM provides a unified interface to call 100+ LLM providers including
    OpenAI, Anthropic, Google, AWS Bedrock, Azure, and many more through a
    single API.

    Model names use the provider/model format (e.g., "anthropic/claude-4-5-opus",
    "openai/gpt-4o", "bedrock/anthropic.claude-4-5-sonnet").

    For a full list of supported providers and model naming conventions, see:
    https://docs.litellm.ai/docs/providers

    Example:
        ```python
        from oumi.core.configs import ModelParams, GenerationParams
        from oumi.inference import LiteLLMInferenceEngine

        engine = LiteLLMInferenceEngine(
            model_params=ModelParams(model_name="anthropic/claude-3-opus-20240229"),
            generation_params=GenerationParams(max_new_tokens=512, temperature=0.7),
        )

        conversations = [Conversation(messages=[Message(role=Role.USER, content="Hi")])]
        results = engine.infer(conversations)

        # Async streaming inference
        async for chunk in engine.infer_stream_async(conversations[0]):
            print(chunk, end="", flush=True)
        ```

    Note:
        Requires the `litellm` package: `pip install oumi[litellm]`
    """

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: GenerationParams | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        num_retries: int = 3,
        timeout: float | None = None,
        num_workers: int = 10,
        politeness_policy: float = 0.0,
    ):
        """Initializes the LiteLLM inference engine.

        Args:
            model_params: Model parameters including the model name in
                provider/model format (e.g., "anthropic/claude-3-opus-20240229").
            generation_params: Generation parameters for inference.
            api_key: Optional API key. If not provided, LiteLLM will attempt to
                read from the appropriate environment variable for the provider.
            api_base: Optional custom API base URL.
            num_retries: Number of retries for failed requests. Defaults to 3.
            timeout: Request timeout in seconds. Defaults to None (no timeout).
            num_workers: Maximum number of concurrent requests. Defaults to 10.
            politeness_policy: Minimum delay in seconds between consecutive requests
                to avoid rate limiting. Defaults to 0.0 (no delay).
        """
        super().__init__(model_params, generation_params=generation_params)

        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "LiteLLM is required for LiteLLMInferenceEngine. "
                "Install it with: pip install oumi[litellm]"
            ) from e

        self._litellm = litellm
        self._api_key = api_key
        self._api_base = api_base
        self._num_retries = num_retries
        self._timeout = timeout
        self._num_workers = num_workers
        self._politeness_policy = politeness_policy

        litellm.suppress_debug_info = True

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "logit_bias",
            "max_new_tokens",
            "presence_penalty",
            "seed",
            "stop_strings",
            "temperature",
            "top_p",
        }

    @override
    def _infer_online(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        """Runs model inference using LiteLLM with concurrent requests."""
        return safe_asyncio_run(self._infer_async(input, inference_config))

    async def _infer_async(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        """Runs model inference asynchronously with concurrency control."""
        output_path = inference_config.output_path if inference_config else None
        generation_params = (
            inference_config.generation
            if inference_config and inference_config.generation
            else self._generation_params
        )

        semaphore = PoliteAdaptiveSemaphore(
            capacity=self._num_workers,
            politeness_policy=self._politeness_policy,
        )

        async def process_conversation(conversation: Conversation) -> Conversation:
            async with semaphore:
                return await self._query_api_async(
                    conversation, generation_params, output_path
                )

        tasks = [process_conversation(conv) for conv in input]
        disable_tqdm = len(tasks) < 2
        results = await tqdm.gather(
            *tasks, desc="Running LiteLLM inference", disable=disable_tqdm
        )
        return list(results)

    async def _query_api_async(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        output_path: str | None,
    ) -> Conversation:
        """Queries the LiteLLM API asynchronously for a single conversation."""
        try:
            messages = self._convert_conversation_to_messages(conversation)
            completion_kwargs = self._build_completion_kwargs(
                messages, generation_params
            )

            response = await self._litellm.acompletion(**completion_kwargs)
            output_conversation = self._convert_response_to_conversation(
                response, conversation
            )

            self._save_conversation_to_scratch(output_conversation, output_path)
            return output_conversation

        except Exception as e:
            logger.error(
                f"Error during LiteLLM inference for conversation "
                f"{conversation.conversation_id}: {e}"
            )
            raise

    async def infer_stream_async(
        self,
        conversation: Conversation,
        inference_config: InferenceConfig | None = None,
    ) -> AsyncIterator[str]:
        """Runs async streaming inference, yielding tokens as they are generated."""
        generation_params = (
            inference_config.generation
            if inference_config and inference_config.generation
            else self._generation_params
        )

        messages = self._convert_conversation_to_messages(conversation)
        completion_kwargs = self._build_completion_kwargs(messages, generation_params)
        completion_kwargs["stream"] = True

        response = await self._litellm.acompletion(**completion_kwargs)

        async for chunk in response:  # type: ignore[union-attr]
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _convert_conversation_to_messages(
        self, conversation: Conversation
    ) -> list[dict[str, Any]]:
        """Converts an Oumi Conversation to LiteLLM message format."""
        return create_list_of_message_json_dicts(
            conversation.messages,
            group_adjacent_same_role_turns=False,
        )

    def _build_completion_kwargs(
        self, messages: list[dict[str, Any]], generation_params: GenerationParams
    ) -> dict[str, Any]:
        """Builds the keyword arguments for LiteLLM completion call."""
        kwargs: dict[str, Any] = {
            "model": self._model_params.model_name,
            "messages": messages,
            "max_tokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
            "num_retries": self._num_retries,
        }

        if self._api_key is not None:
            kwargs["api_key"] = self._api_key

        if self._api_base is not None:
            kwargs["api_base"] = self._api_base

        if self._timeout is not None:
            kwargs["timeout"] = self._timeout

        if generation_params.top_p is not None:
            kwargs["top_p"] = generation_params.top_p

        if generation_params.frequency_penalty != 0.0:
            kwargs["frequency_penalty"] = generation_params.frequency_penalty

        if generation_params.presence_penalty != 0.0:
            kwargs["presence_penalty"] = generation_params.presence_penalty

        if generation_params.stop_strings is not None:
            kwargs["stop"] = generation_params.stop_strings

        if generation_params.seed is not None:
            kwargs["seed"] = generation_params.seed

        if generation_params.logit_bias:
            kwargs["logit_bias"] = generation_params.logit_bias

        return kwargs

    def _convert_response_to_conversation(
        self, response: Any, original_conversation: Conversation
    ) -> Conversation:
        """Converts a LiteLLM response to an Oumi Conversation."""
        assistant_content = response.choices[0].message.content or ""

        metadata: dict[str, Any] = dict(original_conversation.metadata or {})
        if response.usage:
            metadata["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cached_tokens": response.usage.cached_tokens,
            }

        new_message = Message(
            content=assistant_content,
            role=Role.ASSISTANT,
        )

        return Conversation(
            messages=[*original_conversation.messages, new_message],
            metadata=metadata,
            conversation_id=original_conversation.conversation_id,
        )
