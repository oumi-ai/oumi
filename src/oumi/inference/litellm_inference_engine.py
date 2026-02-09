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

from typing import Any

from tqdm import tqdm
from typing_extensions import override

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role, Type
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

        # Disable LiteLLM's verbose logging by default
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
        """Runs model inference using LiteLLM.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output with assistant responses.
        """
        output_path = inference_config.output_path if inference_config else None
        generation_params = (
            inference_config.generation
            if inference_config and inference_config.generation
            else self._generation_params
        )

        results: list[Conversation] = []

        for conversation in tqdm(input, desc="Running LiteLLM inference"):
            try:
                messages = self._convert_conversation_to_messages(conversation)
                completion_kwargs = self._build_completion_kwargs(
                    messages, generation_params
                )

                response = self._litellm.completion(**completion_kwargs)
                output_conversation = self._convert_response_to_conversation(
                    response, conversation
                )

                results.append(output_conversation)
                self._save_conversation_to_scratch(output_conversation, output_path)

            except Exception as e:
                logger.error(
                    f"Error during LiteLLM inference for conversation "
                    f"{conversation.conversation_id}: {e}"
                )
                raise

        return results

    def _convert_conversation_to_messages(
        self, conversation: Conversation
    ) -> list[dict[str, Any]]:
        """Converts an Oumi Conversation to LiteLLM message format.

        Args:
            conversation: The Oumi Conversation object to convert.

        Returns:
            List of message dictionaries in LiteLLM/OpenAI format.
        """
        messages = []

        for message in conversation.messages:
            role = self._convert_role(message.role)
            content = message.content if isinstance(message.content, str) else ""

            # Handle multimodal content if present
            if not isinstance(message.content, str) and message.content:
                content_parts = []
                for item in message.content:
                    if item.type == Type.TEXT and item.content:
                        content_parts.append({"type": "text", "text": item.content})
                    elif item.type == Type.IMAGE_URL and item.content:
                        content_parts.append(
                            {"type": "image_url", "image_url": {"url": item.content}}
                        )
                    elif item.type == Type.IMAGE_BINARY and item.binary:
                        # Handle base64 encoded images
                        import base64

                        b64_data = base64.b64encode(item.binary).decode("utf-8")
                        data_url = f"data:image/png;base64,{b64_data}"
                        content_parts.append(
                            {"type": "image_url", "image_url": {"url": data_url}}
                        )
                    elif item.type == Type.IMAGE_PATH and item.content:
                        # For image paths, read and encode the file
                        import base64
                        from pathlib import Path

                        image_path = Path(item.content)
                        if image_path.exists():
                            with open(image_path, "rb") as f:
                                b64_data = base64.b64encode(f.read()).decode("utf-8")
                            # Detect image type from extension
                            suffix = image_path.suffix.lower()
                            mime_type = {
                                ".png": "image/png",
                                ".jpg": "image/jpeg",
                                ".jpeg": "image/jpeg",
                                ".gif": "image/gif",
                                ".webp": "image/webp",
                            }.get(suffix, "image/png")
                            content_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{b64_data}"
                                    },
                                }
                            )
                if content_parts:
                    messages.append({"role": role, "content": content_parts})
                    continue

            messages.append({"role": role, "content": content})

        return messages

    def _convert_role(self, role: Role) -> str:
        """Converts an Oumi Role to LiteLLM role string.

        Args:
            role: The Oumi Role enum value.

        Returns:
            The corresponding LiteLLM role string.
        """
        role_mapping = {
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
            Role.SYSTEM: "system",
            Role.TOOL: "tool",
        }
        return role_mapping.get(role, "user")

    def _build_completion_kwargs(
        self, messages: list[dict[str, Any]], generation_params: GenerationParams
    ) -> dict[str, Any]:
        """Builds the keyword arguments for LiteLLM completion call.

        Args:
            messages: The conversation messages in LiteLLM format.
            generation_params: Generation parameters.

        Returns:
            Dictionary of keyword arguments for litellm.completion().
        """
        kwargs: dict[str, Any] = {
            "model": self._model_params.model_name,
            "messages": messages,
            "max_tokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
            "num_retries": self._num_retries,
        }

        # Add optional parameters only if set
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
        """Converts a LiteLLM response to an Oumi Conversation.

        Args:
            response: The LiteLLM ModelResponse object.
            original_conversation: The original conversation that was sent.

        Returns:
            A new Conversation with the assistant's response appended.
        """
        # Extract the assistant's response from the LiteLLM response
        assistant_content = response.choices[0].message.content or ""

        new_message = Message(
            content=assistant_content,
            role=Role.ASSISTANT,
        )

        return Conversation(
            messages=[*original_conversation.messages, new_message],
            metadata=original_conversation.metadata,
            conversation_id=original_conversation.conversation_id,
        )
