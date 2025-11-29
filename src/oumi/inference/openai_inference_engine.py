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

import copy
import json
from collections.abc import AsyncGenerator
from typing import Any, Optional

import aiohttp
from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.configs.params.generation_params import ReasoningEffort
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.logging import logger


class OpenAIInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the OpenAI API.

    Supports:
    - Standard text generation
    - Streaming responses with usage tracking
    - Tool/function calling
    - Reasoning tokens (o1, o3, o4 models)
    - Vision/multimodal inputs
    """

    # Reasoning models that require special handling
    REASONING_MODELS = {
        "o1-preview",
        "o1-mini",
        "o3-mini",
        "o4-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-chat-latest",
    }

    @property
    @override
    def base_url(self) -> Optional[str]:
        """Return the default base URL for the OpenAI API."""
        return "https://api.openai.com/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> Optional[str]:
        """Return the default environment variable name for the OpenAI API key."""
        return "OPENAI_API_KEY"

    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if the model is a reasoning model."""
        return any(
            reasoning_model in model_name for reasoning_model in self.REASONING_MODELS
        )

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to an OpenAI input.

        Documentation: https://platform.openai.com/docs/api-reference/chat/create

        Args:
            conversation: The conversation to convert.
            generation_params: Parameters for generation during inference.
            model_params: Model parameters to use during inference.

        Returns:
            Dict[str, Any]: A dictionary representing the OpenAI input.
        """
        # Handle reasoning model constraints
        if self._is_reasoning_model(model_params.model_name):
            generation_params = copy.deepcopy(generation_params)
            # Reasoning models don't support logit_bias
            generation_params.logit_bias = {}
            # Some reasoning models only support temperature = 1
            if model_params.model_name == "o1-preview":
                generation_params.temperature = 1.0

        # Build messages list with tool call support
        from oumi.utils.conversation_utils import convert_message_to_json_content_list

        messages = []
        for message in conversation.messages:
            msg_dict: dict[str, Any] = {
                "role": message.role.value,
            }

            # Handle content (can be None for tool call messages)
            if message.content is not None:
                msg_dict["content"] = convert_message_to_json_content_list(message)

            # Handle tool calls (assistant messages)
            if message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type.value,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]

            # Handle tool responses (tool role messages)
            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id

            messages.append(msg_dict)

        # Build base API input using parent class method
        api_input = super()._convert_conversation_to_api_input(
            conversation=conversation,
            generation_params=generation_params,
            model_params=model_params,
        )

        # Replace messages with our enhanced version
        api_input["messages"] = messages

        # Add tool calling parameters
        if generation_params.tools:
            api_input["tools"] = [
                {
                    "type": tool.type.value,
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters,
                        **(
                            {"strict": tool.function.strict}
                            if tool.function.strict is not None
                            else {}
                        ),
                    },
                }
                for tool in generation_params.tools
            ]

            if generation_params.tool_choice:
                api_input["tool_choice"] = generation_params.tool_choice

            if not generation_params.parallel_tool_calls:
                api_input["parallel_tool_calls"] = False

        # Add reasoning effort for reasoning models
        if generation_params.reasoning_effort and self._is_reasoning_model(
            model_params.model_name
        ):
            effort_value = (
                generation_params.reasoning_effort.value
                if isinstance(generation_params.reasoning_effort, ReasoningEffort)
                else generation_params.reasoning_effort
            )
            api_input["reasoning_effort"] = effort_value

        # Add streaming parameters
        if generation_params.stream:
            api_input["stream"] = True
            if generation_params.stream_options:
                api_input["stream_options"] = generation_params.stream_options
            else:
                # Enable usage tracking for streaming by default
                api_input["stream_options"] = {"include_usage": True}

        return api_input

    def _parse_usage_from_response(self, usage_dict: dict[str, Any]) -> dict[str, Any]:
        """Parse token usage from API response.

        Args:
            usage_dict: The 'usage' dictionary from the API response.

        Returns:
            Dictionary with parsed usage information.
        """
        usage = {
            "prompt_tokens": usage_dict.get("prompt_tokens", 0),
            "completion_tokens": usage_dict.get("completion_tokens", 0),
            "total_tokens": usage_dict.get("total_tokens", 0),
        }

        # Parse detailed completion tokens (reasoning, audio, etc.)
        completion_details = usage_dict.get("completion_tokens_details", {})
        if completion_details:
            if "reasoning_tokens" in completion_details:
                usage["reasoning_tokens"] = completion_details["reasoning_tokens"]
            if "audio_tokens" in completion_details:
                usage["audio_tokens"] = completion_details["audio_tokens"]
            if "cached_tokens" in completion_details:
                usage["cached_tokens"] = completion_details["cached_tokens"]

        return usage

    @override
    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts an OpenAI API response to a conversation.

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
        if "choices" not in response or not response["choices"]:
            raise RuntimeError(f"No choices found in API response: {response}")

        choice = response["choices"][0]
        message_data = choice.get("message")
        if not message_data:
            raise RuntimeError(f"No message found in API response: {response}")

        # Parse message content
        content = message_data.get("content")

        # Parse tool calls if present
        tool_calls = None
        if "tool_calls" in message_data and message_data["tool_calls"]:
            try:
                from oumi.core.types.tool_call import FunctionCall, ToolCall, ToolType

                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        type=ToolType(tc["type"]),
                        function=FunctionCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in message_data["tool_calls"]
                ]
            except Exception as e:
                logger.warning(f"Failed to parse tool calls: {e}")
                tool_calls = None

        # Create new message with tool calls support
        new_message = Message(
            content=content,
            role=Role(message_data["role"]),
            tool_calls=tool_calls,
        )

        # Store usage information in metadata if present
        metadata = dict(original_conversation.metadata)
        if "usage" in response:
            usage_info = self._parse_usage_from_response(response["usage"])
            metadata["usage"] = usage_info
            if "model" in response:
                metadata["model"] = response["model"]

        return Conversation(
            messages=[*original_conversation.messages, new_message],
            metadata=metadata,
            conversation_id=original_conversation.conversation_id,
        )

    async def _stream_api_response(
        self,
        response: aiohttp.ClientResponse,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream SSE chunks from the API response.

        Args:
            response: The aiohttp response object.

        Yields:
            Parsed JSON chunks from the stream.
        """
        async for line in response.content:
            line = line.decode("utf-8").strip()

            if not line or line == "data: [DONE]":
                continue

            if line.startswith("data: "):
                line = line[6:]  # Remove "data: " prefix

                try:
                    chunk = json.loads(line)
                    yield chunk
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse streaming chunk: {line[:100]}")
                    continue

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=50, politeness_policy=60.0)

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "guided_decoding",
            "logit_bias",
            "max_new_tokens",
            "min_p",
            "parallel_tool_calls",
            "presence_penalty",
            "reasoning_effort",
            "seed",
            "stop_strings",
            "stop_token_ids",
            "stream",
            "stream_options",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        }
