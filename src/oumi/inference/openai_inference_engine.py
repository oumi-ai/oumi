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
from typing import Any, Optional

from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.logging import logger


class OpenAIInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the OpenAI API."""

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
        if model_params.model_name == "o1-preview":
            generation_params = copy.deepcopy(generation_params)

            # o1-preview does NOT support logit_bias.
            generation_params.logit_bias = {}

            # o1-preview only supports temperature = 1.
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

        return api_input

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

        return Conversation(
            messages=[*original_conversation.messages, new_message],
            metadata=original_conversation.metadata,
            conversation_id=original_conversation.conversation_id,
        )

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=50, politeness_policy=60.0)
