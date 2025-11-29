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

import json
from typing import Any, Optional

from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.logging import logger

_CONTENT_KEY: str = "content"
_ROLE_KEY: str = "role"


class AnthropicInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Anthropic API.

    This class extends RemoteInferenceEngine to provide specific functionality
    for interacting with Anthropic's language models via their API. It handles
    the conversion of Oumi's Conversation objects to Anthropic's expected input
    format, as well as parsing the API responses back into Conversation objects.
    """

    anthropic_version = "2023-06-01"
    """The version of the Anthropic API to use.

    For more information on Anthropic API versioning, see:
    https://docs.anthropic.com/claude/reference/versioning
    """

    @property
    @override
    def base_url(self) -> Optional[str]:
        """Return the default base URL for the Anthropic API."""
        return "https://api.anthropic.com/v1/messages"

    @property
    @override
    def api_key_env_varname(self) -> Optional[str]:
        """Return the default environment variable name for the Anthropic API key."""
        return "ANTHROPIC_API_KEY"

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to an Anthropic API input.

        This method transforms an Oumi Conversation object into a format
        suitable for the Anthropic API. It handles system messages separately
        and structures the conversation history as required by Anthropic.

        See https://docs.anthropic.com/claude/reference/messages_post for details.

        Args:
            conversation: The Oumi Conversation object to convert.
            generation_params: Parameters for text generation.
            model_params: Model parameters to use during inference.

        Returns:
            Dict[str, Any]: A dictionary containing the formatted input for the
            Anthropic API, including the model, messages, and generation parameters.
        """
        # Anthropic API expects a top level `system` message,
        # Extract and exclude system message from the list of messages
        # in the conversation
        system_messages = [
            message for message in conversation.messages if message.role == Role.SYSTEM
        ]

        if len(system_messages) > 0:
            system_message = system_messages[0].content

            if len(system_messages) > 1:
                logger.warning(
                    "Multiple system messages found in conversation. "
                    "Only using the first one."
                )
        else:
            system_message = None

        # Filter out system messages and build message list
        messages = [
            message for message in conversation.messages if message.role != Role.SYSTEM
        ]

        # Convert messages to Anthropic format
        from oumi.utils.conversation_utils import convert_message_to_json_content_list

        anthropic_messages = []
        for message in messages:
            msg_dict: dict[str, Any] = {
                "role": message.role.value,
            }

            # Handle content
            if message.content is not None:
                content_list = convert_message_to_json_content_list(message)

                # Convert to list format if it's a string
                if isinstance(content_list, str):
                    content_list = [{"type": "text", "text": content_list}]

                msg_dict["content"] = content_list

            # Handle tool calls (assistant calling tools)
            if message.tool_calls:
                if "content" not in msg_dict:
                    msg_dict["content"] = []
                elif isinstance(msg_dict["content"], str):
                    msg_dict["content"] = [
                        {"type": "text", "text": msg_dict["content"]}
                    ]
                elif not isinstance(msg_dict["content"], list):
                    msg_dict["content"] = []

                # Anthropic format for tool calls
                for tc in message.tool_calls:
                    msg_dict["content"].append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": json.loads(tc.function.arguments),
                        }
                    )

            # Handle tool responses (tool role messages)
            if message.tool_call_id:
                msg_dict["content"] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content or "",
                    }
                ]
                msg_dict["role"] = "user"  # Anthropic expects tool results as user role

            anthropic_messages.append(msg_dict)

        # Build request body
        # See https://docs.anthropic.com/claude/reference/messages_post
        body: dict[str, Any] = {
            "model": model_params.model_name,
            "messages": anthropic_messages,
            "max_tokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
            "top_p": generation_params.top_p,
        }

        if system_message:
            body["system"] = system_message

        if generation_params.stop_strings is not None:
            body["stop_sequences"] = generation_params.stop_strings

        # Add tool calling parameters
        if generation_params.tools:
            body["tools"] = [
                {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "input_schema": tool.function.parameters or {"type": "object"},
                }
                for tool in generation_params.tools
            ]

            if generation_params.tool_choice:
                # Convert from OpenAI format to Anthropic format
                if isinstance(generation_params.tool_choice, str):
                    if generation_params.tool_choice == "auto":
                        body["tool_choice"] = {"type": "auto"}
                    elif generation_params.tool_choice == "required":
                        body["tool_choice"] = {"type": "any"}
                    elif generation_params.tool_choice == "none":
                        # Anthropic doesn't have a direct "none" - just don't pass tools
                        body.pop("tools", None)
                elif isinstance(generation_params.tool_choice, dict):
                    # Specific function choice
                    func_name = generation_params.tool_choice.get("function", {}).get(
                        "name"
                    )
                    if func_name:
                        body["tool_choice"] = {"type": "tool", "name": func_name}

        return body

    @override
    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts an Anthropic API response to a conversation."""
        # Handle error responses
        if "error" in response:
            raise RuntimeError(
                f"API error: {response['error'].get('message', response['error'])}"
            )

        # Parse content - Anthropic returns a list of content blocks
        content_blocks = response.get(_CONTENT_KEY, [])

        # Extract text and tool use
        text_content = ""
        tool_calls = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                try:
                    from oumi.core.types.tool_call import (
                        FunctionCall,
                        ToolCall,
                        ToolType,
                    )

                    tool_calls.append(
                        ToolCall(
                            id=block["id"],
                            type=ToolType.FUNCTION,
                            function=FunctionCall(
                                name=block["name"],
                                arguments=json.dumps(block["input"]),
                            ),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse tool call: {e}")

        # Create message
        new_message = Message(
            content=text_content or "",
            role=Role.ASSISTANT,
            tool_calls=tool_calls if tool_calls else None,
        )

        return Conversation(
            messages=[*original_conversation.messages, new_message],
            metadata=original_conversation.metadata,
            conversation_id=original_conversation.conversation_id,
        )

    @override
    def _get_request_headers(self, remote_params: RemoteParams) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
            "X-API-Key": self._get_api_key(remote_params) or "",
        }

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "max_new_tokens",
            "stop_strings",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        }

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=5, politeness_policy=60.0)
