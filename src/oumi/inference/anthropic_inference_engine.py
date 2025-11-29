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

    Supports:
    - Standard text generation
    - Usage tracking with token counts (including cache tokens)
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

        messages = [
            message for message in conversation.messages if message.role != Role.SYSTEM
        ]

        # Build request body
        # See https://docs.anthropic.com/claude/reference/messages_post
        body = {
            "model": model_params.model_name,
            "messages": self._get_list_of_message_json_dicts(
                messages, group_adjacent_same_role_turns=True
            ),
            "max_tokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
            "top_p": generation_params.top_p,
        }

        if system_message:
            body["system"] = system_message

        if generation_params.stop_strings is not None:
            body["stop_sequences"] = generation_params.stop_strings

        return body

    def _parse_usage_from_response(self, usage_dict: dict[str, Any]) -> dict[str, Any]:
        """Parse token usage from Anthropic API response.

        Args:
            usage_dict: The 'usage' dictionary from the API response.

        Returns:
            Dictionary with parsed usage information.
        """
        usage = {
            "prompt_tokens": usage_dict.get("input_tokens", 0),
            "completion_tokens": usage_dict.get("output_tokens", 0),
            "total_tokens": usage_dict.get("input_tokens", 0)
            + usage_dict.get("output_tokens", 0),
        }

        # Parse cache-specific tokens
        if "cache_creation_input_tokens" in usage_dict:
            usage["cache_creation_input_tokens"] = usage_dict[
                "cache_creation_input_tokens"
            ]
        if "cache_read_input_tokens" in usage_dict:
            usage["cache_read_input_tokens"] = usage_dict["cache_read_input_tokens"]

        return usage

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

        # Extract text content
        text_content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")

        # Create message
        new_message = Message(
            content=text_content or "",
            role=Role.ASSISTANT,
        )

        # Store usage information in metadata
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
            "top_p",
        }

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=5, politeness_policy=60.0)
