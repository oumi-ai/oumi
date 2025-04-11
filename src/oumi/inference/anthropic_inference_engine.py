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

    @override
    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts an Anthropic API response to a conversation."""
        new_message = Message(
            content=response[_CONTENT_KEY][0]["text"],
            role=Role.ASSISTANT,
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
            "top_p",
        }

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=5, politeness_policy=60.0)
