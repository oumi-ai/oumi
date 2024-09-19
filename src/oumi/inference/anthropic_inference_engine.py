from typing import Any, Dict

from oumi.core.configs import GenerationConfig, RemoteParams
from oumi.core.types.turn import Conversation, Message, Role, Type
from oumi.inference.remote_inference_engine import RemoteInferenceEngine

_CONTENT_KEY: str = "content"
_ROLE_KEY: str = "role"


class AnthropicInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Anthropic API."""

    def _convert_conversation_to_api_input(
        self, conversation: Conversation, generation_config: GenerationConfig
    ) -> Dict[str, Any]:
        """Converts a conversation to an Anthropic API input."""
        system_messages = [
            message for message in conversation.messages if message.role == Role.SYSTEM
        ]
        if len(system_messages) > 0:
            system_message = system_messages[0].content
        else:
            system_message = None

        messages = [
            message for message in conversation.messages if message.role != Role.SYSTEM
        ]

        body = {
            "model": self._model,
            "messages": [
                {
                    _CONTENT_KEY: message.content,
                    _ROLE_KEY: message.role.value,
                }
                for message in messages
            ],
            "max_tokens": generation_config.max_new_tokens,
        }

        if system_message:
            body["system"] = system_message

        return body

    def _convert_api_output_to_conversation(
        self, response: Dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts an Anthropic API response to a conversation."""
        new_message = Message(
            content=response["content"][0]["text"],
            role=Role.ASSISTANT,
            type=Type.TEXT,
        )
        return Conversation(
            messages=[*original_conversation.messages, new_message],
            metadata=original_conversation.metadata,
            conversation_id=original_conversation.conversation_id,
        )

    def _get_request_headers(self, remote_params: RemoteParams) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-API-Key": self._get_api_key(remote_params) or "",
            "anthropic-version": "2023-06-01",  # latest version, see here
        }
