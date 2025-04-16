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


class SambanovaInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the SambaNova API.

    This class extends RemoteInferenceEngine to provide specific functionality
    for interacting with SambaNova's language models via their API. It handles
    the conversion of Oumi's Conversation objects to SambaNova's expected input
    format, as well as parsing the API responses back into Conversation objects.
    """

    @property
    @override
    def base_url(self) -> Optional[str]:
        """Return the default base URL for the SambaNova API."""
        return "https://api.sambanova.ai/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> Optional[str]:
        """Return the default environment variable name for the SambaNova API key."""
        return "SAMBANOVA_API_KEY"

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to a SambaNova API input.

        This method transforms an Oumi Conversation object into a format
        suitable for the SambaNova API. It handles the conversion of messages
        and generation parameters according to the API specification.

        Args:
            conversation: The Oumi Conversation object to convert.
            generation_params: Parameters for text generation.
            model_params: Model parameters to use during inference.

        Returns:
            Dict[str, Any]: A dictionary containing the formatted input for the
            SambaNova API, including the model, messages, and generation parameters.
        """
        # Build request body according to SambaNova API spec
        body = {
            "model": model_params.model_name,
            "messages": self._get_list_of_message_json_dicts(
                conversation.messages, group_adjacent_same_role_turns=False
            ),
            "max_tokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
            "top_p": generation_params.top_p,
            "stream": False,  # We don't support streaming yet
        }

        if generation_params.stop_strings:
            body["stop"] = generation_params.stop_strings

        return body

    @override
    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts a SambaNova API response to a conversation.

        Args:
            response: The API response to convert.
            original_conversation: The original conversation.

        Returns:
            Conversation: The conversation including the generated response.
        """
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("No choices found in API response")
        if len(choices) != 1:
            raise RuntimeError(
                "Sambanova API only supports one choice per response. "
                f"Got: {len(choices)}"
            )

        message = choices[0].get("message", {})
        if not message:
            raise RuntimeError("No message found in API response")

        new_message = Message(
            content=message.get("content", ""),
            role=Role.ASSISTANT,
        )

        return Conversation(
            messages=[*original_conversation.messages, new_message],
            metadata=original_conversation.metadata,
            conversation_id=original_conversation.conversation_id,
        )

    @override
    def _get_request_headers(self, remote_params: RemoteParams) -> dict[str, str]:
        """Get headers for the API request.

        Args:
            remote_params: Remote server parameters.

        Returns:
            Dict[str, str]: Headers for the API request.
        """
        headers = {
            "Content-Type": "application/json",
        }

        api_key = self._get_api_key(remote_params)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        return headers

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "max_new_tokens",
            "stop_strings",
            "temperature",
            "top_p",
        }
