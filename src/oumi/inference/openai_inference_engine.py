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


class OpenAIInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the OpenAI API.

    Supports:
    - Standard text generation
    - Usage tracking with token counts
    """

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

        return super()._convert_conversation_to_api_input(
            conversation=conversation,
            generation_params=generation_params,
            model_params=model_params,
        )

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

        # Create new message
        new_message = Message(
            content=content,
            role=Role(message_data["role"]),
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

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=50, politeness_policy=60.0)
