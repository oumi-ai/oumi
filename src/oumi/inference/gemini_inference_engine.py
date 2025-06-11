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

from typing import Any

from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation
from oumi.inference.gcp_inference_engine import (
    _convert_guided_decoding_config_to_api_input,
)
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class GoogleGeminiInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against Gemini API."""

    base_url = (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    """The base URL for the Gemini API."""

    api_key_env_varname = "GEMINI_API_KEY"
    """The environment variable name for the Gemini API key."""

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to an Gemini API input.

        Documentation: https://ai.google.dev/docs

        Args:
            conversation: The conversation to convert.
            generation_params: Parameters for generation during inference.
            model_params: Model parameters to use during inference.

        Returns:
            Dict[str, Any]: A dictionary representing the Gemini input.
        """
        api_input = {
            "model": model_params.model_name,
            "messages": self._get_list_of_message_json_dicts(
                conversation.messages, group_adjacent_same_role_turns=True
            ),
            "max_completion_tokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
            "top_p": generation_params.top_p,
            "n": 1,  # Number of completions to generate for each prompt.
        }

        if generation_params.stop_strings:
            api_input["stop"] = generation_params.stop_strings

        if generation_params.guided_decoding:
            api_input["response_format"] = _convert_guided_decoding_config_to_api_input(
                generation_params.guided_decoding
            )

        return api_input

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "guided_decoding",
            "max_new_tokens",
            "stop_strings",
            "temperature",
            "top_p",
        }

    @override
    def infer_batch(
        self, conversations: list[Conversation], inference_config: dict[str, Any]
    ) -> str:
        """Run inference on a batch of conversations.

        Args:
            conversations: The batch of conversations to infer on.
            inference_config: The inference configuration.

        Returns:
            str: The batch ID.
        """
        raise NotImplementedError("Batch inference is not supported for Gemini API.")

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=2, politeness_policy=60.0)
