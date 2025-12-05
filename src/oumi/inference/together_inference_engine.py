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

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.logging import logger


class TogetherInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Together AI API."""

    @property
    @override
    def base_url(self) -> Optional[str]:
        """Return the default base URL for the Together API."""
        return "https://api.together.xyz/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> Optional[str]:
        """Return the default environment variable name for the Together API key."""
        return "TOGETHER_API_KEY"

    @override
    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts an API response to a conversation.

        Args:
            response: The API response to convert.
            original_conversation: The original conversation.

        Returns:
            Conversation: The conversation including the generated response.
        """
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return super()._convert_api_output_to_conversation(
                response, original_conversation
            )

        if "reasoning" in message and "</think>" not in message.get("content", ""):
            logger.debug(
                "Concatenating `reasoning` to `content` for message: %s", message
            )
            response_with_reasoning = copy.deepcopy(response)
            msg = response_with_reasoning["choices"][0]["message"]
            # check if message contains "reasoning" and reasoning is not
            # in the message content
            # Case 1: if reasoning in message but </think> is not in the content field,
            # add it to the content
            # Case 2: if reasoning in message, but </think> is already in the content,
            # do not add it again
            # Case 3: if reasoning not in message and </think> is in the content,
            # do not add it again
            # Case 4: if reasoning not in message and </think> is not in the content,
            # the content is non-reasoning
            msg["content"] = f"<think>{msg['reasoning']}</think> {msg['content']}"
            return super()._convert_api_output_to_conversation(
                response_with_reasoning, original_conversation
            )

        return super()._convert_api_output_to_conversation(
            response, original_conversation
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
        api_input = super()._convert_conversation_to_api_input(
            conversation=conversation,
            generation_params=generation_params,
            model_params=model_params,
        )

        # Then layer on Together-specific / remote-specific kwargs
        remote_params = self._remote_params
        if remote_params.api_kwargs:
            api_input.update(
                remote_params.api_kwargs
            )  # e.g. {"reasoning": {"enabled": True}}

        return api_input
