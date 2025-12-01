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

import pydantic
from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.conversation_utils import (
    convert_message_to_json_content_list,
)
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
        if "error" in response:
            raise RuntimeError(
                f"API error: {response['error'].get('message', response['error'])}"
            )
        if "choices" not in response or not response["choices"]:
            raise RuntimeError(f"No choices found in API response: {response}")
        message = response["choices"][0].get("message")
        if not message:
            raise RuntimeError(f"No message found in API response: {response}")

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
        if "reasoning" in message and "</think>" not in message["content"]:
            logger.debug(
                f"Concatenating `reasoning` to `content` for message: {message}"
            )
            return Conversation(
                messages=[
                    *original_conversation.messages,
                    Message(
                        content=f"<think>{message['reasoning']}</think> {message['content']}",  # noqa: E501
                        role=Role(message["role"]),
                    ),
                ],
                metadata=original_conversation.metadata,
                conversation_id=original_conversation.conversation_id,
            )
        return Conversation(
            messages=[
                *original_conversation.messages,
                Message(
                    content=message["content"],
                    role=Role(message["role"]),
                ),
            ],
            metadata=original_conversation.metadata,
            conversation_id=original_conversation.conversation_id,
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
        # Mandatory generation parameters.
        generation_params_dict = {
            "max_completion_tokens": generation_params.max_new_tokens,
            "seed": generation_params.seed,
            "temperature": generation_params.temperature,
            "top_p": generation_params.top_p,
            "frequency_penalty": generation_params.frequency_penalty,
            "presence_penalty": generation_params.presence_penalty,
        }

        # Optional generation parameters.
        if generation_params.logit_bias:
            generation_params_dict["logit_bias"] = generation_params.logit_bias
        if generation_params.stop_strings:
            generation_params_dict["stop"] = generation_params.stop_strings
        if generation_params.stop_token_ids:
            generation_params_dict["stop_token_ids"] = generation_params.stop_token_ids
        if generation_params.min_p:
            generation_params_dict["min_p"] = generation_params.min_p

        remote_params = self._remote_params

        api_input = {
            "model": model_params.model_name,
            "messages": [
                {
                    "content": convert_message_to_json_content_list(message),
                    "role": message.role.value,
                }
                for message in conversation.messages
            ],
            "n": 1,  # Number of completions to generate for each prompt.
            **(remote_params.api_kwargs or {}),  # "reasoning": {"enabled": True},
            **generation_params_dict,
        }

        if generation_params.guided_decoding:
            json_schema = generation_params.guided_decoding.json

            if json_schema is None:
                raise ValueError(
                    "Only JSON schema guided decoding is supported, got '%s'",
                    generation_params.guided_decoding,
                )

            if isinstance(json_schema, type) and issubclass(
                json_schema, pydantic.BaseModel
            ):
                schema_name = json_schema.__name__
                schema_value = json_schema.model_json_schema()
            elif isinstance(json_schema, dict):
                # Use a generic name if no schema is provided.
                schema_name = "Response"
                schema_value = json_schema
            elif isinstance(json_schema, str):
                # Use a generic name if no schema is provided.
                schema_name = "Response"
                # Try to parse as JSON string
                schema_value = json.loads(json_schema)
            else:
                raise ValueError(
                    f"Got unsupported JSON schema type: {type(json_schema)}"
                    "Please provide a Pydantic model or a JSON schema as a "
                    "string or dict."
                )

            api_input["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema_value,
                },
            }

        return api_input
