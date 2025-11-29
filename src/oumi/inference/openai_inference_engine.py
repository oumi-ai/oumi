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
import json
from collections.abc import AsyncGenerator
from typing import Any, Optional

import aiohttp
from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation
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

        api_input = super()._convert_conversation_to_api_input(
            conversation=conversation,
            generation_params=generation_params,
            model_params=model_params,
        )

        # Add streaming parameters
        if generation_params.stream:
            api_input["stream"] = True
            if generation_params.stream_options:
                api_input["stream_options"] = generation_params.stream_options
            else:
                # Enable usage tracking for streaming by default
                api_input["stream_options"] = {"include_usage": True}

        return api_input

    async def _stream_api_response(
        self,
        response: aiohttp.ClientResponse,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream SSE chunks from the API response.

        Args:
            response: The aiohttp response object.

        Yields:
            Parsed JSON chunks from the stream.
        """
        async for line in response.content:
            line = line.decode("utf-8").strip()

            if not line or line == "data: [DONE]":
                continue

            if line.startswith("data: "):
                line = line[6:]  # Remove "data: " prefix

                try:
                    chunk = json.loads(line)
                    yield chunk
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse streaming chunk: {line[:100]}")
                    continue

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=50, politeness_policy=60.0)

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "guided_decoding",
            "logit_bias",
            "max_new_tokens",
            "min_p",
            "presence_penalty",
            "seed",
            "stop_strings",
            "stop_token_ids",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
        }
