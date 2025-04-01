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

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine

_CONTENT_KEY: str = "content"
_ROLE_KEY: str = "role"


class RemoteVLLMInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against Remote vLLM."""

    @property
    @override
    def base_url(self) -> Optional[str]:
        """Return the default base URL for the Remote vLLM API."""
        return None

    @property
    @override
    def api_key_env_varname(self) -> Optional[str]:
        """Return the default environment variable name for the Remote vLLM API key."""
        return None

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "logit_bias",
            "presence_penalty",
            "seed",
            "stop_strings",
            "stop_token_ids",
            "temperature",
            "top_p",
            "guided_decoding",
            "max_new_tokens",
        }

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
        if model_params.adapter_model:
            model = model_params.adapter_model
        else:
            model = model_params.model_name

        api_input = {
            "model": model,
            "messages": self._get_list_of_message_json_dicts(
                conversation.messages, group_adjacent_same_role_turns=True
            ),
            "max_tokens": generation_params.max_new_tokens,
            # "max_completion_tokens": generation_params.max_new_tokens,
            # Future transition instead of `max_tokens`. See https://github.com/vllm-project/vllm/issues/9845
            "temperature": generation_params.temperature,
            "top_p": generation_params.top_p,
            "frequency_penalty": generation_params.frequency_penalty,
            "presence_penalty": generation_params.presence_penalty,
            "n": 1,  # Number of completions to generate for each prompt.
            "seed": generation_params.seed,
            "logit_bias": generation_params.logit_bias,
        }

        if generation_params.guided_decoding:
            if generation_params.guided_decoding.json:
                api_input["guided_json"] = generation_params.guided_decoding.json

            elif generation_params.guided_decoding.regex is not None:
                api_input["guided_regex"] = generation_params.guided_decoding.regex

            elif generation_params.guided_decoding.choice is not None:
                api_input["guided_choice"] = generation_params.guided_decoding.choice

        if generation_params.stop_strings:
            api_input["stop"] = generation_params.stop_strings
        if generation_params.stop_token_ids:
            api_input["stop_token_ids"] = generation_params.stop_token_ids

        return api_input
