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
from oumi.core.configs.params.generation_params import ReasoningEffort
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class OpenAIInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the OpenAI API.

    Supports:
    - Standard text generation
    - Reasoning tokens (o1, o3, o4 models)
    """

    # Reasoning models that require special handling
    REASONING_MODELS = {
        "o1-preview",
        "o1-mini",
        "o3-mini",
        "o4-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-chat-latest",
    }

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

    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if the model is a reasoning model."""
        return any(
            reasoning_model in model_name for reasoning_model in self.REASONING_MODELS
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
        # Handle reasoning model constraints
        if self._is_reasoning_model(model_params.model_name):
            generation_params = copy.deepcopy(generation_params)
            # Reasoning models don't support logit_bias
            generation_params.logit_bias = {}
            # Some reasoning models only support temperature = 1
            if model_params.model_name == "o1-preview":
                generation_params.temperature = 1.0

        api_input = super()._convert_conversation_to_api_input(
            conversation=conversation,
            generation_params=generation_params,
            model_params=model_params,
        )

        # Add reasoning effort for reasoning models
        if generation_params.reasoning_effort and self._is_reasoning_model(
            model_params.model_name
        ):
            effort_value = (
                generation_params.reasoning_effort.value
                if isinstance(generation_params.reasoning_effort, ReasoningEffort)
                else generation_params.reasoning_effort
            )
            api_input["reasoning_effort"] = effort_value

        return api_input

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
            "reasoning_effort",
            "seed",
            "stop_strings",
            "stop_token_ids",
            "temperature",
            "top_p",
        }
