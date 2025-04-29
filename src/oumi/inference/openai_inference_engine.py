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
import os
from typing import Any, Optional, Tuple

from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class OpenAIInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the OpenAI API."""
    
    @classmethod
    @override
    def check(cls) -> Tuple[bool, str]:
        """Checks if the OpenAI API credentials are configured.
        
        Verifies:
        1. If the OPENAI_API_KEY environment variable is set
        2. If the API key looks valid (basic format check)
        
        Returns:
            Tuple[bool, str]: Whether OpenAI API is properly configured and why
        """
        # Use hardcoded env var name instead of the class property
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            return (False, "OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        
        # Basic format check - OpenAI keys usually start with "sk-" and are ~51 chars
        if not api_key.startswith("sk-") or len(api_key) < 40:
            return (False, "OpenAI API key format appears invalid. Keys should start with 'sk-'.")
        
        return (True, "OpenAI API key is configured")

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

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=50, politeness_policy=60.0)
