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

from typing_extensions import override

from oumi.core.configs import InferenceConfig
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class OpenRouterInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the OpenRouter API.

    OpenRouter provides a unified API that gives access to hundreds of AI models
    through a single endpoint. It supports models from OpenAI, Anthropic, Google,
    Meta, and many other providers.

    Model names should use the provider/model format (e.g., "openai/gpt-4",
    "anthropic/claude-3-opus", "meta-llama/llama-3-70b-instruct").

    Documentation: https://openrouter.ai/docs
    """

    @property
    @override
    def base_url(self) -> str | None:
        """Return the default base URL for the OpenRouter API."""
        return "https://openrouter.ai/api/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """Return the default environment variable name for the OpenRouter API key."""
        return "OPENROUTER_API_KEY"

    @override
    def infer_batch(
        self,
        _conversations: list[Conversation],
        _inference_config: InferenceConfig | None = None,
    ) -> str:
        """Batch inference is not implemented for OpenRouter."""
        raise NotImplementedError(
            "Batch inference is not implemented for OpenRouter. "
            "Open an issue on GitHub if you'd like this feature."
        )
