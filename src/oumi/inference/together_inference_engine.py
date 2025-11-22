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

from typing import Optional

from typing_extensions import override

from oumi.inference.openai_inference_engine import OpenAIInferenceEngine


class TogetherInferenceEngine(OpenAIInferenceEngine):
    """Engine for running inference against the Together AI API.

    Together.ai uses an OpenAI-compatible API, so we inherit from OpenAIInferenceEngine
    and only override the base URL and API key environment variable.

    Supports (inherited from OpenAIInferenceEngine):
    - Standard text generation
    - Streaming responses with usage tracking
    - Tool/function calling
    - Vision/multimodal inputs
    """

    @property
    @override
    def base_url(self) -> Optional[str]:
        """Return the default base URL for the Together AI API."""
        return "https://api.together.xyz/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> Optional[str]:
        """Return the default environment variable name for the Together AI API key."""
        return "TOGETHER_API_KEY"
