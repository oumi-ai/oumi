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

"""Lambda AI inference engine implementation."""

from typing import Optional

from typing_extensions import override

from oumi.core.configs import RemoteParams
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class LambdaInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Lambda AI API.

    This class extends RemoteInferenceEngine to provide specific functionality
    for interacting with Lambda AI's language models via their API. It handles
    the conversion of Oumi's Conversation objects to Lambda AI's expected input
    format, as well as parsing the API responses back into Conversation objects.
    """

    @property
    @override
    def base_url(self) -> Optional[str]:
        """Return the default base URL for the Lambda AI API."""
        return "https://api.lambda.ai/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> Optional[str]:
        """Return the default environment variable name for the Lambda AI API key."""
        return "LAMBDA_API_KEY"

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=20, politeness_policy=60.0)
