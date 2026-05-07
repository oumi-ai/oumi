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

"""HuggingFace Inference Providers engine implementation."""

from typing_extensions import override

from oumi.core.configs import RemoteParams
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class HuggingFaceInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference via the HuggingFace Inference Providers API.

    HuggingFace Inference Providers offer serverless, OpenAI-compatible access to
    hundreds of models hosted by HuggingFace and partner providers (Together AI,
    Fireworks, SambaNova, Cerebras, etc.).

    Authentication:
        Set the ``HF_TOKEN`` environment variable to a HuggingFace token with the
        ``Make calls to Inference Providers`` permission, or supply the token via
        ``RemoteParams.api_key`` / ``RemoteParams.api_key_env_varname``.

    Provider selection:
        The router parses an optional provider/policy suffix off the model name
        server-side. Pass it through on ``ModelParams.model_name`` unchanged:

        * ``"meta-llama/Llama-3.1-8B-Instruct"``          — auto-route (``:fastest``)
        * ``"meta-llama/Llama-3.1-8B-Instruct:cheapest"`` — prefer cheapest provider
        * ``"meta-llama/Llama-3.1-8B-Instruct:preferred"`` — your provider preference order
        * ``"meta-llama/Llama-3.1-8B-Instruct:together"`` — pin to Together AI

    References:
        https://huggingface.co/docs/inference-providers/index
    """

    @property
    @override
    def base_url(self) -> str | None:
        """Return the default base URL for the HuggingFace Inference Providers API."""
        return "https://router.huggingface.co/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """Return the default environment variable name for the HuggingFace token."""
        return "HF_TOKEN"

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters for the HuggingFace API."""
        return RemoteParams(num_workers=20, politeness_policy=0.0)
