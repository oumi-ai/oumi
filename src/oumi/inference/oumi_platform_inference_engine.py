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

"""Inference engine targeting the Oumi Enterprise platform.

The platform exposes an OpenAI-compatible ``/inference/v1/chat/completions``
endpoint, so this engine is a thin subclass of :class:`RemoteInferenceEngine`
that points at the user's platform URL and reuses their platform credentials.
"""

import os

from typing_extensions import override

from oumi.core.configs import RemoteParams
from oumi.inference.remote_inference_engine import RemoteInferenceEngine

_DEFAULT_API_URL = "https://api.oumi.ai"
_CHAT_COMPLETIONS_PATH = "/inference/v1/chat/completions"


class OumiPlatformInferenceEngine(RemoteInferenceEngine):
    """Engine for inference against an Oumi Enterprise hosted endpoint.

    The base URL is derived from the ``OUMI_API_URL`` environment variable
    (or the credentials file, via :mod:`oumi.platform.credentials`). The API
    key is read from ``OUMI_API_KEY`` like every other platform call.
    """

    @property
    @override
    def base_url(self) -> str | None:
        """Return the platform's chat-completions URL."""
        api_url = os.environ.get("OUMI_API_URL")
        if not api_url:
            api_url = _try_load_api_url_from_credentials_file()
        return (api_url or _DEFAULT_API_URL).rstrip("/") + _CHAT_COMPLETIONS_PATH

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """Return the environment variable name for the platform API key."""
        return "OUMI_API_KEY"

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Defaults tuned for a managed multi-tenant endpoint.

        The same conservative defaults as :class:`OpenAIInferenceEngine`; the
        platform applies its own server-side rate limiting on top.
        """
        return RemoteParams(num_workers=50, politeness_policy=60.0)


def _try_load_api_url_from_credentials_file() -> str | None:
    """Fall back to the platform credentials file when ``OUMI_API_URL`` is unset.

    Returns ``None`` if the file is missing, unreadable, or doesn't include an
    ``api_url`` entry. Never raises: credentials-file problems should not
    block plain inference if the user passed an api_url via RemoteParams.
    """
    try:
        from oumi.platform.credentials import (
            CredentialsNotFoundError,
            load_credentials,
        )

        try:
            return load_credentials().api_url
        except CredentialsNotFoundError:
            return None
        except Exception:
            return None
    except ImportError:
        return None
