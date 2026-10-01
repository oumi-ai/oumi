# Copyright 2026 - Oumi
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

"""OrcaRouter credential, catalog, and login support.

OrcaRouter is an OpenAI-compatible gateway. Inference and model discovery live on
the API origin (``https://api.orcarouter.ai/v1``); authentication lives on a
different origin (``https://www.orcarouter.ai``). Both origins are configurable
so self-hosted deployments work, but they are never derived from one another.
"""

from oumi.orcarouter.catalog import (
    CAPABILITY_MULTIMODAL_PREFIX,
    CatalogResult,
    ModelInfo,
    ModelSelector,
    OrcaRouterCapability,
    OrcaRouterCatalogError,
    discover_models,
    fetch_models,
    filter_models,
    resolve_model_selector,
    verified_seed_models,
)
from oumi.orcarouter.credentials import (
    API_BASE_URL_ENV_VAR,
    AUTH_BASE_URL_ENV_VAR,
    DEFAULT_API_BASE_URL,
    DEFAULT_AUTH_BASE_URL,
    ORCAROUTER_API_KEY_ENV_VAR,
    ORCAROUTER_BASE_URL_ENV_VAR,
    CredentialAdapter,
    CredentialResult,
    CredentialSource,
    CredentialStore,
    OrcaRouterApiKeyAdapter,
    OrcaRouterAuthError,
    OrcaRouterCredentialError,
    OrcaRouterSession,
    mask_api_key,
    resolve_api_base_url,
    resolve_auth_base_url,
)
from oumi.orcarouter.login import (
    DEFAULT_CALLBACK_PATH,
    OAuthPkceAdapter,
    PkceAuthorization,
    PkceExchangeError,
    build_authorize_url,
    exchange_code,
    run_oob_authorization,
    run_redirect_authorization,
)

__all__ = [
    "API_BASE_URL_ENV_VAR",
    "AUTH_BASE_URL_ENV_VAR",
    "CAPABILITY_MULTIMODAL_PREFIX",
    "DEFAULT_API_BASE_URL",
    "DEFAULT_AUTH_BASE_URL",
    "DEFAULT_CALLBACK_PATH",
    "ORCAROUTER_API_KEY_ENV_VAR",
    "ORCAROUTER_BASE_URL_ENV_VAR",
    "CatalogResult",
    "CredentialAdapter",
    "CredentialResult",
    "CredentialSource",
    "CredentialStore",
    "ModelInfo",
    "OrcaRouterApiKeyAdapter",
    "OrcaRouterAuthError",
    "OrcaRouterCapability",
    "OrcaRouterCatalogError",
    "OrcaRouterCredentialError",
    "OrcaRouterSession",
    "PkceAuthorization",
    "OAuthPkceAdapter",
    "PkceExchangeError",
    "build_authorize_url",
    "discover_models",
    "resolve_model_selector",
    "ModelSelector",
    "exchange_code",
    "fetch_models",
    "filter_models",
    "mask_api_key",
    "resolve_api_base_url",
    "resolve_auth_base_url",
    "run_oob_authorization",
    "run_redirect_authorization",
    "verified_seed_models",
]
