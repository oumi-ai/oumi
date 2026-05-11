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

"""Client for the Oumi Enterprise platform.

This module lets local ``oumi`` code (CLI, library) read and write resources
hosted in an Oumi Enterprise account: datasets, models, judges, evaluators,
recipes, deployments, and long-running operations.

Credentials are loaded from the ``OUMI_API_URL`` / ``OUMI_API_KEY`` /
``OUMI_PROJECT_ID`` environment variables, falling back to a credentials file
written by ``oumi platform login`` (default location:
``~/.config/oumi/credentials.json``).

Example:
    >>> from oumi.platform import get_default_client
    >>> client = get_default_client()
    >>> for ds in client.datasets.list():
    ...     print(ds["displayName"])
"""

from oumi.platform.client import Client, get_default_client
from oumi.platform.credentials import (
    Credentials,
    CredentialsNotFoundError,
    load_credentials,
    save_credentials,
)
from oumi.platform.exceptions import (
    PlatformAPIError,
    PlatformAuthError,
    PlatformError,
    PlatformOperationError,
)

__all__ = [
    "Client",
    "Credentials",
    "CredentialsNotFoundError",
    "PlatformAPIError",
    "PlatformAuthError",
    "PlatformError",
    "PlatformOperationError",
    "get_default_client",
    "load_credentials",
    "save_credentials",
]
