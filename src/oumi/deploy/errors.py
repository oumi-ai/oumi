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

"""Provider-neutral typed exception hierarchy for deployment API errors.

Raised by :func:`oumi.deploy.utils.raise_api_error` when a deployment
provider's control plane returns an error response. Catch
:class:`DeployApiError` (or a specific subclass) to branch on
retry-vs-user-error-vs-fallback instead of regex-matching message strings.

``str(exc)`` contains only the operation context and provider-reported
detail. Request method, URL, account IDs, and other internal attribution
stay on structured attributes and DEBUG logs.

Hierarchy::

    DeployApiError                  (base)
    ├── DeployInvalidRequestError   (400 / 422)
    ├── DeployNotFoundError         (404)
    ├── DeployRateLimitError        (429)
    └── DeployTransientError        (5xx — sanitized __str__)

This module is provider-neutral by design. Provider-specific subclasses
(for example ``FireworksUnsupportedHardwareError``) live next to their
client implementation — see :mod:`oumi.deploy.fireworks_errors` — and
are plumbed in via the ``classify_4xx`` hook on
:func:`~oumi.deploy.utils.raise_api_error`. Do NOT re-export
provider-specific classes from this module; consumers wanting a
provider-specific catch should import from the provider's module
explicitly.
"""


class DeployApiError(Exception):
    """Base class for typed errors from a deployment provider's API.

    Also used as the concrete class for unclassified 4xx statuses
    (401/403/409/etc.) where no modeled subclass applies.
    """

    def __init__(
        self,
        *,
        detail: str,
        status_code: int,
        method: str,
        url: str,
        context: str,
    ):
        """Stores structured attrs; renders ``"Failed to {context}: {detail}"``."""
        self.detail = detail
        self.status_code = status_code
        self.method = method
        self.url = url
        self.context = context
        super().__init__(f"Failed to {context}: {detail}")


class DeployInvalidRequestError(DeployApiError):
    """HTTP 400 / 422 — request rejected (usually user-actionable)."""


class DeployNotFoundError(DeployApiError):
    """HTTP 404 — resource missing."""


class DeployRateLimitError(DeployApiError):
    """HTTP 429 — rate-limited by the provider.

    The provider's response body can include internal identifiers (account
    IDs, resource paths); :meth:`__str__` returns a sanitized retry message
    rather than echoing :attr:`detail`. Raw detail, status code, method,
    and URL remain available on structured attributes for logging.
    """

    def __str__(self) -> str:
        """Sanitized retry message; raw detail/status/url remain on attributes."""
        return (
            "The deployment provider is currently rate-limiting requests. "
            "Please retry shortly."
        )


class DeployTransientError(DeployApiError):
    """HTTP 5xx — server-side / transient error.

    Provider 5xx responses often have empty or uninformative bodies, so
    :meth:`__str__` returns a sanitized retry message rather than echoing
    :attr:`detail` (which may be empty). The raw detail, status code,
    method, and URL remain available on structured attributes for logging.
    """

    def __str__(self) -> str:
        """Sanitized retry message; raw detail/status/url remain on attributes."""
        return (
            "The deployment service is temporarily unavailable. Please retry shortly."
        )
