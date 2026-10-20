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


from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import aiohttp


class APIStatusError(RuntimeError):
    """An API error that preserves the HTTP status code."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        api_input: dict | None = None,
    ):
        """Initialize with a message and HTTP status code.

        Args:
            message: Human-readable error description.
            status_code: HTTP status code from the provider.
            api_input: The JSON request body sent to the provider, if available.
        """
        super().__init__(message)
        self.status_code = status_code
        self.api_input = api_input


_NON_RETRIABLE_STATUS_CODES = {
    400,  # Bad Request
    401,  # Unauthorized
    403,  # Forbidden
    404,  # Not Found
    422,  # Unprocessable Entity
}

_RETRIABLE_400_PATTERNS = [
    # OpenAI intermittently returns this under concurrent load; valid JSON succeeds
    # on retry. See LOU-1492.
    "could not parse the json body",
]


def is_non_retriable_status_code(
    status_code: int, error_message: str | None = None
) -> bool:
    """Check if a status code is non-retriable.

    Args:
        status_code: HTTP status code from the provider.
        error_message: Optional error message from the response body. When
            provided for a 400 status, known transient patterns are checked
            and treated as retriable.
    """
    if status_code not in _NON_RETRIABLE_STATUS_CODES:
        return False
    if status_code == 400 and error_message is not None:
        lower_msg = error_message.lower()
        if any(pattern in lower_msg for pattern in _RETRIABLE_400_PATTERNS):
            return False
    return True


def parse_retry_after(header_value: str | None, now: datetime) -> float | None:
    """Parse an RFC 7231 Retry-After value into seconds from ``now``.

    Handles both forms: delta-seconds ("120") and HTTP-date
    ("Wed, 21 Oct 2015 07:28:00 GMT"). Returns None when the value is absent or
    unparseable. Past dates and negative deltas clamp to 0.0.

    Args:
        header_value: The raw Retry-After header value, or None if absent.
        now: The reference time used to convert an HTTP-date to a delta.
    """
    if header_value is None:
        return None
    stripped_header = header_value.strip()
    if not stripped_header:
        return None
    try:
        return max(0.0, float(stripped_header))
    except (ValueError, TypeError):
        pass
    try:
        parsed = parsedate_to_datetime(stripped_header)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (parsed - now).total_seconds())


async def get_failure_reason_from_response(
    response: aiohttp.ClientResponse,
) -> str:
    """Return a string describing the error from the provided response."""
    try:
        response_json = await response.json()
        if isinstance(response_json, list):
            response_json = response_json[0]
        error_msg = (
            response_json.get("error", {}).get("message") if response_json else None
        )
        if error_msg is None:
            error_msg = f"HTTP {response.status}"

    except Exception:
        error_msg = f"HTTP {response.status}"

    return error_msg
