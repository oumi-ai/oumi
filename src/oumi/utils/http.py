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

from typing import Any, Union

import aiohttp

_NON_RETRYABLE_STATUS_CODES = {
    400,  # Bad Request
    401,  # Unauthorized
    403,  # Forbidden
    404,  # Not Found
    422,  # Unprocessable Entity
}


def is_non_retryable_status_code(status_code: int) -> bool:
    """Check if a status code is non-retryable."""
    return status_code in _NON_RETRYABLE_STATUS_CODES


async def get_failure_reason_from_non_retriable_error(
    response: aiohttp.ClientResponse,
) -> str:
    """Handle non-retryable errors."""
    try:
        response_json = await response.json()
        if isinstance(response_json, list):
            response_json = response_json[0]
        error_msg = (
            response_json.get("error", {}).get("message")
            if response_json
            else f"HTTP {response.status}"
        )

    except Exception:
        error_msg = f"HTTP {response.status}"

    failure_reason = error_msg
    return f"Non-retryable error: {failure_reason}"


def get_non_200_retriable_failure_reason(
    response: aiohttp.ClientResponse,
    response_json: Union[dict[str, Any], list[dict[str, Any]]],
) -> str:
    """Handle non-200 retryable response."""
    # Handle error response
    if isinstance(response_json, list):
        # If the response is a list, it is likely an error message
        response_json = response_json[0]

    error_msg = response_json.get("error", {}).get("message") if response_json else None
    failure_reason = error_msg if error_msg else f"HTTP {response.status}"

    return failure_reason
