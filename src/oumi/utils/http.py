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


import aiohttp

_NON_RETRIABLE_STATUS_CODES = {
    400,  # Bad Request
    401,  # Unauthorized
    403,  # Forbidden
    404,  # Not Found
    422,  # Unprocessable Entity
}


def is_non_retriable_status_code(status_code: int) -> bool:
    """Check if a status code is non-retriable."""
    return status_code in _NON_RETRIABLE_STATUS_CODES


async def get_failure_reason_from_response(
    response: aiohttp.ClientResponse,
) -> str:
    """Return a string describing the error from the provided response."""
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

    return error_msg
