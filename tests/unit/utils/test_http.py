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

from unittest.mock import AsyncMock

import aiohttp
import pytest

from oumi.utils.http import (
    get_failure_reason_from_response,
    is_non_retriable_status_code,
)


@pytest.mark.parametrize(
    "status_code,expected",
    [
        (400, True),  # Bad Request
        (401, True),  # Unauthorized
        (403, True),  # Forbidden
        (404, True),  # Not Found
        (422, True),  # Unprocessable Entity
        (500, False),  # Server Error
        (502, False),  # Bad Gateway
        (503, False),  # Service Unavailable
        (429, False),  # Too Many Requests
    ],
)
def test_is_non_retryable_status_code(status_code: int, expected: bool):
    """Test identification of non-retryable status codes."""
    assert is_non_retriable_status_code(status_code) == expected


@pytest.mark.parametrize(
    "status_code,error_message,expected",
    [
        # Transient 400 patterns should be retriable (return False)
        (400, "We could not parse the JSON body of your request", False),
        (400, "we COULD NOT PARSE the json body of your request", False),
        (400, "Error: could not parse the json body - try again", False),
        # Non-transient 400s should still be non-retriable (return True)
        (400, "Invalid model: gpt-nonexistent", True),
        (400, "Missing required parameter: messages", True),
        (400, "", True),
        # 400 with no error_message (backward compatible) should be non-retriable
        (400, None, True),
        # Other non-retriable codes ignore error_message entirely
        (401, "could not parse the json body", True),
        (403, "could not parse the json body", True),
        (404, "could not parse the json body", True),
        (422, "could not parse the json body", True),
        # Retriable codes stay retriable regardless of message
        (500, "could not parse the json body", False),
        (429, "could not parse the json body", False),
    ],
)
def test_is_non_retryable_status_code_with_error_message(
    status_code: int, error_message: str | None, expected: bool
):
    """Test that known transient 400 patterns are treated as retriable."""
    assert is_non_retriable_status_code(status_code, error_message) == expected


@pytest.mark.asyncio
async def test_get_failure_reason_from_response_with_json_response():
    """Test handling of non-retryable errors with JSON response."""
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.status = 400
    mock_response.json.return_value = {"error": {"message": "Invalid request"}}

    result = await get_failure_reason_from_response(mock_response)
    assert result == "Invalid request"


@pytest.mark.asyncio
async def test_get_failure_reason_from_response_with_list_response():
    """Test handling of non-retryable errors with list response."""
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.status = 400
    mock_response.json.return_value = [{"error": {"message": "Invalid request"}}]

    result = await get_failure_reason_from_response(mock_response)
    assert result == "Invalid request"


@pytest.mark.asyncio
async def test_get_failure_reason_from_response_with_empty_response():
    """Test handling of non-retryable errors with empty response."""
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.status = 400
    mock_response.json.return_value = {}

    result = await get_failure_reason_from_response(mock_response)
    assert result == "HTTP 400"


@pytest.mark.asyncio
async def test_get_failure_reason_from_response_with_json_error():
    """Test handling of non-retryable errors when JSON parsing fails."""
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.status = 400
    mock_response.json.side_effect = Exception("JSON decode error")

    result = await get_failure_reason_from_response(mock_response)
    assert result == "HTTP 400"
