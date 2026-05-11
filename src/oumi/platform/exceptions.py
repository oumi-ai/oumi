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

"""Exceptions raised by :mod:`oumi.platform`."""


class PlatformError(Exception):
    """Base error for all Oumi Enterprise platform interactions."""


class PlatformAuthError(PlatformError):
    """The platform rejected the request due to missing or invalid credentials."""


class PlatformAPIError(PlatformError):
    """The platform returned a non-success HTTP response.

    Attributes:
        status_code: The HTTP status code returned by the platform.
        response_body: The decoded response body, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: object | None = None,
    ):
        """Build a PlatformAPIError; see class docstring for attribute meanings."""
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class PlatformOperationError(PlatformError):
    """A platform-side long-running operation failed or was cancelled.

    Attributes:
        operation_id: The id of the operation that failed.
        status: The terminal status of the operation (``failed`` or ``cancelled``).
        operation: The full operation payload as returned by the platform.
    """

    def __init__(
        self,
        message: str,
        *,
        operation_id: str | int,
        status: str,
        operation: dict,
    ):
        """Build a PlatformOperationError; see class docstring for fields."""
        super().__init__(message)
        self.operation_id = operation_id
        self.status = status
        self.operation = operation
