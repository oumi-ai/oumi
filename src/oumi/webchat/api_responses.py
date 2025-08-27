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

"""Enhanced API response formatters and validators."""

import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from aiohttp import web
from pydantic import BaseModel, Field, ValidationError, field_validator

from oumi.utils.logging import logger


class ResponseStatus(str, Enum):
    """Standard API response statuses."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    PARTIAL = "partial"


class ErrorType(str, Enum):
    """Standard error types."""
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND_ERROR = "not_found_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    INTERNAL_ERROR = "internal_error"
    TIMEOUT_ERROR = "timeout_error"
    DEPENDENCY_ERROR = "dependency_error"


class ApiError(BaseModel):
    """Standard API error response."""
    type: ErrorType
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = Field(default_factory=time.time)


class ApiResponse(BaseModel):
    """Standard API response wrapper."""
    status: ResponseStatus
    data: Optional[Dict[str, Any]] = None
    error: Optional[ApiError] = None
    timestamp: float = Field(default_factory=time.time)
    request_id: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat completion request validation."""
    messages: List[Dict[str, str]]
    session_id: Optional[str] = "default"
    stream: bool = False
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=8192)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)


class CommandRequest(BaseModel):
    """Command execution request validation."""
    session_id: Optional[str] = "default"
    command: str = Field(min_length=1)
    args: List[str] = Field(default_factory=list)


class BranchRequest(BaseModel):
    """Branch operation request validation."""
    session_id: Optional[str] = "default"
    action: str
    branch_id: Optional[str] = None
    name: Optional[str] = None
    from_branch: Optional[str] = None
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        allowed_actions = {'create', 'switch', 'delete', 'list'}
        if v not in allowed_actions:
            raise ValueError(f'action must be one of {allowed_actions}')
        return v


class ResponseFormatter:
    """Enhanced API response formatter with validation."""

    @staticmethod
    def success(
        data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a success response."""
        response = ApiResponse(
            status=ResponseStatus.SUCCESS,
            data=data,
            request_id=request_id
        )
        return response.model_dump()

    @staticmethod
    def error(
        error_type: ErrorType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 400,
        request_id: Optional[str] = None
    ) -> tuple[Dict[str, Any], int]:
        """Create an error response with status code."""
        error_obj = ApiError(
            type=error_type,
            message=message,
            details=details
        )
        response = ApiResponse(
            status=ResponseStatus.ERROR,
            error=error_obj,
            request_id=request_id
        )
        return response.model_dump(), status_code

    @staticmethod
    def validation_error(
        validation_errors: List[Dict[str, Any]],
        request_id: Optional[str] = None
    ) -> tuple[Dict[str, Any], int]:
        """Create a validation error response."""
        return ResponseFormatter.error(
            error_type=ErrorType.VALIDATION_ERROR,
            message="Request validation failed",
            details={"validation_errors": validation_errors},
            status_code=422,
            request_id=request_id
        )

    @staticmethod
    def not_found(
        resource: str,
        request_id: Optional[str] = None
    ) -> tuple[Dict[str, Any], int]:
        """Create a not found error response."""
        return ResponseFormatter.error(
            error_type=ErrorType.NOT_FOUND_ERROR,
            message=f"{resource} not found",
            status_code=404,
            request_id=request_id
        )

    @staticmethod
    def internal_error(
        message: str = "Internal server error",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> tuple[Dict[str, Any], int]:
        """Create an internal server error response."""
        return ResponseFormatter.error(
            error_type=ErrorType.INTERNAL_ERROR,
            message=message,
            details=details,
            status_code=500,
            request_id=request_id
        )

    @staticmethod
    def rate_limited(
        retry_after: Optional[int] = None,
        request_id: Optional[str] = None
    ) -> tuple[Dict[str, Any], int]:
        """Create a rate limit error response."""
        details = {"retry_after_seconds": retry_after} if retry_after else None
        return ResponseFormatter.error(
            error_type=ErrorType.RATE_LIMIT_ERROR,
            message="Rate limit exceeded",
            details=details,
            status_code=429,
            request_id=request_id
        )


class RequestValidator:
    """Request validation utilities."""

    @staticmethod
    async def validate_chat_request(request: web.Request) -> ChatRequest:
        """Validate chat completion request."""
        try:
            data = await request.json()
            return ChatRequest(**data)
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                error_details.append({
                    "field": ".".join(str(x) for x in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"]
                })
            raise ValidationError(error_details)
        except Exception as e:
            raise ValueError(f"Invalid JSON: {str(e)}")

    @staticmethod
    async def validate_command_request(request: web.Request) -> CommandRequest:
        """Validate command execution request."""
        try:
            data = await request.json()
            return CommandRequest(**data)
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                error_details.append({
                    "field": ".".join(str(x) for x in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"]
                })
            raise ValidationError(error_details)
        except Exception as e:
            raise ValueError(f"Invalid JSON: {str(e)}")

    @staticmethod
    async def validate_branch_request(request: web.Request) -> BranchRequest:
        """Validate branch operation request."""
        try:
            data = await request.json()
            return BranchRequest(**data)
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                error_details.append({
                    "field": ".".join(str(x) for x in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"]
                })
            raise ValidationError(error_details)
        except Exception as e:
            raise ValueError(f"Invalid JSON: {str(e)}")

    @staticmethod
    def extract_session_id(request: web.Request, default: str = "default") -> str:
        """Extract session ID from request (query param or header)."""
        # Try query parameter first
        session_id = request.query.get("session_id")
        if session_id:
            return session_id
        
        # Try header
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            return session_id
            
        return default


def create_json_response(
    data: Dict[str, Any],
    status: int = 200,
    headers: Optional[Dict[str, str]] = None
) -> web.Response:
    """Create a JSON response with proper headers."""
    response_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Session-ID",
    }
    
    if headers:
        response_headers.update(headers)
    
    return web.json_response(data, status=status, headers=response_headers)


def handle_api_errors(func):
    """Decorator for handling common API errors."""
    async def wrapper(self, request: web.Request) -> web.Response:
        try:
            return await func(self, request)
        except ValidationError as e:
            response_data, status_code = ResponseFormatter.validation_error(e.errors())
            return create_json_response(response_data, status_code)
        except ValueError as e:
            response_data, status_code = ResponseFormatter.error(
                ErrorType.VALIDATION_ERROR,
                str(e),
                status_code=400
            )
            return create_json_response(response_data, status_code)
        except FileNotFoundError as e:
            response_data, status_code = ResponseFormatter.not_found(str(e))
            return create_json_response(response_data, status_code)
        except Exception as e:
            logger.error(f"Unhandled API error in {func.__name__}: {e}")
            response_data, status_code = ResponseFormatter.internal_error(
                details={"function": func.__name__, "error": str(e)}
            )
            return create_json_response(response_data, status_code)
    
    return wrapper