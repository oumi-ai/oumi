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

"""Response formatting utilities for WebChat API."""

import time
from typing import Dict, Optional, Any, Tuple, Union

from aiohttp import web


class ResponseBuilder:
    """Helper for building consistent API responses."""
    
    @staticmethod
    def success(
        data: Optional[Dict[str, Any]] = None,
        status: int = 200,
        message: Optional[str] = None
    ) -> web.Response:
        """Build a successful API response.
        
        Args:
            data: Optional data to include in the response.
            status: HTTP status code.
            message: Optional success message.
            
        Returns:
            JSON response with consistent structure.
        """
        response_data = {
            "success": True,
            "timestamp": time.time()
        }
        
        if data:
            response_data["data"] = data
        
        if message:
            response_data["message"] = message
        
        return web.json_response(response_data, status=status)
    
    @staticmethod
    def error(
        message: str,
        status: int = 400,
        error_type: str = "general_error",
        details: Optional[Dict[str, Any]] = None
    ) -> web.Response:
        """Build an error API response.
        
        Args:
            message: Error message.
            status: HTTP status code.
            error_type: Type of error.
            details: Optional error details.
            
        Returns:
            JSON response with consistent structure.
        """
        response_data = {
            "success": False,
            "error": {
                "type": error_type,
                "message": message
            },
            "timestamp": time.time()
        }
        
        if details:
            response_data["error"]["details"] = details
        
        return web.json_response(response_data, status=status)
    
    @staticmethod
    def bad_request(message: str, details: Optional[Dict[str, Any]] = None) -> web.Response:
        """Build a bad request error response.
        
        Args:
            message: Error message.
            details: Optional error details.
            
        Returns:
            JSON response with status 400.
        """
        return ResponseBuilder.error(
            message=message,
            status=400,
            error_type="bad_request",
            details=details
        )
    
    @staticmethod
    def not_found(message: str) -> web.Response:
        """Build a not found error response.
        
        Args:
            message: Error message.
            
        Returns:
            JSON response with status 404.
        """
        return ResponseBuilder.error(
            message=message,
            status=404,
            error_type="not_found"
        )
    
    @staticmethod
    def server_error(message: str, details: Optional[Dict[str, Any]] = None) -> web.Response:
        """Build a server error response.
        
        Args:
            message: Error message.
            details: Optional error details.
            
        Returns:
            JSON response with status 500.
        """
        return ResponseBuilder.error(
            message=message,
            status=500,
            error_type="server_error",
            details=details
        )
    
    @staticmethod
    def json_response(
        data: Dict[str, Any],
        status: int = 200,
        headers: Optional[Dict[str, str]] = None
    ) -> web.Response:
        """Create a JSON response with standard headers.
        
        Args:
            data: Response data.
            status: HTTP status code.
            headers: Optional additional headers.
            
        Returns:
            JSON response with CORS and other standard headers.
        """
        response_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PUT, DELETE",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Session-ID, X-Requested-With",
        }
        
        if headers:
            response_headers.update(headers)
        
        return web.json_response(data, status=status, headers=response_headers)


class EnhancedResponseFallback:
    """Fallback for enhanced response functionality when not available."""
    
    @staticmethod
    def create_standard_response(
        data: Union[Dict[str, Any], Tuple[Dict[str, Any], int]],
        success: bool = True
    ) -> web.Response:
        """Create a response with standard structure, with fallback for enhanced responses.
        
        Args:
            data: Response data or tuple of (data, status_code).
            success: Whether this is a successful response.
            
        Returns:
            JSON response with consistent structure.
        """
        # Check if data is a tuple of (response_data, status_code)
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], dict) and isinstance(data[1], int):
            response_data, status_code = data
        else:
            response_data, status_code = data, 200
        
        # Try to detect if the data is already in enhanced format
        if "status" in response_data and response_data["status"] in ["success", "error"]:
            # Already in enhanced format, just return it
            return ResponseBuilder.json_response(response_data, status_code)
        
        # Wrap in standard structure
        wrapper = {
            "success": success,
            "timestamp": time.time()
        }
        
        if success:
            wrapper["data"] = response_data
        else:
            wrapper["error"] = response_data
        
        return ResponseBuilder.json_response(wrapper, status_code)