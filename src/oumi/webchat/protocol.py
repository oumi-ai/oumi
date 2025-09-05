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

"""Protocol helpers for WebChat API normalization and validation.

This module provides utilities for normalizing message types and actions
across different API interfaces (WebSocket and REST), ensuring consistent
handling of aliases and variations in request formats.
"""

from typing import Dict, Optional, Any, Union

from aiohttp import web
from oumi.utils.logging import logger


def normalize_msg_type(s: str) -> str:
    """Normalize WebSocket message type strings.
    
    Args:
        s: The original message type string
        
    Returns:
        Normalized canonical message type string
    
    Examples:
        >>> normalize_msg_type("chat_message")
        "chat"
        >>> normalize_msg_type("system_stats")
        "system_monitor"
    """
    s = s.lower().strip()
    
    # Define mappings for message type aliases
    type_aliases = {
        "chat": ["chat", "chat_message"],
        "command": ["command", "cmd"],
        "get_branches": ["get_branches", "branches"],
        "system_monitor": ["system_monitor", "system_stats", "sys"]
    }
    
    # Find the canonical type for this alias
    for canonical, aliases in type_aliases.items():
        if s in aliases:
            return canonical
    
    # If no mapping found, just return the lowercased string and log
    if s and s not in ["ping"]:  # Don't log for common known types
        logger.debug(f"Unknown message type alias: '{s}'")
    
    return s


def normalize_branch_action(s: str) -> str:
    """Normalize branch action strings.
    
    Args:
        s: The original branch action string
        
    Returns:
        Normalized canonical action string
    
    Examples:
        >>> normalize_branch_action("new")
        "create"
        >>> normalize_branch_action("remove")
        "delete"
    """
    s = s.lower().strip()
    
    # Define mappings for branch action aliases
    action_aliases = {
        "create": ["create", "new"],
        "switch": ["switch", "select"],
        "delete": ["delete", "remove"]
    }
    
    # Find the canonical action for this alias
    for canonical, aliases in action_aliases.items():
        if s in aliases:
            return canonical
    
    # If no mapping found, just return the lowercased string and log
    if s:
        logger.debug(f"Unknown branch action alias: '{s}'")
    
    return s


def extract_session_id(
    request: Optional[web.Request] = None, 
    data: Optional[Dict[str, Any]] = None,
    required: bool = True
) -> Optional[str]:
    """Extract session_id from request or data, with validation.
    
    This function prioritizes session_id from query params over body data,
    and returns an error if session_id is required but missing.
    
    Args:
        request: Optional web request containing query parameters
        data: Optional dict containing request body data
        required: Whether session_id is required (raises ValueError if missing)
        
    Returns:
        Extracted session_id or None if not required and not found
        
    Raises:
        ValueError: If session_id is required but not found in request or data
    """
    session_id = None
    
    # Try to get session_id from query parameters
    if request and hasattr(request, "query"):
        session_id = request.query.get("session_id")
    
    # If not found in query, try to get from data
    if not session_id and data:
        session_id = data.get("session_id")
    
    # Validate session_id if required
    if required and not session_id:
        raise ValueError(
            "Missing required 'session_id' parameter. "
            "Include it as a query parameter or in the request body."
        )
    
    return session_id


def extract_branch_id(data: Dict[str, Any], required: bool = True) -> Optional[str]:
    """Extract branch_id from request data, with validation.
    
    Args:
        data: Dict containing request data
        required: Whether branch_id is required (raises ValueError if missing)
        
    Returns:
        Extracted branch_id or None if not required and not found
        
    Raises:
        ValueError: If branch_id is required but not found in data
    """
    branch_id = data.get("branch_id")
    
    # Validate branch_id if required
    if required and not branch_id:
        raise ValueError(
            "Missing required 'branch_id' parameter. "
            "Include it in the request body."
        )
    
    return branch_id


def get_valid_message_types() -> str:
    """Get a comma-separated list of all valid message types.
    
    Returns:
        String with all valid message types for error messages
    """
    return "chat, command, get_branches, system_monitor"


def get_valid_branch_actions() -> str:
    """Get a comma-separated list of all valid branch actions.
    
    Returns:
        String with all valid branch actions for error messages
    """
    return "create, switch, delete"