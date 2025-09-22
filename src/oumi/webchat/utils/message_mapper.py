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

"""Message format conversion utilities for WebChat."""

import time
from typing import Dict, List, Optional, Any

from oumi.core.types.conversation import Conversation, Message, Role


class MessageMapper:
    """Handles conversions between different message formats."""
    
    @staticmethod
    def openai_to_oumi(
        messages: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None
    ) -> Conversation:
        """Convert OpenAI-format messages to Oumi Conversation.
        
        Args:
            messages: List of OpenAI-format message dictionaries.
            system_prompt: Optional system prompt to prepend.
            
        Returns:
            Oumi Conversation object.
        """
        oumi_messages = []
        
        # Add system prompt if provided and not already in messages
        if system_prompt and not any(msg.get("role") == "system" for msg in messages):
            oumi_messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        
        # Convert messages
        for msg in messages:
            role_mapping = {
                "system": Role.SYSTEM,
                "user": Role.USER,
                "assistant": Role.ASSISTANT,
                "function": Role.FUNCTION,
                "tool": Role.FUNCTION,  # Map tool to function for compatibility
            }
            role = role_mapping.get(msg.get("role"), Role.USER)
            content = msg.get("content", "")
            
            # Skip empty messages
            if content:
                oumi_messages.append(Message(role=role, content=content))
        
        return Conversation(messages=oumi_messages)
    
    @staticmethod
    def oumi_to_openai(conversation: Conversation) -> List[Dict[str, Any]]:
        """Convert Oumi Conversation to OpenAI-format messages.
        
        Args:
            conversation: Oumi Conversation object.
            
        Returns:
            List of OpenAI-format message dictionaries.
        """
        openai_messages = []
        
        # Convert messages
        for msg in conversation.messages:
            role_mapping = {
                Role.SYSTEM: "system",
                Role.USER: "user",
                Role.ASSISTANT: "assistant",
                Role.FUNCTION: "function",
            }
            role = role_mapping.get(msg.role, "user")
            
            # Handle content based on type
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                # Try to join list content if it's a list of strings or content objects
                parts = []
                for item in msg.content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif hasattr(item, "content") and item.content:
                        parts.append(str(item.content))
                content = " ".join(parts)
            else:
                # Fallback for other content types
                content = str(msg.content)
            
            openai_messages.append({
                "role": role,
                "content": content
            })
        
        return openai_messages
    
    @staticmethod
    def webchat_to_oumi(
        messages: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None
    ) -> Conversation:
        """Convert WebChat-format messages to Oumi Conversation.
        
        Args:
            messages: List of WebChat-format message dictionaries.
            system_prompt: Optional system prompt to prepend.
            
        Returns:
            Oumi Conversation object.
        """
        oumi_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            oumi_messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        
        # Convert messages
        for msg in messages:
            role_mapping = {
                "system": Role.SYSTEM,
                "user": Role.USER,
                "assistant": Role.ASSISTANT,
            }
            role = role_mapping.get(msg.get("role"), Role.USER)
            content = msg.get("content", "")
            
            # Skip empty messages
            if content:
                oumi_messages.append(Message(role=role, content=content))
        
        return Conversation(messages=oumi_messages)
    
    @staticmethod
    def oumi_to_webchat(conversation: Conversation) -> List[Dict[str, Any]]:
        """Convert Oumi Conversation to WebChat-format messages.
        
        Args:
            conversation: Oumi Conversation object.
            
        Returns:
            List of WebChat-format message dictionaries.
        """
        webchat_messages = []
        
        # Convert messages
        for msg in conversation.messages:
            role_mapping = {
                Role.SYSTEM: "system",
                Role.USER: "user",
                Role.ASSISTANT: "assistant",
                Role.FUNCTION: "function",
            }
            role = role_mapping.get(msg.role, "user")
            
            # Handle content based on type
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                # Try to join list content if it's a list of strings or content objects
                parts = []
                for item in msg.content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif hasattr(item, "content") and item.content:
                        parts.append(str(item.content))
                content = " ".join(parts)
            else:
                # Fallback for other content types
                content = str(msg.content)
            
            webchat_messages.append({
                "role": role,
                "content": content,
                "timestamp": time.time()
            })
        
        return webchat_messages
    
    @staticmethod
    def extract_last_assistant_message(
        conversation: Conversation
    ) -> Optional[str]:
        """Extract the last assistant message from a conversation.
        
        Args:
            conversation: Oumi Conversation object.
            
        Returns:
            Content of the last assistant message, or None if not found.
        """
        for msg in reversed(conversation.messages):
            if msg.role == Role.ASSISTANT:
                if isinstance(msg.content, str):
                    return msg.content
                else:
                    return str(msg.content)
        return None
    
    @staticmethod
    def extract_last_user_message(
        conversation: Conversation
    ) -> Optional[str]:
        """Extract the last user message from a conversation.
        
        Args:
            conversation: Oumi Conversation object.
            
        Returns:
            Content of the last user message, or None if not found.
        """
        for msg in reversed(conversation.messages):
            if msg.role == Role.USER:
                if isinstance(msg.content, str):
                    return msg.content
                else:
                    return str(msg.content)
        return None
    
    @staticmethod
    def format_for_openai_api(
        conversation: Conversation,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format a conversation for the OpenAI API format response.
        
        Args:
            conversation: Oumi Conversation object.
            model: Model name to include in the response.
            
        Returns:
            Dictionary in OpenAI API response format.
        """
        # Extract the last assistant message
        response_content = MessageMapper.extract_last_assistant_message(conversation)
        if not response_content:
            response_content = "No response generated"
        
        # Extract the last user message for token counting
        user_content = MessageMapper.extract_last_user_message(conversation) or ""
        
        # Determine model name with a contextual fallback if not provided
        if not model:
            try:
                from oumi.webchat.utils.fallbacks import model_name_fallback
                model = model_name_fallback("message_mapper.model")
            except Exception:
                model = "Not found (<unknown>)"

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_content.split()),
                "completion_tokens": len(response_content.split()),
                "total_tokens": len(user_content.split()) + len(response_content.split()),
            },
        }
