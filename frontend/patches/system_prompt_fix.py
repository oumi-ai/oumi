"""
Patch for fixing system prompt role alternation error in Oumi webchat server.

This patch ensures system prompts are only added once at the beginning of conversations
and don't break the required user/assistant role alternation pattern.

Apply this by replacing the relevant sections in:
/src/oumi/webchat/server.py
"""

def fix_system_prompt_handling():
    """
    Fix for the system prompt role alternation error.
    
    The issue is that system prompts are being added unconditionally,
    which can create duplicate system messages or break alternation patterns.
    
    Solution: Only add system prompt if there isn't already one in the conversation.
    """
    
    # REPLACE THIS SECTION (around line 580-582):
    # # Add system prompt if provided
    # if self.system_prompt:
    #     conversation_messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
    
    # WITH THIS:
    
    # Check if conversation already has a system message
    has_system_message = False
    if session.conversation_history:
        for msg in session.conversation_history:
            if msg.get("role") == "system":
                has_system_message = True
                break
    
    # Add system prompt only if not already present
    if self.system_prompt and not has_system_message:
        conversation_messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
    
    # ALSO REPLACE THE FALLBACK SECTION (around line 608-609):
    # # Add system prompt if provided
    # if self.system_prompt:
    
    # WITH THIS:
    
    # Always add system prompt for new conversations (fallback case)
    if self.system_prompt:
        conversation_messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))

def alternative_fix_proper_conversation_building():
    """
    Alternative approach: Build conversation properly from the start.
    
    This ensures system messages are always first and conversations
    maintain proper alternation patterns.
    """
    
    # REPLACE THE ENTIRE CONVERSATION BUILDING SECTION WITH:
    
    def build_conversation_with_system_prompt(self, session, latest_user_content):
        """Build conversation ensuring proper role alternation."""
        conversation_messages = []
        
        # 1. Add system prompt FIRST (if provided and not already in history)
        has_system_in_history = False
        if session.conversation_history:
            has_system_in_history = any(
                msg.get("role") == "system" 
                for msg in session.conversation_history
            )
        
        if self.system_prompt and not has_system_in_history:
            conversation_messages.append(
                Message(role=Role.SYSTEM, content=self.system_prompt)
            )
        
        # 2. Add conversation history (preserving existing system messages)
        if session.conversation_history:
            for msg in session.conversation_history:
                role_mapping = {
                    "system": Role.SYSTEM,
                    "user": Role.USER, 
                    "assistant": Role.ASSISTANT,
                }
                role = role_mapping.get(msg.get("role"), Role.USER)
                conversation_messages.append(
                    Message(role=role, content=msg.get("content", ""))
                )
        
        # 3. Add current user message
        conversation_messages.append(
            Message(role=Role.USER, content=latest_user_content)
        )
        
        return Conversation(messages=conversation_messages)

# Additional fix for other handlers that might have the same issue:

def fix_openai_api_handler():
    """Fix system prompt handling in OpenAI API handler (around line 514-520)."""
    
    # REPLACE:
    # # Add system prompt if provided
    # if self.system_prompt:
    #     from oumi.core.types.conversation import Message, Role
    #     oumi_messages.append(
    #         Message(role=Role.SYSTEM, content=self.system_prompt)
    #     )
    
    # WITH:
    
    # Check if there's already a system message
    has_system = any(msg.role == Role.SYSTEM for msg in oumi_messages)
    
    # Add system prompt only if not already present
    if self.system_prompt and not has_system:
        oumi_messages.insert(0, Message(role=Role.SYSTEM, content=self.system_prompt))

def fix_websocket_handler():
    """Fix system prompt handling in WebSocket handler (around line 962-966)."""
    
    # REPLACE:
    # # Add system prompt if configured
    # if self.system_prompt:
    #     oumi_messages.append(
    #         Message(role=Role.SYSTEM, content=self.system_prompt)
    #     )
    
    # WITH:
    
    # Check if there's already a system message
    has_system = any(msg.role == Role.SYSTEM for msg in oumi_messages)
    
    # Add system prompt only if not already present (and put it first)
    if self.system_prompt and not has_system:
        oumi_messages.insert(0, Message(role=Role.SYSTEM, content=self.system_prompt))

# Key principles for the fix:
# 1. System messages should always be FIRST in the conversation
# 2. Only add system prompt if there isn't already a system message
# 3. Use insert(0, ...) instead of append() to ensure proper ordering
# 4. Maintain user/assistant alternation after system messages
# 5. Handle both new conversations and continuing conversations properly