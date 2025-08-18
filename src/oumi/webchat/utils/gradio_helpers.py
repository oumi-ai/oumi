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

"""Utility functions for Gradio integration."""

from typing import Any, Dict, List, Tuple


def format_conversation_for_gradio(conversation: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert conversation history to Gradio messages format.
    
    Args:
        conversation: List of conversation messages with role/content.
        
    Returns:
        List of message dictionaries with 'role' and 'content' keys for Gradio.
    """
    gradio_messages = []
    
    for msg in conversation:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        if role in ['user', 'assistant']:
            # Add user and assistant messages directly
            gradio_messages.append({
                'role': role,
                'content': content
            })
            
        elif role == 'system':
            # Skip system messages in chat display
            continue
            
        elif role == 'attachment':
            # Format attachment info and add as user message
            attachment_info = msg.get('attachment_info', {})
            filename = attachment_info.get('filename', 'Unknown file')
            file_type = attachment_info.get('type', 'file')
            
            attachment_text = f"📎 Attached {file_type}: {filename}"
            gradio_messages.append({
                'role': 'user',
                'content': attachment_text
            })
    
    return gradio_messages


def format_message_for_display(message: Dict[str, Any]) -> str:
    """Format a single message for display.
    
    Args:
        message: Message dictionary with role and content.
        
    Returns:
        Formatted message string.
    """
    role = message.get('role', 'unknown')
    content = message.get('content', '')
    timestamp = message.get('timestamp')
    
    if role == 'user':
        prefix = "👤 **You:**"
    elif role == 'assistant':
        prefix = "🤖 **Assistant:**"
    elif role == 'system':
        prefix = "⚙️ **System:**"
    elif role == 'attachment':
        attachment_info = message.get('attachment_info', {})
        filename = attachment_info.get('filename', 'file')
        return f"📎 **Attached:** {filename}"
    else:
        prefix = f"**{role.title()}:**"
    
    # Add timestamp if available
    time_str = ""
    if timestamp:
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        time_str = f" _{dt.strftime('%H:%M:%S')}_"
    
    return f"{prefix}{time_str}\n{content}"


def extract_thinking_content(content: str) -> Tuple[str, str]:
    """Extract thinking content from assistant messages.
    
    Args:
        content: The full message content.
        
    Returns:
        Tuple of (thinking_content, final_content).
    """
    import re
    
    # Patterns for different thinking formats
    thinking_patterns = [
        (r'<thinking>(.*?)</thinking>', re.DOTALL),
        (r'<think>(.*?)</think>', re.DOTALL),
        (r'<reasoning>(.*?)</reasoning>', re.DOTALL),
        (r'<reflection>(.*?)</reflection>', re.DOTALL),
        (r'<!-- thinking(.*?)-->', re.DOTALL),
    ]
    
    thinking_content = ""
    final_content = content
    
    for pattern, flags in thinking_patterns:
        matches = re.findall(pattern, content, flags)
        if matches:
            thinking_content = matches[0].strip()
            final_content = re.sub(pattern, '', content, flags=flags).strip()
            break
    
    return thinking_content, final_content


def format_branch_info(branch: Dict[str, Any]) -> str:
    """Format branch information for display.
    
    Args:
        branch: Branch dictionary with metadata.
        
    Returns:
        Formatted branch info string.
    """
    name = branch.get('name', 'Unknown')
    message_count = branch.get('message_count', 0)
    created_at = branch.get('created_at', '')
    preview = branch.get('preview', 'No preview available')
    is_current = branch.get('is_current', False)
    
    # Format created time
    created_str = ""
    if created_at:
        try:
            import datetime
            if isinstance(created_at, str):
                dt = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                created_str = dt.strftime('%m/%d %H:%M')
        except:
            created_str = str(created_at)
    
    current_indicator = "→ " if is_current else ""
    
    return f"""
    **{current_indicator}{name}**
    - Messages: {message_count}
    - Created: {created_str}
    - Preview: {preview[:50]}{'...' if len(preview) > 50 else ''}
    """


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe use.
    
    Args:
        filename: Original filename.
        
    Returns:
        Sanitized filename.
    """
    import re
    import os
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Ensure it's not empty
    if not filename or filename == '_':
        filename = 'untitled'
    
    return filename


def get_file_type_emoji(filename: str) -> str:
    """Get emoji for file type based on filename.
    
    Args:
        filename: The filename.
        
    Returns:
        Appropriate emoji for the file type.
    """
    import os
    
    ext = os.path.splitext(filename.lower())[1]
    
    emoji_map = {
        '.pdf': '📄',
        '.txt': '📝', 
        '.md': '📝',
        '.json': '🔧',
        '.csv': '📊',
        '.xlsx': '📊',
        '.xls': '📊', 
        '.png': '🖼️',
        '.jpg': '🖼️',
        '.jpeg': '🖼️',
        '.gif': '🖼️',
        '.webp': '🖼️',
        '.mp3': '🎵',
        '.wav': '🎵',
        '.mp4': '🎬',
        '.mov': '🎬',
        '.py': '🐍',
        '.js': '📜',
        '.html': '🌐',
        '.css': '🎨',
        '.zip': '📦',
        '.tar': '📦',
        '.gz': '📦',
    }
    
    return emoji_map.get(ext, '📎')


def format_error_message(error: Exception, context: str = "") -> str:
    """Format an error message for user display.
    
    Args:
        error: The exception that occurred.
        context: Additional context about where the error occurred.
        
    Returns:
        Formatted error message.
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    formatted = f"❌ **{error_type}**"
    
    if context:
        formatted += f" in {context}"
    
    formatted += f": {error_msg}"
    
    return formatted


def create_progress_html(current: int, total: int, label: str = "Progress") -> str:
    """Create HTML for a progress bar.
    
    Args:
        current: Current progress value.
        total: Total progress value.
        label: Label for the progress bar.
        
    Returns:
        HTML string for progress bar.
    """
    if total <= 0:
        percentage = 0
    else:
        percentage = min(100, (current / total) * 100)
    
    return f"""
    <div style="margin: 8px 0;">
        <div style="font-size: 12px; margin-bottom: 4px;">
            {label}: {current}/{total} ({percentage:.1f}%)
        </div>
        <div style="background: #e0e0e0; border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="background: #007bff; height: 100%; width: {percentage}%; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """


def format_system_stats(stats: Dict[str, Any]) -> str:
    """Format system statistics for display.
    
    Args:
        stats: Dictionary of system statistics.
        
    Returns:
        Formatted HTML string.
    """
    gpu_usage = stats.get('gpu_usage', 0)
    memory_usage = stats.get('memory_usage', 0)
    context_used = stats.get('context_used', 0)
    context_total = stats.get('context_total', 4096)
    
    return f"""
    <div class="system-stats" style="font-family: monospace; font-size: 12px; background: #1e1e1e; color: #00ff00; padding: 8px; border-radius: 4px;">
        <div>GPU: {gpu_usage:.1f}%</div>
        <div>Memory: {memory_usage:.1f}MB</div>
        <div>Context: {context_used}/{context_total} tokens</div>
        <div>Free: {context_total - context_used} tokens</div>
    </div>
    """


def validate_command_syntax(command: str) -> Tuple[bool, str]:
    """Validate command syntax.
    
    Args:
        command: The command string to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not command.strip():
        return False, "Command cannot be empty"
    
    if not command.startswith('/'):
        return False, "Commands must start with '/'"
    
    # Basic syntax validation
    if command.count('(') != command.count(')'):
        return False, "Mismatched parentheses"
    
    # Check for common command names
    valid_commands = {
        'help', 'clear', 'delete', 'regen', 'attach', 'save', 'export',
        'import', 'swap', 'set', 'branch', 'switch', 'branches', 
        'branch_delete', 'show', 'render', 'compact', 'full_thoughts',
        'clear_thoughts', 'fetch', 'shell', 'list_engines'
    }
    
    command_name = command[1:].split('(')[0].strip()
    if command_name not in valid_commands:
        return False, f"Unknown command: {command_name}"
    
    return True, ""