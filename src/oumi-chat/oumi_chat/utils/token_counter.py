"""Token counting utilities for conversation history."""

from typing import Any


def count_conversation_tokens(
    conversation_history: list[dict[str, Any]],
    context_window_manager=None,
) -> int:
    """Count tokens in conversation history.

    Uses context_window_manager for accurate tiktoken-based estimation if available,
    otherwise falls back to character-based estimation.

    Args:
        conversation_history: List of conversation messages (dicts with 'role' and 'content').
        context_window_manager: Optional ContextWindowManager for accurate token counting.

    Returns:
        Estimated token count for the conversation.
    """
    if context_window_manager:
        # Accurate tiktoken-based estimation
        conversation_text = ""
        for msg in conversation_history:
            if isinstance(msg, dict):
                # Handle regular messages
                if "content" in msg:
                    conversation_text += str(msg["content"]) + "\n"
                # Handle attachment messages
                elif msg.get("role") == "attachment" and "text_content" in msg:
                    conversation_text += str(msg["text_content"]) + "\n"
                elif msg.get("role") == "attachment" and "content" in msg:
                    conversation_text += str(msg["content"]) + "\n"

        return context_window_manager.estimate_tokens(conversation_text)

    # Fallback: character-based estimation
    total_chars = 0
    for msg in conversation_history:
        if msg.get("role") == "attachment":
            content = msg.get("text_content", "") or msg.get("content", "")
        else:
            content = msg.get("content", "")
        total_chars += len(str(content))

    # Rough estimation: ~4 characters per token
    return total_chars // 4
