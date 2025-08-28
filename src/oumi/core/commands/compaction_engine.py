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

"""Context compaction engine for reducing conversation history size."""

from typing import Optional

from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role


class CompactionEngine:
    """Engine for compacting conversation history to reduce token usage."""

    COMPACT_PROMPT = """You are a conversation summarizer. Your task is to create a
concise summary of the conversation history provided below, preserving the most
important information, context, and any key decisions or conclusions.

Important guidelines:
1. Maintain factual accuracy
2. Preserve key technical details, code snippets, and specific instructions
3. Keep important context about the user's goals and requirements
4. Summarize repetitive exchanges concisely
5. Maintain the conversational flow and context

Conversation to summarize:
{conversation_text}

Please provide a concise summary that captures the essence of this conversation:"""

    def __init__(self, inference_engine: BaseInferenceEngine, model_config):
        """Initialize the compaction engine.

        Args:
            inference_engine: The inference engine to use for summarization.
            model_config: Model configuration.
        """
        self.inference_engine = inference_engine
        self.model_config = model_config

    def compact_conversation(
        self, conversation_history: list[dict], preserve_recent: int = 2
    ) -> tuple[list[dict], str]:
        """Compact the conversation history while preserving recent messages.

        Args:
            conversation_history: List of conversation messages.
            preserve_recent: Number of recent turns to preserve unchanged.

        Returns:
            Tuple of (compacted_history, summary_text)
        """
        if len(conversation_history) <= preserve_recent * 2:
            # Not enough history to compact
            return conversation_history, ""

        # Split history into parts to compact and preserve
        messages_to_compact = []
        messages_to_preserve = []

        # Count complete turns (user + assistant pairs)
        turn_count = 0
        for i in range(len(conversation_history) - 1, -1, -1):
            msg = conversation_history[i]
            if msg.get("role") == "assistant":
                turn_count += 1
                if turn_count > preserve_recent:
                    messages_to_compact = conversation_history[: i + 1]
                    messages_to_preserve = conversation_history[i + 1 :]
                    break

        if not messages_to_compact:
            return conversation_history, ""

        # Convert messages to text for summarization
        conversation_text = self._format_messages_for_summary(messages_to_compact)

        # Create summary using the model
        summary = self._generate_summary(conversation_text)

        if not summary:
            # Fallback if summarization fails
            return conversation_history, ""

        # Create new compacted history
        compacted_history = [
            {
                "role": "system",
                "content": f"[Previous conversation summary]:\n{summary}",
            }
        ] + messages_to_preserve

        return compacted_history, summary

    def estimate_token_reduction(
        self, original_history: list[dict], compacted_history: list[dict]
    ) -> dict[str, int]:
        """Estimate the token reduction from compaction.

        Args:
            original_history: Original conversation history.
            compacted_history: Compacted conversation history.

        Returns:
            Dict with token counts and reduction percentage.
        """
        original_text = ""
        for msg in original_history:
            if "content" in msg:
                original_text += str(msg["content"]) + "\n"

        compacted_text = ""
        for msg in compacted_history:
            if "content" in msg:
                compacted_text += str(msg["content"]) + "\n"

        # Rough token estimation (4 chars per token)
        original_tokens = len(original_text) // 4
        compacted_tokens = len(compacted_text) // 4
        reduction_percent = (
            ((original_tokens - compacted_tokens) / original_tokens * 100)
            if original_tokens > 0
            else 0
        )

        return {
            "original_tokens": original_tokens,
            "compacted_tokens": compacted_tokens,
            "tokens_saved": original_tokens - compacted_tokens,
            "reduction_percent": reduction_percent,
        }

    def _format_messages_for_summary(self, messages: list[dict]) -> str:
        """Format messages into text for summarization.

        Args:
            messages: List of messages to format.

        Returns:
            Formatted conversation text.
        """
        formatted_lines = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Skip system messages and attachments
            if role in ["system", "attachment"]:
                continue

            # Format the message
            if role == "user":
                formatted_lines.append(f"User: {content}")
            elif role == "assistant":
                formatted_lines.append(f"Assistant: {content}")

        return "\n\n".join(formatted_lines)

    def _generate_summary(self, conversation_text: str) -> Optional[str]:
        """Generate a summary of the conversation text.

        Args:
            conversation_text: The conversation text to summarize.

        Returns:
            Summary text or None if generation fails.
        """
        try:
            # Create the prompt with the conversation
            prompt = self.COMPACT_PROMPT.format(conversation_text=conversation_text)

            # Create a conversation for the summary request
            summary_conversation = Conversation(
                messages=[Message(role=Role.USER, content=prompt)]
            )

            # Generate the summary
            # Note: We're using a simplified inference config here
            # In production, you might want to use specific settings for summarization
            from oumi.core.configs import InferenceConfig

            summary_config = InferenceConfig(model=self.model_config)

            response = self.inference_engine.infer(
                input=[summary_conversation], inference_config=summary_config
            )

            # Extract the summary from the response
            if response and len(response) > 0:
                for message in response[0].messages:
                    if message.role == Role.ASSISTANT and isinstance(
                        message.content, str
                    ):
                        return message.content.strip()

            return None

        except Exception as e:
            # Log error but don't crash
            print(f"Error generating summary: {e}")
            return None
