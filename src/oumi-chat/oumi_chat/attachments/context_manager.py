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

"""Context window management for file attachments."""

from dataclasses import dataclass

import tiktoken


@dataclass
class ContextBudget:
    """Context window budget allocation."""

    total_tokens: int
    reserved_for_history: int
    reserved_for_response: int
    available_for_content: int
    safety_margin: int


@dataclass
class TokenEstimate:
    """Token count estimate for content."""

    content: str
    estimated_tokens: int
    fits_in_budget: bool
    overflow_amount: int = 0


class ContextWindowManager:
    """Manages context window allocation and token budgeting for file attachments.

    This class helps ensure that file attachments don't exceed the model's context
    window by providing token estimation, budget allocation, and overflow handling.
    """

    def __init__(
        self, max_context_length: int = 4096, model_name: str = "gpt-3.5-turbo"
    ):
        """Initialize the context manager.

        Args:
            max_context_length: Maximum context window size in tokens.
            model_name: Model name for tokenization (affects token counting).
        """
        self.max_context_length = max_context_length
        self.model_name = model_name

        # Try to get the appropriate tokenizer
        try:
            if "gpt" in model_name.lower():
                self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            elif "llama" in model_name.lower() or "qwen" in model_name.lower():
                # Use cl100k_base as a reasonable approximation for most models
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            else:
                # Default fallback
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Ultimate fallback if tiktoken fails
            self.tokenizer = None

    def calculate_budget(self, conversation_length: int = 0) -> ContextBudget:
        """Calculate the context budget allocation.

        Args:
            conversation_length: Current conversation length in tokens.

        Returns:
            ContextBudget with token allocations.
        """
        # Reserve space for conversation history (20% of total, minimum current length)
        history_reserve = max(int(self.max_context_length * 0.2), conversation_length)

        # Reserve space for model response (30% of total)
        response_reserve = int(self.max_context_length * 0.3)

        # Safety margin (5% of total)
        safety_margin = int(self.max_context_length * 0.05)

        # Available for file content
        available_for_content = (
            self.max_context_length - history_reserve - response_reserve - safety_margin
        )

        # Ensure we don't go negative
        available_for_content = max(0, available_for_content)

        return ContextBudget(
            total_tokens=self.max_context_length,
            reserved_for_history=history_reserve,
            reserved_for_response=response_reserve,
            available_for_content=available_for_content,
            safety_margin=safety_margin,
        )

    def estimate_tokens(self, content: str) -> int:
        """Estimate token count for given content.

        Args:
            content: Text content to estimate.

        Returns:
            Estimated token count.
        """
        if not content:
            return 0

        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(content))
            except Exception:
                pass

        # Fallback estimation: roughly 4 characters per token
        return len(content) // 4

    def check_content_fit(self, content: str, budget: ContextBudget) -> TokenEstimate:
        """Check if content fits within the available budget.

        Args:
            content: Content to check.
            budget: Context budget to check against.

        Returns:
            TokenEstimate with fit analysis.
        """
        estimated_tokens = self.estimate_tokens(content)
        fits = estimated_tokens <= budget.available_for_content
        overflow = max(0, estimated_tokens - budget.available_for_content)

        return TokenEstimate(
            content=content,
            estimated_tokens=estimated_tokens,
            fits_in_budget=fits,
            overflow_amount=overflow,
        )

    def truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token limit.

        Args:
            content: Content to truncate.
            max_tokens: Maximum tokens allowed.

        Returns:
            Truncated content that fits within the limit.
        """
        if not content:
            return content

        if self.tokenizer is not None:
            try:
                # Encode and truncate
                tokens = self.tokenizer.encode(content)
                if len(tokens) <= max_tokens:
                    return content

                # Truncate tokens and decode back
                truncated_tokens = tokens[:max_tokens]
                truncated_content = self.tokenizer.decode(truncated_tokens)

                # Add truncation indicator
                return truncated_content + "\n\n[Content truncated due to length...]"
            except Exception:
                pass

        # Fallback: character-based truncation
        max_chars = max_tokens * 4  # Rough estimate
        if len(content) <= max_chars:
            return content

        return content[:max_chars] + "\n\n[Content truncated due to length...]"

    def suggest_chunking_strategy(
        self, total_tokens: int, available_tokens: int
    ) -> dict:
        """Suggest a chunking strategy for oversized content.

        Args:
            total_tokens: Total tokens in the content.
            available_tokens: Available tokens in context budget.

        Returns:
            Dictionary with chunking suggestions.
        """
        if total_tokens <= available_tokens:
            return {"strategy": "no_chunking", "reason": "Content fits in context"}

        chunk_size = max(
            available_tokens - 100, available_tokens // 2
        )  # Leave some margin
        num_chunks = (total_tokens + chunk_size - 1) // chunk_size  # Ceiling division

        return {
            "strategy": "chunk_processing",
            "reason": (
                f"Content ({total_tokens} tokens) exceeds available space "
                f"({available_tokens} tokens)"
            ),
            "chunk_size": chunk_size,
            "estimated_chunks": num_chunks,
            "recommendation": (
                f"Process in {num_chunks} chunks of ~{chunk_size} tokens each. "
                f"This will require {num_chunks} separate interactions."
            ),
        }

    def format_budget_info(self, budget: ContextBudget) -> str:
        """Format budget information for display to user.

        Args:
            budget: Context budget to format.

        Returns:
            Formatted budget information string.
        """
        return f"""**Context Budget:**
• Total context window: {budget.total_tokens:,} tokens
• Reserved for conversation: {budget.reserved_for_history:,} tokens
• Reserved for response: {budget.reserved_for_response:,} tokens
• Available for attachments: {budget.available_for_content:,} tokens
• Safety margin: {budget.safety_margin:,} tokens"""
