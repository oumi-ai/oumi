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

"""Deduplication analyzer implementation and result model."""

import hashlib
import logging
from collections import defaultdict

from pydantic import BaseModel, Field

from oumi.analyze.base import DatasetAnalyzer
from oumi.core.types.conversation import Conversation

__all__ = ["DuplicateGroup", "DeduplicationResult", "DeduplicationAnalyzer"]

logger = logging.getLogger(__name__)


class DuplicateGroup(BaseModel):
    """A group of duplicate or near-duplicate conversations.

    Represents a cluster of conversations that are identical or very similar.
    The first index is typically the "canonical" version to keep.
    """

    indices: list[int] = Field(
        description="Indices of duplicate conversations in this group"
    )
    similarity: float = Field(
        description="Average similarity score within the group (1.0 = exact match)"
    )
    sample_text: str | None = Field(
        default=None, description="Sample text from the first conversation in the group"
    )


class DeduplicationResult(BaseModel):
    """Result model for dataset-level deduplication analysis.

    Contains statistics about duplicate conversations and groups of duplicates
    that can be used to clean the dataset.

    Example:
        >>> result = DeduplicationResult(
        ...     total_conversations=1000,
        ...     unique_conversations=950,
        ...     duplicate_count=50,
        ...     duplicate_ratio=0.05,
        ...     num_duplicate_groups=30,
        ... )
        >>> print(f"{result.duplicate_ratio:.1%} duplicates")
        5.0% duplicates
    """

    # Summary statistics
    total_conversations: int = Field(
        description="Total number of conversations in the dataset"
    )
    unique_conversations: int = Field(
        description="Number of unique conversations (after deduplication)"
    )
    duplicate_count: int = Field(
        description="Number of duplicate conversations that could be removed"
    )
    duplicate_ratio: float = Field(
        description="Ratio of duplicates to total (0.0 to 1.0)"
    )

    # Group information
    num_duplicate_groups: int = Field(
        default=0, description="Number of groups containing duplicates"
    )
    largest_group_size: int = Field(
        default=0, description="Size of the largest duplicate group"
    )
    duplicate_groups: list[DuplicateGroup] = Field(
        default_factory=list,
        description="Detailed information about each duplicate group",
    )

    # Indices for filtering
    duplicate_indices: list[int] = Field(
        default_factory=list,
        description="Flat list of all duplicate indices (recommended to remove)",
    )
    keep_indices: list[int] = Field(
        default_factory=list,
        description="Indices of conversations to keep (one per group)",
    )


class DeduplicationAnalyzer(DatasetAnalyzer[DeduplicationResult]):
    """Analyzer for detecting duplicate conversations in a dataset.

    Uses content hashing to find exact or near-exact duplicates. The analyzer
    groups duplicate conversations and provides indices for filtering.

    Example:
        >>> from oumi.analyze.analyzers.deduplication import DeduplicationAnalyzer
        >>> from oumi.core.types.conversation import Conversation, Message, Role
        >>>
        >>> analyzer = DeduplicationAnalyzer()
        >>> conversations = [...]  # List of conversations
        >>> result = analyzer.analyze(conversations)
        >>> print(f"Found {result.duplicate_count} duplicates")
        >>> # Filter duplicates
        >>> clean_data = [c for i, c in enumerate(conversations)
        ...               if i not in result.duplicate_indices]

    Args:
        hash_method: Method for generating content hash.
            - "exact": Hash exact content (case-sensitive, whitespace-sensitive)
            - "normalized": Normalize whitespace and lowercase before hashing
        include_system: Whether to include system messages in the hash.
        include_roles: Whether to include role prefixes in the hash.
        sample_text_length: Max length of sample text to include in results.
    """

    def __init__(
        self,
        hash_method: str = "normalized",
        include_system: bool = False,
        include_roles: bool = True,
        sample_text_length: int = 100,
    ):
        """Initialize the deduplication analyzer.

        Args:
            hash_method: Hash method ("exact" or "normalized").
            include_system: Include system messages in hash comparison.
            include_roles: Include role prefixes in hash (helps distinguish
                user vs assistant content).
            sample_text_length: Max length of sample text in results.
        """
        if hash_method not in ("exact", "normalized"):
            raise ValueError(
                f"hash_method must be 'exact' or 'normalized', got '{hash_method}'"
            )

        self.hash_method = hash_method
        self.include_system = include_system
        self.include_roles = include_roles
        self.sample_text_length = sample_text_length

    def _get_text_content(self, message) -> str:
        """Extract text content from a message."""
        content = message.content
        if isinstance(content, str):
            return content
        # Handle multimodal content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if hasattr(item, "content") and isinstance(item.content, str):
                    text_parts.append(item.content)
            return " ".join(text_parts)
        return str(content) if content else ""

    def _get_conversation_hash(self, conversation: Conversation) -> str:
        """Generate a hash for a conversation's content.

        Args:
            conversation: The conversation to hash.

        Returns:
            SHA-256 hash string of the conversation content.
        """
        parts = []

        for msg in conversation.messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

            # Skip system messages if configured
            if not self.include_system and role == "system":
                continue

            content = self._get_text_content(msg)

            # Normalize content if configured
            if self.hash_method == "normalized":
                # Lowercase and normalize whitespace
                content = " ".join(content.lower().split())

            # Include role prefix if configured
            if self.include_roles:
                parts.append(f"{role}:{content}")
            else:
                parts.append(content)

        combined = "\n".join(parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _get_sample_text(self, conversation: Conversation) -> str:
        """Get a sample of text from a conversation for display.

        Args:
            conversation: The conversation to sample.

        Returns:
            Truncated sample text.
        """
        if not conversation.messages:
            return ""

        # Get first non-system message content
        for msg in conversation.messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            if role != "system":
                content = self._get_text_content(msg)
                if len(content) > self.sample_text_length:
                    return content[: self.sample_text_length] + "..."
                return content

        return ""

    def analyze(self, conversations: list[Conversation]) -> DeduplicationResult:
        """Analyze dataset for duplicate conversations.

        Args:
            conversations: All conversations in the dataset.

        Returns:
            DeduplicationResult with duplicate statistics and indices.
        """
        if not conversations:
            return DeduplicationResult(
                total_conversations=0,
                unique_conversations=0,
                duplicate_count=0,
                duplicate_ratio=0.0,
                num_duplicate_groups=0,
                largest_group_size=0,
            )

        # Group conversations by hash
        hash_to_indices: dict[str, list[int]] = defaultdict(list)

        for idx, conv in enumerate(conversations):
            try:
                content_hash = self._get_conversation_hash(conv)
                hash_to_indices[content_hash].append(idx)
            except Exception as e:
                logger.warning(f"Failed to hash conversation at index {idx}: {e}")

        # Find duplicate groups (more than one conversation with same hash)
        duplicate_groups: list[DuplicateGroup] = []
        duplicate_indices: list[int] = []
        keep_indices: list[int] = []
        largest_group_size = 0

        for content_hash, indices in hash_to_indices.items():
            if len(indices) > 1:
                # This is a group of duplicates
                largest_group_size = max(largest_group_size, len(indices))

                # Get sample text from first conversation
                sample_text = self._get_sample_text(conversations[indices[0]])

                duplicate_groups.append(
                    DuplicateGroup(
                        indices=indices,
                        similarity=1.0,  # Exact hash match
                        sample_text=sample_text,
                    )
                )

                # Keep the first occurrence, mark rest as duplicates to remove
                keep_indices.append(indices[0])
                duplicate_indices.extend(indices[1:])
            else:
                # Unique conversation
                keep_indices.append(indices[0])

        # Sort for consistent output
        duplicate_indices.sort()
        keep_indices.sort()

        total = len(conversations)
        unique = total - len(duplicate_indices)

        return DeduplicationResult(
            total_conversations=total,
            unique_conversations=unique,
            duplicate_count=len(duplicate_indices),
            duplicate_ratio=len(duplicate_indices) / total if total > 0 else 0.0,
            num_duplicate_groups=len(duplicate_groups),
            largest_group_size=largest_group_size,
            duplicate_groups=duplicate_groups,
            duplicate_indices=duplicate_indices,
            keep_indices=keep_indices,
        )
