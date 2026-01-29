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

"""Deduplication result model."""

from pydantic import BaseModel, Field


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
        default=None,
        description="Sample text from the first conversation in the group"
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
        default=0,
        description="Number of groups containing duplicates"
    )
    largest_group_size: int = Field(
        default=0,
        description="Size of the largest duplicate group"
    )
    duplicate_groups: list[DuplicateGroup] = Field(
        default_factory=list,
        description="Detailed information about each duplicate group"
    )

    # Indices for filtering
    duplicate_indices: list[int] = Field(
        default_factory=list,
        description="Flat list of all duplicate indices (recommended to remove)"
    )
    keep_indices: list[int] = Field(
        default_factory=list,
        description="Indices of conversations to keep (one per group)"
    )
