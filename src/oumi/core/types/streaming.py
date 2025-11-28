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

"""Types for streaming inference responses."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StreamingChunkType(str, Enum):
    """Type of streaming chunk."""

    CONTENT_DELTA = "content_delta"
    """Incremental text content."""

    FINISH = "finish"
    """Stream completion marker."""

    ERROR = "error"
    """Error during streaming."""


@dataclass
class StreamingChunk:
    r"""Represents a single chunk in a streaming response.

    This is the atomic unit yielded during streaming inference. Each chunk
    contains either incremental content, a completion signal, or an error.

    Example:
        >>> async for chunk in engine.infer_stream(conversation):
        ...     if chunk.chunk_type == StreamingChunkType.CONTENT_DELTA:
        ...         print(chunk.delta, end="", flush=True)
        ...     elif chunk.is_final:
        ...         print(f"\nDone: {chunk.finish_reason}")
    """

    chunk_type: StreamingChunkType
    """The type of this chunk."""

    delta: str = ""
    """The incremental text content (for CONTENT_DELTA type)."""

    finish_reason: Optional[str] = None
    """The reason streaming finished (for FINISH type), e.g., 'stop', 'length'."""

    error_message: Optional[str] = None
    """Error message if chunk_type is ERROR."""

    accumulated_content: str = ""
    """Total content accumulated so far (for convenience)."""

    conversation_id: Optional[str] = None
    """ID of the conversation this chunk belongs to."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (e.g., token counts, latency info)."""

    @property
    def is_final(self) -> bool:
        """Returns True if this is the final chunk in the stream."""
        return self.chunk_type in (StreamingChunkType.FINISH, StreamingChunkType.ERROR)
