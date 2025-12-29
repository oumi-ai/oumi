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

"""Enums for column configuration types."""

from enum import Enum


class ColumnType(str, Enum):
    """Enumeration of supported column data types for pandas DataFrames."""

    STRING = "string"
    """String/text data type."""

    INT = "int"
    """Integer data type."""

    FLOAT = "float"
    """Floating-point data type."""

    BOOL = "bool"
    """Boolean data type."""

    TIMESTAMP = "timestamp"
    """Timestamp/datetime data type."""

    CATEGORICAL = "categorical"
    """Categorical data type."""

    OBJECT = "object"
    """Generic object data type."""


class ContentType(str, Enum):
    """Enumeration of content types for analysis purposes."""

    TEXT = "text"
    """Text content that can be analyzed for length, sentiment, etc."""

    IMAGE = "image"
    """Image content that can be analyzed for visual features."""

    AUDIO = "audio"
    """Audio content that can be analyzed for audio features."""

    VIDEO = "video"
    """Video content that can be analyzed for video features."""

    NUMERIC = "numeric"
    """Numeric content that can be analyzed statistically."""

    BOOLEAN = "boolean"
    """Boolean flags that can be analyzed for true/false distributions."""

    CATEGORICAL = "categorical"
    """Single-valued categorical content that can be analyzed for distributions."""

    LIST = "list"
    """Multi-valued list/array content (e.g., comma-separated values)."""

    IDENTIFIER = "identifier"
    """Unique identifiers, keys, or IDs used for indexing and joins."""
