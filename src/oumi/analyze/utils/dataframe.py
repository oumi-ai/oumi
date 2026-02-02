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

"""DataFrame conversion utilities for typed analysis results."""

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel

from oumi.core.types.conversation import Conversation

logger = logging.getLogger(__name__)


def to_analysis_dataframe(
    conversations: list[Conversation],
    results: dict[str, list[BaseModel] | BaseModel],
    message_to_conversation_idx: list[int] | None = None,
) -> pd.DataFrame:
    """Convert typed analysis results to a pandas DataFrame.

    Creates a DataFrame with one row per conversation, with columns for
    conversation metadata and all analyzer metrics. Analyzer field names
    are prefixed with the analyzer name to avoid collisions.

    Example:
        >>> results = {"LengthAnalyzer": [LengthMetrics(...), LengthMetrics(...)]}
        >>> df = to_analysis_dataframe(conversations, results)
        >>> print(df.columns.tolist())
        ['conversation_id', 'conversation_index', 'num_messages',
         'length__total_chars', 'length__total_words', ...]

    Args:
        conversations: List of conversations that were analyzed.
        results: Dictionary mapping analyzer names to results.
            - For per-conversation results: list of BaseModel (len = num conversations)
            - For message-level results: list of BaseModel (len = num messages)
            - For dataset-level results: single BaseModel (will be repeated)
        message_to_conversation_idx: Optional mapping from message index to
            conversation index. Required for proper aggregation of message-level
            results. If provided, message-level results will be aggregated per
            conversation.

    Returns:
        DataFrame with conversation metadata and all metrics as columns.
    """
    rows: list[dict[str, Any]] = []

    # Determine expected counts
    num_conversations = len(conversations)
    total_messages = sum(len(conv.messages) for conv in conversations)

    for i, conv in enumerate(conversations):
        row: dict[str, Any] = {
            "conversation_index": i,
            "conversation_id": conv.conversation_id or f"conv_{i}",
            "num_messages": len(conv.messages),
        }

        # Add results from each analyzer
        for analyzer_name, analyzer_results in results.items():
            # Get the prefix for column names
            prefix = _get_column_prefix(analyzer_name)

            if isinstance(analyzer_results, list):
                result_count = len(analyzer_results)

                if result_count == num_conversations:
                    # Per-conversation results - direct mapping
                    if i < result_count:
                        result = analyzer_results[i]
                        _add_result_to_row(row, result, prefix)

                elif result_count == total_messages and message_to_conversation_idx:
                    # Message-level results - aggregate for this conversation
                    conv_messages = [
                        analyzer_results[msg_idx]
                        for msg_idx, conv_idx in enumerate(message_to_conversation_idx)
                        if conv_idx == i
                    ]
                    if conv_messages:
                        # Use first message result (or could aggregate)
                        # TODO: Consider aggregation strategy (first, mean, etc.)
                        _add_result_to_row(row, conv_messages[0], prefix)
                        row[f"{prefix}__message_count"] = len(conv_messages)

                elif i < result_count:
                    # Fallback: try to use result at index i
                    result = analyzer_results[i]
                    _add_result_to_row(row, result, prefix)
                    # Warn on first conversation only to avoid spam
                    if i == 0:
                        logger.warning(
                            f"Analyzer '{analyzer_name}' returned {result_count} results "
                            f"for {num_conversations} conversations (expected equal counts "
                            f"or {total_messages} for message-level). Some conversations "
                            "may have missing metric values."
                        )
                else:
                    # Results list is shorter than conversation index - warn once
                    if i == result_count:  # Only warn when we first exceed
                        logger.warning(
                            f"Analyzer '{analyzer_name}' returned {result_count} results "
                            f"for {num_conversations} conversations. Conversations "
                            f"{result_count}-{num_conversations - 1} will have missing values."
                        )

            elif isinstance(analyzer_results, BaseModel):
                # Dataset-level result - same for all conversations
                _add_result_to_row(row, analyzer_results, prefix)

        rows.append(row)

    return pd.DataFrame(rows)


def to_message_dataframe(
    conversations: list[Conversation],
    results: dict[str, list[BaseModel]],
) -> pd.DataFrame:
    """Convert message-level analysis results to a pandas DataFrame.

    Creates a DataFrame with one row per message, including conversation
    context and message-level metrics.

    Args:
        conversations: List of conversations that were analyzed.
        results: Dictionary mapping analyzer names to message-level results.
            Must have one result per message (flattened across conversations).

    Returns:
        DataFrame with one row per message.
    """
    rows: list[dict[str, Any]] = []
    message_idx = 0

    for conv_idx, conv in enumerate(conversations):
        for msg_idx, message in enumerate(conv.messages):
            row: dict[str, Any] = {
                "conversation_index": conv_idx,
                "conversation_id": conv.conversation_id or f"conv_{conv_idx}",
                "message_index": msg_idx,
                "message_id": message.id or f"msg_{conv_idx}_{msg_idx}",
                "role": message.role.value,
            }

            # Add text content (handle multimodal)
            if isinstance(message.content, str):
                row["text_content"] = message.content
            else:
                # Concatenate text from content items
                text_parts = []
                for item in message.content:
                    if hasattr(item, "content") and isinstance(item.content, str):
                        text_parts.append(item.content)
                row["text_content"] = " ".join(text_parts)

            # Add results from each message-level analyzer
            for analyzer_name, analyzer_results in results.items():
                prefix = _get_column_prefix(analyzer_name)
                if message_idx < len(analyzer_results):
                    result = analyzer_results[message_idx]
                    _add_result_to_row(row, result, prefix)

            rows.append(row)
            message_idx += 1

    return pd.DataFrame(rows)


def _get_column_prefix(analyzer_name: str) -> str:
    """Get the column name prefix for an analyzer.

    Converts analyzer names to lowercase prefixes:
    - "LengthAnalyzer" -> "length"
    - "QualityAnalyzer" -> "quality"
    - "CustomName" -> "customname"

    Args:
        analyzer_name: Name of the analyzer.

    Returns:
        Lowercase prefix for column names.
    """
    # Remove common suffixes
    name = analyzer_name
    for suffix in ["Analyzer", "Metrics"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    return name.lower()


def _add_result_to_row(
    row: dict[str, Any],
    result: BaseModel | dict[str, Any],
    prefix: str,
) -> None:
    """Add fields from a result model to a row dictionary.

    Handles nested structures and lists appropriately:
    - Scalar values: prefix__field_name
    - Lists: prefix__field_name (stored as-is for DataFrame)
    - Nested models: prefix__nested_field_name (flattened)

    Args:
        row: Row dictionary to add fields to.
        result: Pydantic model or dict with fields to add.
        prefix: Prefix for column names.
    """
    # Handle both Pydantic models and raw dicts (from cache)
    if isinstance(result, dict):
        result_dict = result
    else:
        result_dict = result.model_dump()

    for field_name, value in result_dict.items():
        column_name = f"{prefix}__{field_name}"

        if isinstance(value, dict):
            # Nested structure - flatten with additional prefix
            for nested_key, nested_value in value.items():
                row[f"{column_name}__{nested_key}"] = nested_value
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            # List of dicts - skip for now (complex structure)
            pass
        else:
            # Scalar or simple list - add directly
            row[column_name] = value


def results_to_dict(
    results: dict[str, list[BaseModel] | BaseModel],
) -> dict[str, list[dict[str, Any]] | dict[str, Any]]:
    """Convert typed results to a serializable dictionary.

    Useful for saving results to JSON or other formats.

    Args:
        results: Dictionary of analyzer results.

    Returns:
        Dictionary with Pydantic models converted to dicts.
    """
    output: dict[str, list[dict[str, Any]] | dict[str, Any]] = {}

    for name, result in results.items():
        if isinstance(result, list):
            output[name] = [r.model_dump() for r in result]
        elif isinstance(result, BaseModel):
            output[name] = result.model_dump()

    return output
