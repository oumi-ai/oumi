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

"""Utilities for analyzer column naming and parsing.

This module provides standardized column naming conventions for analyzer
outputs. All analyzer-generated columns follow the format:
{source_column}__{analyzer_id}__{metric_name}

Note: The analyzer_id in column names refers to the analyzer instance
identifier. When using multiple instances of the same analyzer type (e.g.,
multiple llm_judge instances), each instance has a unique instance_id that
becomes the analyzer_id in column names.

The double underscore separator makes it easy to:
1. Split column names: col.split('__') → [source, analyzer, metric]
2. Use regex patterns: ^(.+?)__(.+?)__(.+?)$
3. Distinguish from internal underscores in each component
"""

from typing import NamedTuple


class AnalyzerColumnInfo(NamedTuple):
    """Parsed information from an analyzer column name."""

    source_column: str
    """The original source column that was analyzed."""

    analyzer_id: str
    """The instance ID of the analyzer that generated this column.

    This is the unique identifier for the analyzer instance, which may
    differ from the analyzer type (e.g., 'response_quality' vs
    'llm_judge').
    """

    metric_name: str
    """The specific metric name for this column."""


def make_analyzer_column_name(
    source_column: str, analyzer_id: str, metric_name: str
) -> str:
    """Create a standardized analyzer column name.

    Args:
        source_column: The source column being analyzed (e.g.,
            'text_content').
        analyzer_id: The analyzer instance identifier. This is the unique
            ID for this analyzer instance (e.g., 'response_quality',
            'llm_judge', 'ifd').
        metric_name: The metric name (e.g., 'has_pii', 'score').

    Returns:
        Standardized column name: {source_column}__{analyzer_id}__{metric_name}

    Examples:
        >>> make_analyzer_column_name('text_content', 'quality', 'has_pii')
        'text_content__quality__has_pii'

        >>> make_analyzer_column_name('text_content', 'response_quality', 'score')
        'text_content__response_quality__score'

        >>> make_analyzer_column_name(
        ...     'conversation_text_content', 'cost', 'fits_context_4k'
        ... )
        'conversation_text_content__cost__fits_context_4k'
    """
    return f"{source_column}__{analyzer_id}__{metric_name}"


def parse_analyzer_column_name(column_name: str) -> AnalyzerColumnInfo | None:
    """Parse an analyzer column name into its components.

    Args:
        column_name: Column name to parse.

    Returns:
        AnalyzerColumnInfo with source_column, analyzer_id (instance ID),
        and metric_name, or None if the column name doesn't match the
        analyzer format.

    Examples:
        >>> parse_analyzer_column_name('text_content__quality__has_pii')
        AnalyzerColumnInfo(source_column='text_content', \
analyzer_id='quality', metric_name='has_pii')

        >>> parse_analyzer_column_name(
        ...     'text_content__response_quality__score'
        ... )
        AnalyzerColumnInfo(source_column='text_content', \
analyzer_id='response_quality', metric_name='score')

        >>> parse_analyzer_column_name('regular_column')
        None
    """
    parts = column_name.split("__")
    if len(parts) == 3:
        return AnalyzerColumnInfo(
            source_column=parts[0], analyzer_id=parts[1], metric_name=parts[2]
        )
    return None


def is_analyzer_column(column_name: str) -> bool:
    """Check if a column name follows the analyzer naming convention.

    Args:
        column_name: Column name to check.

    Returns:
        True if the column name matches the analyzer format.

    Examples:
        >>> is_analyzer_column('text_content__quality__has_pii')
        True

        >>> is_analyzer_column('text_content')
        False
    """
    return parse_analyzer_column_name(column_name) is not None


def filter_analyzer_columns(
    columns: list[str], analyzer_id: str | None = None, source_column: str | None = None
) -> list[str]:
    """Filter columns to only include analyzer-generated columns.

    Args:
        columns: List of column names to filter.
        analyzer_id: Optional analyzer instance ID to filter by. This
            filters by the specific analyzer instance (e.g.,
            'response_quality' for a specific llm_judge instance), not
            the analyzer type.
        source_column: Optional source column to filter by.

    Returns:
        List of analyzer column names matching the filters.

    Examples:
        >>> cols = [
        ...     'text_content__quality__has_pii',
        ...     'text_content__ifd__score', 'regular_col'
        ... ]
        >>> filter_analyzer_columns(cols)
        ['text_content__quality__has_pii', 'text_content__ifd__score']

        >>> filter_analyzer_columns(cols, analyzer_id='quality')
        ['text_content__quality__has_pii']

        >>> # Filter by instance ID (multiple instances of same analyzer)
        >>> cols = [
        ...     'text_content__response_quality__score',
        ...     'text_content__instruction_quality__score'
        ... ]
        >>> filter_analyzer_columns(cols, analyzer_id='response_quality')
        ['text_content__response_quality__score']

        >>> filter_analyzer_columns(cols, source_column='text_content')
        ['text_content__quality__has_pii', 'text_content__ifd__score']
    """
    result = []
    for col in columns:
        info = parse_analyzer_column_name(col)
        if info is None:
            continue

        # Apply filters
        if analyzer_id is not None and info.analyzer_id != analyzer_id:
            continue
        if source_column is not None and info.source_column != source_column:
            continue

        result.append(col)

    return result


def get_analyzer_columns_by_source(columns: list[str]) -> dict[str, list[str]]:
    """Group analyzer columns by their source column.

    Args:
        columns: List of column names.

    Returns:
        Dictionary mapping source column name to list of analyzer columns.

    Examples:
        >>> cols = ['text_content__quality__has_pii', 'text_content__ifd__score',
        ...         'other_col__length__count', 'regular_col']
        >>> get_analyzer_columns_by_source(cols)
        {'text_content': ['text_content__quality__has_pii', 'text_content__ifd__score'],
         'other_col': ['other_col__length__count']}
    """
    result: dict[str, list[str]] = {}
    for col in columns:
        info = parse_analyzer_column_name(col)
        if info is not None:
            if info.source_column not in result:
                result[info.source_column] = []
            result[info.source_column].append(col)
    return result


def get_analyzer_columns_by_analyzer(columns: list[str]) -> dict[str, list[str]]:
    """Group analyzer columns by their analyzer instance ID.

    Args:
        columns: List of column names.

    Returns:
        Dictionary mapping analyzer instance ID to list of analyzer columns.
        When using multiple instances of the same analyzer type, each instance
        will have a separate entry in the dictionary.

    Examples:
        >>> cols = ['text_content__quality__has_pii', 'text_content__ifd__score',
        ...         'other_col__quality__score', 'regular_col']
        >>> get_analyzer_columns_by_analyzer(cols)
        {'quality': ['text_content__quality__has_pii', 'other_col__quality__score'],
         'ifd': ['text_content__ifd__score']}

        >>> # With multiple instances of same analyzer type
        >>> cols = [
        ...     'text_content__response_quality__score',
        ...     'text_content__instruction_quality__score'
        ... ]
        >>> get_analyzer_columns_by_analyzer(cols)
        {'response_quality': ['text_content__response_quality__score'],
         'instruction_quality': \
['text_content__instruction_quality__score']}
    """
    result: dict[str, list[str]] = {}
    for col in columns:
        info = parse_analyzer_column_name(col)
        if info is not None:
            if info.analyzer_id not in result:
                result[info.analyzer_id] = []
            result[info.analyzer_id].append(col)
    return result
