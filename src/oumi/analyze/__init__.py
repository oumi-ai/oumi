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

"""Analyzer framework for dataset analysis.

This module provides a typed, Pydantic-based approach to analyzing datasets,
with typed Conversation objects and strongly-typed result models.

Example usage:

    from oumi.analyze import AnalysisPipeline
    from oumi.analyze.base import ConversationAnalyzer

    # Create custom analyzer
    class MyAnalyzer(ConversationAnalyzer):
        ...

    # Batch processing with pipeline
    pipeline = AnalysisPipeline(analyzers=[MyAnalyzer()])
    results = pipeline.run(conversations)
"""

from oumi.analyze.base import (
    BaseAnalyzer,
    ConversationAnalyzer,
    DatasetAnalyzer,
    MessageAnalyzer,
    PreferenceAnalyzer,
)

# Import config
from oumi.analyze.config import (
    AnalyzerConfig,
    TypedAnalyzeConfig,
)

# Import discovery utilities
from oumi.analyze.discovery import (
    describe_analyzer,
    get_analyzer_info,
    list_available_metrics,
    print_analyzer_metrics,
)
from oumi.analyze.pipeline import AnalysisPipeline

# Import utilities
from oumi.analyze.utils.dataframe import to_analysis_dataframe

# Import registry from core (reuse existing infrastructure)
from oumi.core.registry import (
    REGISTRY,
)
from oumi.core.registry import (
    register_sample_analyzer as register_analyzer,
)


def get_analyzer_class(name: str) -> type | None:
    """Get an analyzer class by name.

    Args:
        name: Name of the analyzer.

    Returns:
        The analyzer class or None if not found.
    """
    from typing import cast

    result = REGISTRY.get_sample_analyzer(name)
    return cast(type | None, result)


def create_analyzer_from_config(
    analyzer_id: str,
    params: dict,
) -> "MessageAnalyzer | ConversationAnalyzer | DatasetAnalyzer | None":
    """Create an analyzer instance from configuration.

    Args:
        analyzer_id: Analyzer type identifier.
        params: Analyzer-specific parameters.

    Returns:
        Analyzer instance or None if not found.
    """
    import logging

    logger = logging.getLogger(__name__)

    analyzer_class = REGISTRY.get_sample_analyzer(analyzer_id)
    if analyzer_class is None:
        logger.warning(f"Unknown analyzer: {analyzer_id}")
        return None

    try:
        return analyzer_class(**params)
    except Exception as e:
        logger.error(f"Failed to create analyzer {analyzer_id}: {e}")
        return None


__all__ = [
    # Base classes
    "BaseAnalyzer",
    "MessageAnalyzer",
    "ConversationAnalyzer",
    "DatasetAnalyzer",
    "PreferenceAnalyzer",
    # Pipeline
    "AnalysisPipeline",
    # Utilities
    "to_analysis_dataframe",
    # Config
    "TypedAnalyzeConfig",
    "AnalyzerConfig",
    # Discovery utilities
    "list_available_metrics",
    "print_analyzer_metrics",
    "get_analyzer_info",
    "describe_analyzer",
    "register_analyzer",
    "get_analyzer_class",
    "create_analyzer_from_config",
]
