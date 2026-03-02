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

"""Analyzer framework for dataset analysis."""

from oumi.analyze.analyzers import (
    DataQualityAnalyzer,
    DataQualityMetrics,
    LengthAnalyzer,
    LengthAnalyzerConfig,
    LengthMetrics,
    TurnStatsAnalyzer,
    TurnStatsMetrics,
)
from oumi.analyze.base import (
    BaseAnalyzer,
    ConversationAnalyzer,
    DatasetAnalyzer,
    MessageAnalyzer,
    PreferenceAnalyzer,
)
from oumi.analyze.config import (
    AnalyzerConfig,
    TypedAnalyzeConfig,
)
from oumi.analyze.discovery import (
    describe_analyzer,
    get_analyzer_info,
    get_instance_metrics,
    list_available_metrics,
    print_analyzer_metrics,
)
from oumi.analyze.pipeline import AnalysisPipeline
from oumi.analyze.testing import TestEngine, TestResult, TestSummary
from oumi.analyze.utils.dataframe import to_analysis_dataframe
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

    Prefers using the analyzer's from_config() classmethod if available,
    otherwise falls back to direct instantiation with **params.

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
        # Prefer from_config() if available for better config handling
        if hasattr(analyzer_class, "from_config") and callable(
            getattr(analyzer_class, "from_config")
        ):
            return analyzer_class.from_config(params)  # type: ignore[union-attr]
        else:
            return analyzer_class(**params)
    except Exception as e:
        logger.error(f"Failed to create analyzer {analyzer_id}: {e}")
        return None


__all__ = [
    "AnalysisPipeline",
    "AnalyzerConfig",
    "BaseAnalyzer",
    "ConversationAnalyzer",
    "DataQualityAnalyzer",
    "DataQualityMetrics",
    "DatasetAnalyzer",
    "LengthAnalyzer",
    "LengthAnalyzerConfig",
    "LengthMetrics",
    "MessageAnalyzer",
    "PreferenceAnalyzer",
    "TestEngine",
    "TestResult",
    "TestSummary",
    "TurnStatsAnalyzer",
    "TurnStatsMetrics",
    "TypedAnalyzeConfig",
    "create_analyzer_from_config",
    "describe_analyzer",
    "get_analyzer_class",
    "get_analyzer_info",
    "get_instance_metrics",
    "list_available_metrics",
    "print_analyzer_metrics",
    "register_analyzer",
    "to_analysis_dataframe",
]
