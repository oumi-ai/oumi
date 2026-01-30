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

from oumi.analyze.analyzers import LengthAnalyzer, TurnStatsAnalyzer
from oumi.analyze.base import (
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

# Import registry
from oumi.analyze.registry import (
    ANALYZER_REGISTRY,
    create_analyzer_from_config,
    get_analyzer_class,
    register_analyzer,
)

# Import CLI utilities
from oumi.analyze.cli import (
    generate_tests,
    list_metrics,
    load_conversations_from_path,
    print_summary,
    run_from_config_file,
    run_typed_analysis,
    save_results,
)

# Import custom metrics
from oumi.analyze.custom_metrics import (
    CustomConversationMetric,
    CustomMessageMetric,
    CustomMetricResult,
    create_custom_metric,
)

# Import result models
from oumi.analyze.results.length import LengthMetrics
from oumi.analyze.results.turn_stats import TurnStatsMetrics

# Import testing
from oumi.analyze.testing import TestEngine, TestResult, TestSummary

# Import utilities
from oumi.analyze.utils.dataframe import to_analysis_dataframe

__all__ = [
    # Base classes
    "MessageAnalyzer",
    "ConversationAnalyzer",
    "DatasetAnalyzer",
    "PreferenceAnalyzer",
    # Pipeline
    "AnalysisPipeline",
    # Analyzers
    "LengthAnalyzer",
    "TurnStatsAnalyzer",
    # Result models
    "LengthMetrics",
    "TurnStatsMetrics",
    # Utilities
    "to_analysis_dataframe",
    "load_conversations_from_path",
    # Config
    "TypedAnalyzeConfig",
    "AnalyzerConfig",
    # Discovery utilities
    "list_available_metrics",
    "print_analyzer_metrics",
    "get_analyzer_info",
    "describe_analyzer",
    # Registry
    "ANALYZER_REGISTRY",
    "register_analyzer",
    "get_analyzer_class",
    "create_analyzer_from_config",
    # CLI utilities
    "run_typed_analysis",
    "run_from_config_file",
    "save_results",
    "print_summary",
    "list_metrics",
    "generate_tests",
    # Testing
    "TestEngine",
    "TestResult",
    "TestSummary",
    # Custom metrics
    "CustomConversationMetric",
    "CustomMessageMetric",
    "CustomMetricResult",
    "create_custom_metric",
]
