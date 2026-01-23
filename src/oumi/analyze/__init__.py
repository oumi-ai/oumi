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

"""Typed analyzer framework for dataset analysis.

This module provides a typed, Pydantic-based approach to analyzing datasets,
replacing the DataFrame-centric approach with typed Conversation objects
and strongly-typed result models.

Example usage:

    from oumi.analyze import LengthAnalyzer, AnalysisPipeline

    # Single conversation analysis
    analyzer = LengthAnalyzer()
    result = analyzer.analyze(conversation)
    print(f"Total words: {result.total_words}")

    # Batch processing with pipeline
    pipeline = AnalysisPipeline(analyzers=[LengthAnalyzer()])
    results = pipeline.run(conversations)

    # Convert to DataFrame when needed
    df = pipeline.to_dataframe()
"""

# Import analyzers
from oumi.analyze.analyzers.length import LengthAnalyzer
from oumi.analyze.base import (
    ConversationAnalyzer,
    DatasetAnalyzer,
    MessageAnalyzer,
    PreferenceAnalyzer,
)

# Import CLI utilities
from oumi.analyze.cli import (
    generate_tests,
    list_metrics,
    print_summary,
    run_from_config_file,
    run_typed_analysis,
    save_results,
)

# Import config
from oumi.analyze.config import (
    AnalyzerConfig,
    CustomMetricConfig,
    TypedAnalyzeConfig,
)

# Import custom metrics
from oumi.analyze.custom_metrics import (
    CustomConversationMetric,
    CustomMessageMetric,
    CustomMetricResult,
    create_custom_metric,
)

# Import discovery utilities
from oumi.analyze.discovery import (
    describe_analyzer,
    generate_test_template,
    get_analyzer_info,
    list_available_metrics,
    print_analyzer_metrics,
)
from oumi.analyze.pipeline import AnalysisPipeline

# Import result models
from oumi.analyze.results.length import LengthMetrics

# Import testing
from oumi.analyze.testing import TestEngine, TestResult, TestSummary
from oumi.analyze.utils.dataframe import to_analysis_dataframe

__all__ = [
    # Base classes
    "MessageAnalyzer",
    "ConversationAnalyzer",
    "DatasetAnalyzer",
    "PreferenceAnalyzer",
    # Pipeline
    "AnalysisPipeline",
    # Utilities
    "to_analysis_dataframe",
    # Analyzers
    "LengthAnalyzer",
    # Result models
    "LengthMetrics",
    # Config
    "TypedAnalyzeConfig",
    "AnalyzerConfig",
    "CustomMetricConfig",
    # Testing
    "TestEngine",
    "TestResult",
    "TestSummary",
    # CLI utilities
    "run_typed_analysis",
    "run_from_config_file",
    "save_results",
    "print_summary",
    "list_metrics",
    "generate_tests",
    # Custom metrics
    "CustomConversationMetric",
    "CustomMessageMetric",
    "CustomMetricResult",
    "create_custom_metric",
    # Discovery utilities
    "list_available_metrics",
    "print_analyzer_metrics",
    "get_analyzer_info",
    "generate_test_template",
    "describe_analyzer",
]
