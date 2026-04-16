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

"""Typed analyzer framework for dataset analysis."""

from oumi.analyze.analyzers.length import LengthAnalyzer, LengthMetrics
from oumi.analyze.analyzers.quality import DataQualityAnalyzer, DataQualityMetrics
from oumi.analyze.analyzers.turn_stats import TurnStatsAnalyzer, TurnStatsMetrics
from oumi.analyze.base import (
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
    generate_test_template,
    get_analyzer_info,
    list_available_metrics,
    print_analyzer_metrics,
)
from oumi.analyze.pipeline import AnalysisPipeline
from oumi.analyze.testing import TestEngine, TestResult, TestSummary
from oumi.analyze.utils.dataframe import to_analysis_dataframe
from oumi.cli.analyze import (
    create_analyzer_from_config,
    generate_tests,
    get_analyzer_class,
    list_metrics,
    print_summary,
    run_from_config_file,
    run_typed_analysis,
    save_results,
)

__all__ = [
    "AnalysisPipeline",
    "AnalyzerConfig",
    "ConversationAnalyzer",
    "DataQualityAnalyzer",
    "DataQualityMetrics",
    "DatasetAnalyzer",
    "LengthAnalyzer",
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
    "generate_test_template",
    "generate_tests",
    "get_analyzer_class",
    "get_analyzer_info",
    "list_available_metrics",
    "list_metrics",
    "print_analyzer_metrics",
    "print_summary",
    "run_from_config_file",
    "run_typed_analysis",
    "save_results",
    "to_analysis_dataframe",
]
