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

"""Sample analyzer plugin system for Oumi.

This package provides a plugin-based architecture for analyzing conversation data
with different types of sample analyzers (length, diversity, format, quality, etc.).
"""

# Import analyzers to register them
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.analyze.diversity_analyzer import DiversityAnalyzer
from oumi.core.analyze.format_analyzer import FormatAnalyzer
from oumi.core.analyze.health_score import (
    DatasetHealthScore,
    HealthScoreCalculator,
    HealthScoreComponent,
)
from oumi.core.analyze.length_analyzer import LengthAnalyzer
from oumi.core.analyze.presets import (
    get_preset,
    get_preset_with_language_detection,
    get_preset_with_tokenizer,
    list_presets,
)
from oumi.core.analyze.quality_analyzer import QualityAnalyzer
from oumi.core.analyze.recommendations import Recommendation, RecommendationsEngine
from oumi.core.analyze.report_generator import HTMLReportGenerator
from oumi.core.analyze.sample_analyzer import SampleAnalyzer

# Conditional import for EmbeddingAnalyzer (requires optional dependencies)
try:
    from oumi.core.analyze.embedding_analyzer import EmbeddingAnalyzer
except ImportError:
    EmbeddingAnalyzer = None  # type: ignore[misc, assignment]

# Conditional import for LLMJudgeAnalyzer (requires inference dependencies)
try:
    from oumi.core.analyze.llm_judge_analyzer import LLMJudgeAnalyzer
except ImportError:
    LLMJudgeAnalyzer = None  # type: ignore[misc, assignment]

__all__ = [
    # Core
    "DatasetAnalyzer",
    "SampleAnalyzer",
    # Analyzers
    "DiversityAnalyzer",
    "EmbeddingAnalyzer",
    "FormatAnalyzer",
    "LengthAnalyzer",
    "LLMJudgeAnalyzer",
    "QualityAnalyzer",
    # Recommendations
    "Recommendation",
    "RecommendationsEngine",
    # Health Score
    "DatasetHealthScore",
    "HealthScoreCalculator",
    "HealthScoreComponent",
    # Report
    "HTMLReportGenerator",
    # Presets
    "get_preset",
    "get_preset_with_language_detection",
    "get_preset_with_tokenizer",
    "list_presets",
]
