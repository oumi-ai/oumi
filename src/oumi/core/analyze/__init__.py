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
from oumi.core.analyze.content_pattern_analyzer import ContentPatternAnalyzer
from oumi.core.analyze.cost_analyzer import CostAnalyzer
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
from oumi.core.analyze.test_engine import TestEngine
from oumi.core.analyze.test_result import TestResult, TestSummary
from oumi.core.analyze.report_generator import HTMLReportGenerator
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.analyze.training_quality_analyzer import TrainingQualityAnalyzer

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

# Conditional import for FastTextAnalyzer (requires fast-langdetect or fasttext)
try:
    from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer
except ImportError:
    FastTextAnalyzer = None  # type: ignore[misc, assignment]

# Conditional import for QuestionDiversityAnalyzer (requires sentence-transformers)
try:
    from oumi.core.analyze.question_diversity_analyzer import (
        QuestionDiversityAnalyzer,
    )
except ImportError:
    QuestionDiversityAnalyzer = None  # type: ignore[misc, assignment]

# Conditional import for IFDAnalyzer (requires transformers)
try:
    from oumi.core.analyze.ifd_analyzer import IFDAnalyzer
except ImportError:
    IFDAnalyzer = None  # type: ignore[misc, assignment]

# Conditional import for ReprDiversityAnalyzer (requires sentence-transformers)
try:
    from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer
except ImportError:
    ReprDiversityAnalyzer = None  # type: ignore[misc, assignment]

# Conditional import for EvolComplexityAnalyzer (requires inference dependencies)
try:
    from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer
except ImportError:
    EvolComplexityAnalyzer = None  # type: ignore[misc, assignment]

# Conditional import for EvolQualityAnalyzer (requires inference dependencies)
try:
    from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer
except ImportError:
    EvolQualityAnalyzer = None  # type: ignore[misc, assignment]

# New analyzers based on "Fixing It in Post" paper
from oumi.core.analyze.conversation_structure_analyzer import (
    ConversationStructureAnalyzer,
)
from oumi.core.analyze.difficulty_analyzer import DifficultyAnalyzer
from oumi.core.analyze.input_quality_analyzer import InputQualityAnalyzer
from oumi.core.analyze.instruct_reward_analyzer import InstructRewardAnalyzer
from oumi.core.analyze.response_completeness_analyzer import (
    ResponseCompletenessAnalyzer,
)
from oumi.core.analyze.safety_analyzer import SafetyAnalyzer
from oumi.core.analyze.task_category_analyzer import TaskCategoryAnalyzer
from oumi.core.analyze.token_stats_analyzer import TokenStatsAnalyzer

__all__ = [
    # Core
    "DatasetAnalyzer",
    "SampleAnalyzer",
    # Analyzers
    "ContentPatternAnalyzer",
    "CostAnalyzer",
    "DiversityAnalyzer",
    "EmbeddingAnalyzer",
    "EvolComplexityAnalyzer",
    "EvolQualityAnalyzer",
    "FastTextAnalyzer",
    "FormatAnalyzer",
    "IFDAnalyzer",
    "LengthAnalyzer",
    "LLMJudgeAnalyzer",
    "QualityAnalyzer",
    "QuestionDiversityAnalyzer",
    "ReprDiversityAnalyzer",
    "TrainingQualityAnalyzer",
    # New analyzers (from "Fixing It in Post" paper)
    "TaskCategoryAnalyzer",
    "InstructRewardAnalyzer",
    "InputQualityAnalyzer",
    "ConversationStructureAnalyzer",
    "SafetyAnalyzer",
    "DifficultyAnalyzer",
    "ResponseCompletenessAnalyzer",
    "TokenStatsAnalyzer",
    # Test Engine and Results
    "TestEngine",
    "TestResult",
    "TestSummary",
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
