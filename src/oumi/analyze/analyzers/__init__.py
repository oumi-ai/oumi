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

"""Analyzer implementations and result models.

This module contains concrete analyzer implementations that inherit from
the base analyzer classes and return typed result models. Each analyzer
file contains both the analyzer class and its result model for better cohesion.
"""

from oumi.analyze.analyzers.deduplication import (
    DeduplicationAnalyzer,
    DeduplicationResult,
    DuplicateGroup,
)
from oumi.analyze.analyzers.length import LengthAnalyzer, LengthMetrics
from oumi.analyze.analyzers.llm_analyzer import (
    CoherenceAnalyzer,
    FactualityAnalyzer,
    InstructionFollowingAnalyzer,
    JudgmentType,
    LLMAnalyzer,
    LLMJudgmentMetrics,
    SafetyAnalyzer,
    TargetScope,
    UsefulnessAnalyzer,
    get_available_criteria,
    get_criteria_info,
)
from oumi.analyze.analyzers.quality import DataQualityAnalyzer, DataQualityMetrics
from oumi.analyze.analyzers.turn_stats import TurnStatsAnalyzer, TurnStatsMetrics

__all__ = [
    # Non-LLM analyzers
    "LengthAnalyzer",
    "LengthMetrics",
    "TurnStatsAnalyzer",
    "TurnStatsMetrics",
    "DataQualityAnalyzer",
    "DataQualityMetrics",
    # Dataset-level analyzers
    "DeduplicationAnalyzer",
    "DeduplicationResult",
    "DuplicateGroup",
    # LLM-based analyzers
    "LLMAnalyzer",
    "LLMJudgmentMetrics",
    "UsefulnessAnalyzer",
    "SafetyAnalyzer",
    "FactualityAnalyzer",
    "CoherenceAnalyzer",
    "InstructionFollowingAnalyzer",
    # Enums
    "TargetScope",
    "JudgmentType",
    # Utilities
    "get_available_criteria",
    "get_criteria_info",
]
