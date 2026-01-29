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

"""Pydantic result models for analyzer outputs.

This module contains strongly-typed result models that analyzers return.
Each model defines the exact schema of metrics an analyzer produces.
"""

from oumi.analyze.results.deduplication import DeduplicationResult, DuplicateGroup
from oumi.analyze.results.length import LengthMetrics
from oumi.analyze.results.llm_judgment import LLMJudgmentMetrics
from oumi.analyze.results.turn_stats import TurnStatsMetrics

__all__ = [
    "LengthMetrics",
    "LLMJudgmentMetrics",
    "TurnStatsMetrics",
    "DeduplicationResult",
    "DuplicateGroup",
]
