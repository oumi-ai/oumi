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
with different types of sample analyzers (length, safety, etc.).
"""

# Import analyzers to register them
from oumi.core.analyze.analysis_results import (
    AnalysisResultsManager,
    DatasetAnalysisResult,
)
from oumi.core.analyze.config_reader import AnalysisComponents, ConfigReader
from oumi.core.analyze.conversation_handler import ConversationHandler
from oumi.core.analyze.dataframe_analyzer import (
    AnalysisResult,
    DataFrameAnalyzer,
    DataFrameWithSchema,
)
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.analyze.length_analyzer import LengthAnalyzer
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.analyze.summary_generator import SummaryGenerator

__all__ = [
    "AnalysisComponents",
    "AnalysisResult",
    "AnalysisResultsManager",
    "ConfigReader",
    "ConversationHandler",
    "DataFrameAnalyzer",
    "DataFrameWithSchema",
    "DatasetAnalysisResult",
    "DatasetAnalyzer",
    "LengthAnalyzer",
    "SampleAnalyzer",
    "SummaryGenerator",
]
