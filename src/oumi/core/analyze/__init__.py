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
with different types of sample analyzers (length, quality, etc.).
"""

# Import analyzers to register them
from oumi.core.analyze.category_analyzer import CategoryDistributionAnalyzer
from oumi.core.analyze.conversation_structure_analyzer import (
    ConversationStructureAnalyzer,
)
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.analyze.duplicate_analyzer import DuplicateAnalyzer
from oumi.core.analyze.empty_content_analyzer import EmptyContentAnalyzer
from oumi.core.analyze.encoding_analyzer import EncodingAnalyzer
from oumi.core.analyze.format_validator import FormatValidationAnalyzer
from oumi.core.analyze.length_analyzer import LengthAnalyzer
from oumi.core.analyze.ngram_analyzer import NgramAnalyzer
from oumi.core.analyze.qa_pair_analyzer import QuestionAnswerPairAnalyzer
from oumi.core.analyze.question_duplicate_analyzer import QuestionDuplicateAnalyzer
from oumi.core.analyze.readability_analyzer import ReadabilityAnalyzer
from oumi.core.analyze.repetition_analyzer import RepetitionAnalyzer
from oumi.core.analyze.request_type_analyzer import RequestTypeAnalyzer
from oumi.core.analyze.response_duplicate_analyzer import ResponseDuplicateAnalyzer
from oumi.core.analyze.role_sequence_analyzer import RoleSequenceAnalyzer
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.analyze.statistical_analyzer import StatisticalOutlierAnalyzer
from oumi.core.analyze.system_prompt_analyzer import SystemPromptAnalyzer
from oumi.core.analyze.vocabulary_analyzer import VocabularyAnalyzer

__all__ = [
    "CategoryDistributionAnalyzer",
    "ConversationStructureAnalyzer",
    "DatasetAnalyzer",
    "DuplicateAnalyzer",
    "EmptyContentAnalyzer",
    "EncodingAnalyzer",
    "FormatValidationAnalyzer",
    "LengthAnalyzer",
    "NgramAnalyzer",
    "QuestionAnswerPairAnalyzer",
    "QuestionDuplicateAnalyzer",
    "ReadabilityAnalyzer",
    "RepetitionAnalyzer",
    "RequestTypeAnalyzer",
    "ResponseDuplicateAnalyzer",
    "RoleSequenceAnalyzer",
    "SampleAnalyzer",
    "StatisticalOutlierAnalyzer",
    "SystemPromptAnalyzer",
    "VocabularyAnalyzer",
]
