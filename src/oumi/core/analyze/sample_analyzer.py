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

"""Base classes for sample analyzer plugins."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from oumi.core.analyze.dataset_analyzer import (
    ConversationAnalysisResult,
    MessageAnalysisResult,
)
from oumi.core.types.conversation import Conversation


class SampleAnalyzer(ABC):
    """Base class for sample analyzer plugins that analyze individual samples."""

    @abstractmethod
    def analyze_sample(
        self, conversation: Conversation, tokenizer: Optional[Any] = None
    ) -> tuple[list[MessageAnalysisResult], ConversationAnalysisResult]:
        """Analyze a conversation sample and return comprehensive analysis results.

        This method analyzes a conversation and returns metrics for both individual
        messages and the conversation as a whole. Each analyzer can decide its own
        strategy for computing conversation-level metrics (e.g., aggregating message
        metrics or implementing custom conversation-level analysis).

        Args:
            conversation: The conversation object to analyze
            tokenizer: Optional tokenizer to use for tokenization-based analysis

        Returns:
            Tuple containing:
            - List of MessageAnalysisResult objects for each message
            - ConversationAnalysisResult for the conversation as a whole
        """
        pass
