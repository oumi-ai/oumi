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

"""Length analyzer for text content."""

import re
from typing import Any

from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry.registry import register_sample_analyzer


@register_sample_analyzer("length")
class LengthAnalyzer(SampleAnalyzer):
    """Analyzer that computes various length metrics for text content."""

    def __init__(
        self,
        *,
        char_count: bool = True,
        word_count: bool = True,
        sentence_count: bool = True,
        token_count: bool = False,
    ):
        """Initialize the length analyzer.

        Args:
            char_count: Whether to compute character count
            word_count: Whether to compute word count
            sentence_count: Whether to compute sentence count
            token_count: Whether to compute token count (placeholder)
        """
        self.char_count = char_count
        self.word_count = word_count
        self.sentence_count = sentence_count
        self.token_count = token_count

    def analyze_message(self, text_content: str) -> dict[str, Any]:
        """Analyze text content and return length metrics.

        Args:
            text_content: The text content to analyze

        Returns:
            Dictionary containing requested length metrics
        """
        metrics = {}

        if self.char_count:
            metrics["char_count"] = len(text_content)

        if self.word_count:
            # Simple word count - split on whitespace
            metrics["word_count"] = len(text_content.split())

        if self.sentence_count:
            # Simple sentence count - split on common sentence endings
            sentences = re.split(r"[.!?]+", text_content)
            # Filter out empty strings
            sentences = [s.strip() for s in sentences if s.strip()]
            metrics["sentence_count"] = len(sentences)

        if self.token_count:
            # Placeholder for token count - would need a tokenizer
            metrics["token_count"] = len(text_content.split())  # Fallback to word count

        return metrics
