"""Length analyzer plugin for computing text length metrics."""

import re
from typing import Any

from oumi.core.analyze.sample_analyzer import SampleAnalyzer


class LengthAnalyzer(SampleAnalyzer):
    """Analyzer for length-related metrics."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the length analyzer.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.char_count = config.get("char_count", True)
        self.word_count = config.get("word_count", True)
        self.sentence_count = config.get("sentence_count", False)
        self.token_count = config.get("token_count", False)

    def analyze_message(
        self, text_content: str, message_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze length metrics for a single message."""
        metrics = {}

        if self.char_count:
            metrics["char_count"] = len(text_content)

        if self.word_count:
            # Simple word counting - split on whitespace
            words = text_content.split()
            metrics["word_count"] = len(words)

        if self.sentence_count:
            # Simple sentence counting - split on common sentence endings
            sentences = re.split(r"[.!?]+", text_content)
            # Filter out empty strings
            sentences = [s.strip() for s in sentences if s.strip()]
            metrics["sentence_count"] = len(sentences)

        if self.token_count:
            # Simple token estimation - roughly 4 characters per token
            metrics["token_count"] = len(text_content) // 4

        return metrics
