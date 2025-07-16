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
from typing import Any, Union


class SampleAnalyzer(ABC):
    """Base class for sample analyzer plugins that analyze individual samples."""

    def __init__(self):
        """Initialize the sample analyzer."""
        pass

    @abstractmethod
    def analyze_message(
        self, text_content: str, message_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze a single message and return metrics.

        Args:
            text_content: The text content to analyze
            message_metadata: Metadata about the message

        Returns:
            Dictionary containing analysis metrics
        """
        pass


class AnalyzerRegistry:
    """Registry for sample analyzer plugins."""

    _analyzers: dict[str, type[SampleAnalyzer]] = {}

    @classmethod
    def register(cls, analyzer_id: str, analyzer_class: type[SampleAnalyzer]) -> None:
        """Register a sample analyzer class.

        Args:
            analyzer_id: Unique identifier for the analyzer
            analyzer_class: The sample analyzer class to register

        Raises:
            ValueError: If the analyzer_id is already registered
        """
        if analyzer_id in cls._analyzers:
            raise ValueError(f"Analyzer ID '{analyzer_id}' is already registered")
        cls._analyzers[analyzer_id] = analyzer_class

    @classmethod
    def get_analyzer(cls, analyzer_id: str) -> Union[type[SampleAnalyzer], None]:
        """Get a sample analyzer class by ID.

        Args:
            analyzer_id: The analyzer ID to look up

        Returns:
            The sample analyzer class or None if not found
        """
        return cls._analyzers.get(analyzer_id)

    @classmethod
    def list_analyzers(cls) -> list[str]:
        """List all registered sample analyzer IDs.

        Returns:
            List of registered sample analyzer IDs
        """
        return list(cls._analyzers.keys())

    @classmethod
    def create_analyzer(cls, analyzer_id: str) -> SampleAnalyzer:
        """Create a sample analyzer instance.

        Args:
            analyzer_id: The analyzer ID to create

        Returns:
            An instance of the sample analyzer

        Raises:
            ValueError: If the analyzer ID is not registered
        """
        analyzer_class = cls.get_analyzer(analyzer_id)
        if analyzer_class is None:
            raise ValueError(f"Unknown analyzer ID: {analyzer_id}")
        return analyzer_class()
