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

"""Base classes for analyzer plugins."""

from abc import ABC, abstractmethod
from typing import Any, Union


class BaseAnalyzer(ABC):
    """Base class for all analyzer plugins."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the analyzer with configuration.

        Args:
            config: Configuration dictionary for the analyzer
        """
        self.config = config

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
    """Registry for analyzer plugins."""

    _analyzers: dict[str, type[BaseAnalyzer]] = {}

    @classmethod
    def register(cls, analyzer_id: str, analyzer_class: type[BaseAnalyzer]) -> None:
        """Register an analyzer class.

        Args:
            analyzer_id: Unique identifier for the analyzer
            analyzer_class: The analyzer class to register
        """
        cls._analyzers[analyzer_id] = analyzer_class

    @classmethod
    def get_analyzer(cls, analyzer_id: str) -> Union[type[BaseAnalyzer], None]:
        """Get an analyzer class by ID.

        Args:
            analyzer_id: The analyzer ID to look up

        Returns:
            The analyzer class or None if not found
        """
        return cls._analyzers.get(analyzer_id)

    @classmethod
    def list_analyzers(cls) -> list[str]:
        """List all registered analyzer IDs.

        Returns:
            List of registered analyzer IDs
        """
        return list(cls._analyzers.keys())

    @classmethod
    def create_analyzer(cls, analyzer_id: str, config: dict[str, Any]) -> BaseAnalyzer:
        """Create an analyzer instance.

        Args:
            analyzer_id: The analyzer ID to create
            config: Configuration for the analyzer

        Returns:
            An instance of the analyzer

        Raises:
            ValueError: If the analyzer ID is not registered
        """
        analyzer_class = cls.get_analyzer(analyzer_id)
        if analyzer_class is None:
            raise ValueError(f"Unknown analyzer ID: {analyzer_id}")
        return analyzer_class(config)
