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

"""Analyzer registry for the typed analyzer framework.

This module provides the central registry for all analyzers.
"""

from typing import Any

from oumi.analyze.base import ConversationAnalyzer, DatasetAnalyzer, MessageAnalyzer

# Global registry for analyzers
ANALYZER_REGISTRY: dict[str, type] = {}


def register_analyzer(name: str):
    """Decorator to register an analyzer class.

    Args:
        name: Name to register the analyzer under.

    Returns:
        Decorator function.
    """

    def decorator(cls):
        ANALYZER_REGISTRY[name] = cls
        return cls

    return decorator


def get_analyzer_class(name: str) -> type | None:
    """Get an analyzer class by name.

    Args:
        name: Name of the analyzer (e.g., "length" or "LengthAnalyzer").

    Returns:
        The analyzer class or None if not found.
    """
    return ANALYZER_REGISTRY.get(name)


def _register_builtin_analyzers():
    """Register built-in analyzers in the registry."""
    # Non-LLM analyzers (fast, cheap)
    try:
        from oumi.analyze.analyzers.length import LengthAnalyzer

        ANALYZER_REGISTRY["length"] = LengthAnalyzer
        ANALYZER_REGISTRY["LengthAnalyzer"] = LengthAnalyzer
    except ImportError:
        pass

    try:
        from oumi.analyze.analyzers.turn_stats import TurnStatsAnalyzer

        ANALYZER_REGISTRY["turn_stats"] = TurnStatsAnalyzer
        ANALYZER_REGISTRY["TurnStatsAnalyzer"] = TurnStatsAnalyzer
    except ImportError:
        pass

    try:
        from oumi.analyze.analyzers.quality import DataQualityAnalyzer

        ANALYZER_REGISTRY["quality"] = DataQualityAnalyzer
        ANALYZER_REGISTRY["DataQualityAnalyzer"] = DataQualityAnalyzer
    except ImportError:
        pass

    # Dataset-level analyzers
    try:
        from oumi.analyze.analyzers.deduplication import DeduplicationAnalyzer

        ANALYZER_REGISTRY["deduplication"] = DeduplicationAnalyzer
        ANALYZER_REGISTRY["DeduplicationAnalyzer"] = DeduplicationAnalyzer
    except ImportError:
        pass

    # LLM-based analyzers
    try:
        from oumi.analyze.analyzers.llm_analyzer import (
            CoherenceAnalyzer,
            FactualityAnalyzer,
            InstructionFollowingAnalyzer,
            LLMAnalyzer,
            SafetyAnalyzer,
            UsefulnessAnalyzer,
        )

        ANALYZER_REGISTRY["llm"] = LLMAnalyzer
        ANALYZER_REGISTRY["LLMAnalyzer"] = LLMAnalyzer
        ANALYZER_REGISTRY["usefulness"] = UsefulnessAnalyzer
        ANALYZER_REGISTRY["UsefulnessAnalyzer"] = UsefulnessAnalyzer
        ANALYZER_REGISTRY["safety"] = SafetyAnalyzer
        ANALYZER_REGISTRY["SafetyAnalyzer"] = SafetyAnalyzer
        ANALYZER_REGISTRY["factuality"] = FactualityAnalyzer
        ANALYZER_REGISTRY["FactualityAnalyzer"] = FactualityAnalyzer
        ANALYZER_REGISTRY["coherence"] = CoherenceAnalyzer
        ANALYZER_REGISTRY["CoherenceAnalyzer"] = CoherenceAnalyzer
        ANALYZER_REGISTRY["instruction_following"] = InstructionFollowingAnalyzer
        ANALYZER_REGISTRY["InstructionFollowingAnalyzer"] = InstructionFollowingAnalyzer
    except ImportError:
        pass


# Register on module import
_register_builtin_analyzers()


def create_analyzer_from_config(
    analyzer_id: str,
    params: dict[str, Any],
) -> MessageAnalyzer | ConversationAnalyzer | DatasetAnalyzer | None:
    """Create an analyzer instance from configuration.

    Args:
        analyzer_id: Analyzer type identifier.
        params: Analyzer-specific parameters.

    Returns:
        Analyzer instance or None if not found.
    """
    import logging

    logger = logging.getLogger(__name__)

    analyzer_class = ANALYZER_REGISTRY.get(analyzer_id)
    if analyzer_class is None:
        logger.warning(f"Unknown analyzer: {analyzer_id}")
        return None

    try:
        return analyzer_class(**params)
    except Exception as e:
        logger.error(f"Failed to create analyzer {analyzer_id}: {e}")
        return None
