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

"""Analysis pipeline for orchestrating multiple analyzers."""

import json
import logging
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel

from oumi.analyze.base import (
    ConversationAnalyzer,
    DatasetAnalyzer,
    MessageAnalyzer,
    PreferenceAnalyzer,
)
from oumi.core.types.conversation import Conversation

logger = logging.getLogger(__name__)

# Type alias for any analyzer
AnyAnalyzer = Union[
    MessageAnalyzer[Any],
    ConversationAnalyzer[Any],
    DatasetAnalyzer[Any],
    PreferenceAnalyzer[Any],
]


class AnalysisPipeline:
    """Pipeline for orchestrating multiple analyzers on a dataset.

    The AnalysisPipeline manages running multiple analyzers on conversations,
    handling different analyzer scopes appropriately, and providing unified
    access to results.

    Example:
        >>> from oumi.analyze import AnalysisPipeline, LengthAnalyzer
        >>>
        >>> pipeline = AnalysisPipeline(
        ...     analyzers=[LengthAnalyzer(count_tokens=True)],
        ...     cache_dir="./analysis_cache",
        ... )
        >>> results = pipeline.run(conversations)
        >>>
        >>> # Access results by analyzer name
        >>> length_results = results["LengthAnalyzer"]
        >>> for r in length_results:
        ...     print(f"Words: {r.total_words}")
        >>>
        >>> # Convert to DataFrame for analysis
        >>> df = pipeline.to_dataframe()

    Args:
        analyzers: List of analyzer instances to run.
        cache_dir: Optional directory for caching results.
    """

    def __init__(
        self,
        analyzers: list[AnyAnalyzer],
        cache_dir: str | Path | None = None,
    ):
        """Initialize the analysis pipeline.

        Args:
            analyzers: List of analyzer instances to run.
            cache_dir: Optional directory for caching results.
        """
        self.analyzers = analyzers
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Results storage
        self._results: dict[str, list[BaseModel] | BaseModel] = {}
        self._conversations: list[Conversation] = []

        # Categorize analyzers by type for appropriate handling
        self._message_analyzers: list[MessageAnalyzer[Any]] = []
        self._conversation_analyzers: list[ConversationAnalyzer[Any]] = []
        self._dataset_analyzers: list[DatasetAnalyzer[Any]] = []
        self._preference_analyzers: list[PreferenceAnalyzer[Any]] = []

        for analyzer in analyzers:
            if isinstance(analyzer, MessageAnalyzer):
                self._message_analyzers.append(analyzer)
            elif isinstance(analyzer, ConversationAnalyzer):
                self._conversation_analyzers.append(analyzer)
            elif isinstance(analyzer, DatasetAnalyzer):
                self._dataset_analyzers.append(analyzer)
            elif isinstance(analyzer, PreferenceAnalyzer):
                self._preference_analyzers.append(analyzer)

    def run(
        self,
        conversations: list[Conversation],
    ) -> dict[str, list[BaseModel] | BaseModel]:
        """Run all analyzers on the provided conversations.

        Args:
            conversations: List of conversations to analyze.

        Returns:
            Dictionary mapping analyzer names to their results.
            - For ConversationAnalyzer: list of results (one per conversation)
            - For MessageAnalyzer: list of results (one per message)
            - For DatasetAnalyzer: single result for entire dataset
        """
        self._conversations = conversations
        self._results = {}

        logger.info(
            f"Running analysis pipeline with {len(self.analyzers)} analyzers "
            f"on {len(conversations)} conversations"
        )

        # Separate primary and derived analyzers
        from oumi.analyze.custom_metrics import CustomConversationMetric

        primary_analyzers = []
        derived_analyzers = []

        for analyzer in self._conversation_analyzers:
            if isinstance(analyzer, CustomConversationMetric) and analyzer.depends_on:
                derived_analyzers.append(analyzer)
            else:
                primary_analyzers.append(analyzer)

        # Run primary conversation-level analyzers first
        for analyzer in primary_analyzers:
            name = self._get_analyzer_name(analyzer)
            logger.debug(f"Running conversation analyzer: {name}")
            try:
                results = analyzer.analyze_batch(conversations)
                self._results[name] = results
                logger.debug(f"  Completed {name}: {len(results)} results")
            except Exception as e:
                logger.error(f"  Failed {name}: {e}")
                raise

        # Run derived analyzers (with access to primary results)
        if derived_analyzers:
            logger.debug("Running derived conversation analyzers...")
            CustomConversationMetric.set_pipeline_results(self._results)
            try:
                for analyzer in derived_analyzers:
                    name = self._get_analyzer_name(analyzer)
                    logger.debug(f"Running derived analyzer: {name}")
                    try:
                        results = analyzer.analyze_batch(conversations)
                        self._results[name] = results
                        logger.debug(f"  Completed {name}: {len(results)} results")
                    except Exception as e:
                        logger.error(f"  Failed {name}: {e}")
                        raise
            finally:
                CustomConversationMetric.clear_pipeline_results()

        # Run message-level analyzers
        for analyzer in self._message_analyzers:
            name = self._get_analyzer_name(analyzer)
            logger.debug(f"Running message analyzer: {name}")
            try:
                # Flatten all messages from all conversations
                all_messages = [msg for conv in conversations for msg in conv.messages]
                results = analyzer.analyze_batch(all_messages)
                self._results[name] = results
                logger.debug(f"  Completed {name}: {len(results)} results")
            except Exception as e:
                logger.error(f"  Failed {name}: {e}")
                raise

        # Run dataset-level analyzers
        for analyzer in self._dataset_analyzers:
            name = self._get_analyzer_name(analyzer)
            logger.debug(f"Running dataset analyzer: {name}")
            try:
                result = analyzer.analyze(conversations)
                self._results[name] = result  # Single result, not list
                logger.debug(f"  Completed {name}")
            except Exception as e:
                logger.error(f"  Failed {name}: {e}")
                raise

        logger.info(f"Analysis complete: {len(self._results)} analyzer results")

        # Cache results if cache_dir is set
        if self.cache_dir:
            self._save_cache()

        return self._results

    def run_preference(
        self,
        pairs: list[tuple[Conversation, Conversation]],
    ) -> dict[str, list[BaseModel]]:
        """Run preference analyzers on conversation pairs.

        Args:
            pairs: List of (chosen, rejected) conversation tuples.

        Returns:
            Dictionary mapping analyzer names to their results.
        """
        results: dict[str, list[BaseModel]] = {}

        for analyzer in self._preference_analyzers:
            name = self._get_analyzer_name(analyzer)
            logger.debug(f"Running preference analyzer: {name}")
            try:
                analyzer_results = analyzer.analyze_batch(pairs)
                results[name] = analyzer_results
                logger.debug(f"  Completed {name}: {len(analyzer_results)} results")
            except Exception as e:
                logger.error(f"  Failed {name}: {e}")
                raise

        return results

    def to_dataframe(self):
        """Convert cached results to a pandas DataFrame.

        Returns:
            DataFrame with one row per conversation, columns for each metric.

        Raises:
            RuntimeError: If no results are cached (run() not called).
        """
        from oumi.analyze.utils.dataframe import to_analysis_dataframe

        if not self._results:
            raise RuntimeError(
                "No results available. Call run() first to analyze conversations."
            )

        return to_analysis_dataframe(self._conversations, self._results)

    @property
    def results(self) -> dict[str, list[BaseModel] | BaseModel]:
        """Get the cached analysis results.

        Returns:
            Dictionary mapping analyzer names to results.
        """
        return self._results

    @property
    def conversations(self) -> list[Conversation]:
        """Get the analyzed conversations.

        Returns:
            List of conversations that were analyzed.
        """
        return self._conversations

    def _get_analyzer_name(self, analyzer: AnyAnalyzer) -> str:
        """Get the name for an analyzer.

        Uses the class name by default, but can be overridden by
        setting an 'analyzer_id' attribute on the analyzer.

        Args:
            analyzer: The analyzer instance.

        Returns:
            Name string for the analyzer.
        """
        if hasattr(analyzer, "analyzer_id"):
            return analyzer.analyzer_id
        return analyzer.__class__.__name__

    def _save_cache(self) -> None:
        """Save results to cache directory."""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Save results as JSON
        results_path = self.cache_dir / "analysis_results.json"
        serialized = {}
        for name, result in self._results.items():
            if isinstance(result, list):
                serialized[name] = [r.model_dump() for r in result]
            else:
                serialized[name] = result.model_dump()

        with open(results_path, "w") as f:
            json.dump(serialized, f, indent=2, default=str)

        logger.debug(f"Cached results to {results_path}")

    def get_analyzer(self, name: str) -> AnyAnalyzer | None:
        """Get an analyzer by name.

        Args:
            name: Name of the analyzer to find.

        Returns:
            Analyzer instance or None if not found.
        """
        for analyzer in self.analyzers:
            if self._get_analyzer_name(analyzer) == name:
                return analyzer
        return None
