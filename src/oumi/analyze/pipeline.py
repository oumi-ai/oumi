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

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    import pandas as pd

from oumi.analyze.base import (
    ConversationAnalyzer,
    DatasetAnalyzer,
    MessageAnalyzer,
    PreferenceAnalyzer,
)
from oumi.core.types.conversation import Conversation

logger = logging.getLogger(__name__)

# Constants
_CACHE_FILENAME = "analysis_results.json"

# Type aliases for consistency
AnyAnalyzer = (
    MessageAnalyzer[Any]
    | ConversationAnalyzer[Any]
    | DatasetAnalyzer[Any]
    | PreferenceAnalyzer[Any]
)
AnalysisResults = dict[str, list[BaseModel] | BaseModel]

T = TypeVar("T")


class AnalysisPipeline:
    """Pipeline for orchestrating multiple analyzers on a dataset.

    The AnalysisPipeline manages running multiple analyzers on conversations,
    handling different analyzer scopes appropriately, and providing unified
    access to results.

    Note:
        PreferenceAnalyzers are not run by `run()`. Use `run_preference()`
        separately to analyze preference pairs (chosen/rejected conversations).

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
        self._results: AnalysisResults = {}
        self._conversations: list[Conversation] = []
        # Track which conversation each message belongs to (for message analyzers)
        self._message_to_conversation_idx: list[int] = []

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
    ) -> AnalysisResults:
        """Run all analyzers on the provided conversations.

        Note:
            PreferenceAnalyzers are not run by this method. Use `run_preference()`
            separately to analyze preference pairs.

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

        # Build message-to-conversation index for later correlation
        self._message_to_conversation_idx = []
        for conv_idx, conv in enumerate(conversations):
            for _ in conv.messages:
                self._message_to_conversation_idx.append(conv_idx)

        logger.info(
            f"Running analysis pipeline with {len(self.analyzers)} analyzers "
            f"on {len(conversations)} conversations"
        )

        # Run conversation-level analyzers
        self._run_conversation_analyzers(conversations)

        # Run message-level analyzers
        self._run_message_analyzers(conversations)

        # Run dataset-level analyzers
        self._run_dataset_analyzers(conversations)

        logger.info(f"Analysis complete: {len(self._results)} analyzer results")

        # Cache results if cache_dir is set
        if self.cache_dir:
            self._save_cache()

        return self._results

    def run_preference(
        self,
        pairs: list[tuple[Conversation, Conversation]],
    ) -> AnalysisResults:
        """Run preference analyzers on conversation pairs.

        Args:
            pairs: List of (chosen, rejected) conversation tuples.

        Returns:
            Dictionary mapping analyzer names to their results.
        """
        results: AnalysisResults = {}

        sorted_analyzers = self._topological_sort(self._preference_analyzers)

        for analyzer in sorted_analyzers:
            self._inject_dependencies(analyzer)
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

    def to_dataframe(self) -> "pd.DataFrame":
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

        return to_analysis_dataframe(
            self._conversations,
            self._results,
            message_to_conversation_idx=self._message_to_conversation_idx,
        )

    def load_cache(self) -> bool:
        """Load results from cache directory.

        Note:
            Loaded results are raw dictionaries, not Pydantic model instances.
            Use `get_cached_result()` to reconstruct typed results if needed,
            or access raw data directly via `self.results`.

        Returns:
            True if cache was loaded successfully, False otherwise.
        """
        import json

        if not self.cache_dir:
            return False

        results_path = self.cache_dir / _CACHE_FILENAME
        if not results_path.exists():
            logger.debug(f"No cache found at {results_path}")
            return False

        try:
            with open(results_path) as f:
                self._results = json.load(f)
            logger.debug(f"Loaded cached results from {results_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    @property
    def results(self) -> AnalysisResults:
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

    @property
    def message_to_conversation_idx(self) -> list[int]:
        """Get the mapping from message index to conversation index.

        This is useful for correlating message-level results back to
        their parent conversations.

        Returns:
            List where index i contains the conversation index for message i.
        """
        return self._message_to_conversation_idx

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

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _run_conversation_analyzers(self, conversations: list[Conversation]) -> None:
        """Run all conversation-level analyzers in dependency order."""
        sorted_analyzers = self._topological_sort(self._conversation_analyzers)

        for analyzer in self._iter_with_progress(
            sorted_analyzers, "Running conversation analyzers"
        ):
            self._inject_dependencies(analyzer)
            self._run_single_analyzer(
                analyzer, lambda a: a.analyze_batch(conversations), is_batch=True
            )

    def _run_message_analyzers(self, conversations: list[Conversation]) -> None:
        """Run all message-level analyzers in dependency order."""
        if not self._message_analyzers:
            return

        # Flatten all messages from all conversations
        all_messages = [msg for conv in conversations for msg in conv.messages]

        sorted_analyzers = self._topological_sort(self._message_analyzers)

        for analyzer in self._iter_with_progress(
            sorted_analyzers, "Running message analyzers"
        ):
            self._inject_dependencies(analyzer)
            self._run_single_analyzer(
                analyzer, lambda a: a.analyze_batch(all_messages), is_batch=True
            )

    def _run_dataset_analyzers(self, conversations: list[Conversation]) -> None:
        """Run all dataset-level analyzers in dependency order."""
        sorted_analyzers = self._topological_sort(self._dataset_analyzers)

        for analyzer in self._iter_with_progress(
            sorted_analyzers, "Running dataset analyzers"
        ):
            self._inject_dependencies(analyzer)
            self._run_single_analyzer(
                analyzer, lambda a: a.analyze(conversations), is_batch=False
            )

    def _run_single_analyzer(
        self,
        analyzer: AnyAnalyzer,
        run_func: Any,
        is_batch: bool,
    ) -> None:
        """Run a single analyzer and store results.

        Args:
            analyzer: The analyzer to run.
            run_func: Function that takes the analyzer and returns results.
            is_batch: Whether the result is a list (batch) or single value.
        """
        name = self._get_analyzer_name(analyzer)
        scope = self._get_analyzer_scope(analyzer)
        logger.debug(f"Running {scope} analyzer: {name}")

        try:
            result = run_func(analyzer)
            self._results[name] = result
            if is_batch and isinstance(result, list):
                logger.debug(f"  Completed {name}: {len(result)} results")
            else:
                logger.debug(f"  Completed {name}")
        except Exception as e:
            logger.error(f"  Failed {name}: {e}")
            raise

    def _topological_sort(self, analyzers: list[T]) -> list[T]:
        """Sort analyzers by dependencies using topological sort.

        Uses Python's built-in graphlib.TopologicalSorter for proper
        dependency ordering. Handles chained dependencies like A → B → C.

        Args:
            analyzers: List of analyzers to sort.

        Returns:
            List of analyzers in dependency order.

        Raises:
            ValueError: If there's a circular dependency.
        """
        from graphlib import CycleError, TopologicalSorter

        if not analyzers:
            return []

        # Build name -> analyzer mapping
        name_to_analyzer: dict[str, T] = {}
        for analyzer in analyzers:
            name = self._get_analyzer_name(analyzer)  # type: ignore[arg-type]
            name_to_analyzer[name] = analyzer

        all_names = set(name_to_analyzer.keys())

        # Build dependency graph: {node: [dependencies]}
        graph: dict[str, set[str]] = {}
        for analyzer in analyzers:
            name = self._get_analyzer_name(analyzer)  # type: ignore[arg-type]
            depends_on = getattr(analyzer, "depends_on", None) or []
            # Only include dependencies within this analyzer group
            graph[name] = {dep for dep in depends_on if dep in all_names}

        try:
            sorter = TopologicalSorter(graph)
            sorted_names = list(sorter.static_order())
        except CycleError as e:
            raise ValueError(f"Circular dependency detected: {e}") from e

        return [name_to_analyzer[name] for name in sorted_names]

    def _inject_dependencies(self, analyzer: AnyAnalyzer) -> None:
        """Inject dependency results into a derived analyzer.

        If the analyzer has a `depends_on` attribute listing dependency names,
        and a `set_dependencies` method, this will pass the results from
        those dependencies to the analyzer.

        Args:
            analyzer: The derived analyzer to inject dependencies into.
        """
        depends_on = getattr(analyzer, "depends_on", None)
        if not depends_on:
            return

        # Check if analyzer can receive dependencies
        if not hasattr(analyzer, "set_dependencies"):
            logger.warning(
                f"Analyzer {self._get_analyzer_name(analyzer)} has depends_on "
                f"but no set_dependencies method"
            )
            return

        # Gather dependency results
        dependency_results: dict[str, list[BaseModel] | BaseModel] = {}
        for dep_name in depends_on:
            if dep_name in self._results:
                dependency_results[dep_name] = self._results[dep_name]
            else:
                logger.warning(
                    f"Dependency '{dep_name}' not found for analyzer "
                    f"'{self._get_analyzer_name(analyzer)}'"
                )

        # Inject dependencies
        analyzer.set_dependencies(dependency_results)  # type: ignore[union-attr]

    def _iter_with_progress(self, items: list[T], desc: str) -> Iterable[T]:
        """Iterate with optional progress bar.

        Args:
            items: Items to iterate over.
            desc: Description for the progress bar.

        Returns:
            Iterator (with tqdm wrapper if available).
        """
        try:
            from tqdm import tqdm

            return tqdm(items, desc=desc, unit="analyzer")
        except ImportError:
            return items

    def _get_analyzer_name(self, analyzer: AnyAnalyzer) -> str:
        """Get the name for an analyzer.

        Uses the class name by default, but can be overridden by
        setting an 'analyzer_id' attribute on the analyzer.

        Args:
            analyzer: The analyzer instance.

        Returns:
            Name string for the analyzer.
        """
        analyzer_id = getattr(analyzer, "analyzer_id", None)
        if analyzer_id is not None:
            return str(analyzer_id)
        return analyzer.__class__.__name__

    def _get_analyzer_scope(self, analyzer: AnyAnalyzer) -> str:
        """Get the scope name for an analyzer.

        Args:
            analyzer: The analyzer instance.

        Returns:
            Scope string ('message', 'conversation', 'dataset', or 'preference').
        """
        if isinstance(analyzer, MessageAnalyzer):
            return "message"
        elif isinstance(analyzer, ConversationAnalyzer):
            return "conversation"
        elif isinstance(analyzer, DatasetAnalyzer):
            return "dataset"
        elif isinstance(analyzer, PreferenceAnalyzer):
            return "preference"
        return "unknown"

    def _save_cache(self) -> None:
        """Save results to cache directory."""
        import json

        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Save results as JSON
        results_path = self.cache_dir / _CACHE_FILENAME
        serialized = {}
        for name, result in self._results.items():
            if isinstance(result, list):
                serialized[name] = [r.model_dump() for r in result]
            else:
                serialized[name] = result.model_dump()

        with open(results_path, "w") as f:
            json.dump(serialized, f, indent=2, default=str)

        logger.debug(f"Cached results to {results_path}")
