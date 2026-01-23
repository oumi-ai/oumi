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

"""Custom metrics support for user-defined Python functions.

This module provides the infrastructure for executing user-defined Python
functions as custom metrics. Custom metrics allow users to define additional
computations without creating full analyzer classes.

Custom metrics are computed during the analysis phase and cached with other
results. This separates the expensive computation from the cheap validation
(tests).

Example YAML config:
    custom_metrics:
      - id: turn_pattern
        scope: conversation
        function: |
          def compute(conversation):
              roles = [m.role.value for m in conversation.messages]
              alternating = all(roles[i] != roles[i+1] for i in range(len(roles)-1))
              return {"has_alternating_turns": alternating}
"""

import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from oumi.analyze.base import ConversationAnalyzer, MessageAnalyzer
from oumi.analyze.config import CustomMetricConfig
from oumi.core.types.conversation import Conversation, Message

logger = logging.getLogger(__name__)


class CustomMetricResult(BaseModel):
    """Result model for custom metrics.

    Custom metrics return a dictionary of field names to values.
    This wrapper provides a consistent interface.
    """

    values: dict[str, Any] = Field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        """Allow accessing values as attributes."""
        if name in self.values:
            return self.values[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class CustomConversationMetric(ConversationAnalyzer[CustomMetricResult]):
    """Analyzer wrapper for custom conversation-level metrics.

    Executes a user-defined Python function for each conversation.

    The function can have two signatures:
        def compute(conversation: Conversation) -> dict[str, Any]
        def compute(conversation: Conversation, results: dict, index: int) -> dict[str, Any]

    The second signature allows accessing results from other analyzers:
        def compute(conversation, results, index):
            tokens = results["LengthAnalyzer"][index].total_tokens
            return {"cost": tokens * 0.001}

    Args:
        metric_id: Unique identifier for the metric.
        function_code: Python code defining the compute function.
        description: Optional description of the metric.
        depends_on: List of analyzer names this metric depends on.
    """

    # Class variable to hold results for derived metrics
    _pipeline_results: dict[str, Any] | None = None
    _current_index: int = 0

    def __init__(
        self,
        metric_id: str,
        function_code: str,
        description: str | None = None,
        depends_on: list[str] | None = None,
    ):
        """Initialize the custom metric.

        Args:
            metric_id: Unique identifier for the metric.
            function_code: Python code defining the compute function.
            description: Optional description of the metric.
            depends_on: List of analyzer names this metric depends on.
        """
        self.metric_id = metric_id
        self.analyzer_id = metric_id  # For pipeline naming
        self.function_code = function_code
        self.description = description
        self.depends_on = depends_on or []
        self._compute_func: Callable | None = None
        self._uses_results = False  # Will be set during compilation

        # Compile the function
        self._compile_function()

    def _compile_function(self) -> None:
        """Compile the user-provided function code."""
        # Create a namespace with allowed imports and builtins
        namespace: dict[str, Any] = {
            "__builtins__": {
                # Safe builtins
                "len": len,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "any": any,
                "all": all,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "True": True,
                "False": False,
                "None": None,
            },
        }

        # Add common imports
        try:
            import re

            namespace["re"] = re
        except ImportError:
            pass

        # Execute the function definition
        try:
            exec(self.function_code, namespace)
        except Exception as e:
            raise ValueError(f"Failed to compile custom metric '{self.metric_id}': {e}")

        # Get the compute function
        if "compute" not in namespace:
            raise ValueError(
                f"Custom metric '{self.metric_id}' must define a 'compute' function. "
                "Example: def compute(conversation): return {'key': value}"
            )

        self._compute_func = namespace["compute"]

        # Check if function accepts results parameter
        import inspect

        if self._compute_func is not None:
            sig = inspect.signature(self._compute_func)
            self._uses_results = len(sig.parameters) >= 2

    def analyze(self, conversation: Conversation) -> CustomMetricResult:
        """Run the custom metric on a conversation.

        Args:
            conversation: The conversation to analyze.

        Returns:
            CustomMetricResult with computed values.
        """
        if self._compute_func is None:
            return CustomMetricResult(values={})

        try:
            # If function uses results, pass them
            if self._uses_results and self._pipeline_results is not None:
                result = self._compute_func(
                    conversation, self._pipeline_results, self._current_index
                )
            else:
                result = self._compute_func(conversation)

            if not isinstance(result, dict):
                raise ValueError(
                    f"Custom metric '{self.metric_id}' must return a dict, "
                    f"got {type(result).__name__}"
                )
            return CustomMetricResult(values=result)
        except Exception as e:
            logger.warning(
                f"Custom metric '{self.metric_id}' failed for conversation: {e}"
            )
            return CustomMetricResult(values={"error": str(e)})

    def analyze_batch(
        self, conversations: list[Conversation]
    ) -> list[CustomMetricResult]:
        """Analyze a batch of conversations.

        Overrides base to track index for results access.
        """
        results = []
        for i, conv in enumerate(conversations):
            self._current_index = i
            results.append(self.analyze(conv))
        return results

    @classmethod
    def set_pipeline_results(cls, results: dict[str, Any]) -> None:
        """Set the pipeline results for derived metrics.

        Called by the pipeline after running primary analyzers.
        """
        cls._pipeline_results = results

    @classmethod
    def clear_pipeline_results(cls) -> None:
        """Clear the pipeline results."""
        cls._pipeline_results = None


class CustomMessageMetric(MessageAnalyzer[CustomMetricResult]):
    """Analyzer wrapper for custom message-level metrics.

    Executes a user-defined Python function for each message.

    The function should have the signature:
        def compute(message: Message) -> dict[str, Any]

    Args:
        metric_id: Unique identifier for the metric.
        function_code: Python code defining the compute function.
        description: Optional description of the metric.
    """

    def __init__(
        self,
        metric_id: str,
        function_code: str,
        description: str | None = None,
    ):
        """Initialize the custom metric.

        Args:
            metric_id: Unique identifier for the metric.
            function_code: Python code defining the compute function.
            description: Optional description of the metric.
        """
        self.metric_id = metric_id
        self.analyzer_id = metric_id  # For pipeline naming
        self.function_code = function_code
        self.description = description
        self._compute_func: Callable | None = None

        # Compile the function
        self._compile_function()

    def _compile_function(self) -> None:
        """Compile the user-provided function code."""
        # Same namespace as conversation metric
        namespace: dict[str, Any] = {
            "__builtins__": {
                "len": len,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "any": any,
                "all": all,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "True": True,
                "False": False,
                "None": None,
            },
        }

        try:
            import re

            namespace["re"] = re
        except ImportError:
            pass

        try:
            exec(self.function_code, namespace)
        except Exception as e:
            raise ValueError(f"Failed to compile custom metric '{self.metric_id}': {e}")

        if "compute" not in namespace:
            raise ValueError(
                f"Custom metric '{self.metric_id}' must define a 'compute' function. "
                "Example: def compute(message): return {'key': value}"
            )

        self._compute_func = namespace["compute"]

    def analyze(self, message: Message) -> CustomMetricResult:
        """Run the custom metric on a message.

        Args:
            message: The message to analyze.

        Returns:
            CustomMetricResult with computed values.
        """
        if self._compute_func is None:
            return CustomMetricResult(values={})

        try:
            result = self._compute_func(message)
            if not isinstance(result, dict):
                raise ValueError(
                    f"Custom metric '{self.metric_id}' must return a dict, "
                    f"got {type(result).__name__}"
                )
            return CustomMetricResult(values=result)
        except Exception as e:
            logger.warning(f"Custom metric '{self.metric_id}' failed for message: {e}")
            return CustomMetricResult(values={"error": str(e)})


def create_custom_metric(
    config: CustomMetricConfig,
) -> CustomConversationMetric | CustomMessageMetric:
    """Create a custom metric analyzer from configuration.

    Args:
        config: Custom metric configuration.

    Returns:
        Custom metric analyzer instance.
    """
    if config.scope == "conversation":
        return CustomConversationMetric(
            metric_id=config.id,
            function_code=config.function,
            description=config.description,
            depends_on=getattr(config, "depends_on", None),
        )
    elif config.scope == "message":
        return CustomMessageMetric(
            metric_id=config.id,
            function_code=config.function,
            description=config.description,
        )
    else:
        raise ValueError(f"Unknown custom metric scope: {config.scope}")
