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

import pandas as pd

# Standard text columns that analyzers typically process
DEFAULT_TEXT_COLUMNS = ["text_content", "conversation_text_content"]


class SampleAnalyzer(ABC):
    """Base class for sample analyzer plugins that analyze individual samples.

    All analyzers work with pandas DataFrames for efficient processing.
    """

    # Whether this analyzer requires an LLM for analysis.
    # Set to True in subclasses that use LLM inference (e.g., LLMJudgeAnalyzer).
    requires_llm: bool = False

    # Whether this analyzer requires a remote LLM API (e.g., OpenAI, Anthropic).
    # Set to True in subclasses that use remote API inference by default.
    # Local model-based analyzers (e.g., IFDAnalyzer) should keep this False.
    requires_remote_llm: bool = False

    def get_output_schema(
        self,
        df: pd.DataFrame | None = None,
        schema: dict | None = None,
        analyzer_id: str | None = None,
    ) -> dict:
        """Return the schema this analyzer will produce.

        This is the single source of truth for schema definitions.

        Args:
            df: DataFrame to analyze. Used to determine which columns exist.
                If None, uses DEFAULT_TEXT_COLUMNS for preview purposes.
            schema: Column schema dict to identify text columns.
                If None, uses DEFAULT_TEXT_COLUMNS for preview purposes.
            analyzer_id: The analyzer ID used for column naming. If None,
                uses the class's default or 'unknown'.

        Returns:
            Schema dict mapping column names to their schema config with keys:
            'type', 'content_type', 'description'. Returns empty dict if the
            analyzer doesn't implement this method.

        Note:
            Subclasses should override this method to provide their specific
            output schema. The default implementation returns an empty dict.
        """
        return {}

    @abstractmethod
    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: dict | None = None,
    ) -> pd.DataFrame:
        """Analyze text fields and return the DataFrame with analysis results.

        Args:
            df: Input DataFrame with text fields
            schema: Column schema dict to identify text fields

        Returns:
            DataFrame with added analysis columns.
            Caller should call get_output_schema(df, schema) to get the schema.
        """
        pass
