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

    @abstractmethod
    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: dict | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze text fields and return analysis results.

        This method performs analysis on the input DataFrame and returns
        the DataFrame with added analysis columns along with schema information
        for the generated columns. All analyzers must implement this method.

        Args:
            df: Input DataFrame with text fields
            schema: Column schema dict to identify text fields

        Returns:
            Tuple of (DataFrame with added analysis columns,
            generated column schema dict).
            The schema dict maps column names to their schema config with keys:
            'type', 'content_type', 'description'.
        """
        pass
