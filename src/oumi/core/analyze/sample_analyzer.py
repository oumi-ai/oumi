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
from typing import Optional

import pandas as pd


class SampleAnalyzer(ABC):
    """Base class for sample analyzer plugins that analyze individual samples.

    All analyzers work with pandas DataFrames for efficient processing.
    """

    @abstractmethod
    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields and return analysis results.

        This method performs analysis on the input DataFrame and returns
        the DataFrame with added analysis columns. All analyzers must implement this
        method.

        Args:
            df: Input DataFrame with text fields
            schema: Column schema dict to identify text fields

        Returns:
            DataFrame with added analysis columns
        """
        pass
