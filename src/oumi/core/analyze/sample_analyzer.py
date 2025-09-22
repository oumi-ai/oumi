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
from typing import Any, Optional
import pandas as pd


class SampleAnalyzer(ABC):
    """Base class for sample analyzer plugins that analyze individual samples.
    
    All analyzers work with pandas DataFrames for efficient processing.
    """

    @abstractmethod
    def analyze_fields(
        self, 
        df: pd.DataFrame, 
        text_fields: list[str],
        tokenizer: Optional[Any] = None
    ) -> pd.DataFrame:
        """Analyze individual text fields and add field-level metrics to DataFrame.
        
        This method adds field-level analysis columns to the input DataFrame.
        All analyzers must implement this method.
        
        Args:
            df: Input DataFrame with text fields
            text_fields: List of field names that contain text content to analyze
            tokenizer: Optional tokenizer to use for analysis
            
        Returns:
            DataFrame with added field-level analysis columns
        """
        pass

    @abstractmethod
    def analyze_sample(
        self, 
        df: pd.DataFrame, 
        text_fields: list[str],
        tokenizer: Optional[Any] = None
    ) -> pd.DataFrame:
        """Analyze samples as a whole and add sample-level metrics to DataFrame.
        
        This method adds sample-level analysis columns to the input DataFrame.
        All analyzers must implement this method.
        
        Args:
            df: Input DataFrame with text fields
            text_fields: List of field names that contain text content to analyze
            tokenizer: Optional tokenizer to use for analysis
            
        Returns:
            DataFrame with added sample-level analysis columns
        """
        pass
