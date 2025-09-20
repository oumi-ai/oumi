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

from oumi.core.analyze.dataset_analyzer import (
    FieldAnalysisResult,
    SampleAnalysisResult,
)


class SampleAnalyzer(ABC):
    """Base class for sample analyzer plugins that analyze individual samples.
    
    All analyzers work with dictionary data.
    """

    @abstractmethod
    def analyze_fields(
        self,
        text_fields: list[tuple[str, str]],
        tokenizer: Optional[Any] = None
    ) -> list[FieldAnalysisResult]:
        """Analyze individual text fields.
        
        This method provides field-level analysis for dictionary data. All analyzers
        must implement this method.
        
        Args:
            text_fields: List of (field_name, text_content) tuples
            tokenizer: Optional tokenizer to use for analysis
            
        Returns:
            List of FieldAnalysisResult objects, one for each field
        """
        pass

    @abstractmethod
    def analyze_sample(
        self, 
        sample: dict, 
        text_fields: list[str],
        tokenizer: Optional[Any] = None
    ) -> SampleAnalysisResult:
        """Analyze a dictionary sample as a whole.
        
        This method provides sample-level analysis for dictionary data. All analyzers
        must implement this method.
        
        Args:
            sample: The sample dictionary to analyze
            text_fields: List of field names that contain text content to analyze
            tokenizer: Optional tokenizer to use for analysis
            
        Returns:
            SampleAnalysisResult for the entire sample
        """
        pass
