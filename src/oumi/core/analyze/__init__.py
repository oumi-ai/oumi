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

"""Sample analyzer plugin system for OUMI.

This package provides a plugin-based architecture for analyzing conversation data
with different types of sample analyzers (length, safety, etc.).
"""

# Import base classes
from oumi.core.analyze.dataset_analyzer import (
    DatasetAnalysisResult,
    DatasetAnalyzer,
    MessageAnalysisResult,
)

# Import analyzers to register them
from oumi.core.analyze.length_analyzer import LengthAnalyzer
from oumi.core.analyze.sample_analyzer import SampleAnalyzer

__all__ = [
    "SampleAnalyzer",
    "DatasetAnalyzer",
    "DatasetAnalysisResult",
    "MessageAnalysisResult",
    "LengthAnalyzer",
]
