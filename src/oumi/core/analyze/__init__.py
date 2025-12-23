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

"""Sample analyzer plugin system for Oumi.

This package provides a plugin-based architecture for analyzing conversation data
with different types of sample analyzers (length, safety, etc.).
"""

# Import analyzers to register them
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.analyze.length_analyzer import LengthAnalyzer
from oumi.core.analyze.sample_analyzer import SampleAnalyzer

# Conditional import for EmbeddingBasedAnalyzer base class
try:
    from oumi.core.analyze.embedding_base_analyzer import EmbeddingBasedAnalyzer
except ImportError:
    EmbeddingBasedAnalyzer = None  # type: ignore[misc, assignment]

# Conditional import for EmbeddingAnalyzer (requires optional dependencies)
try:
    from oumi.core.analyze.embedding_analyzer import EmbeddingAnalyzer
except ImportError:
    EmbeddingAnalyzer = None  # type: ignore[misc, assignment]

__all__ = [
    "DatasetAnalyzer",
    "EmbeddingAnalyzer",
    "EmbeddingBasedAnalyzer",
    "LengthAnalyzer",
    "SampleAnalyzer",
]
