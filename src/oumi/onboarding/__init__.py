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

"""Oumi Onboarding Module.

This module provides tools to help customers quickly onboard to Oumi by:
- Analyzing customer data formats (CSV, Excel, JSON, Word)
- Auto-generating configuration files for synth, judge, and train
- Mapping customer fields to Oumi placeholders
- Orchestrating end-to-end pipelines

Example:
    >>> from oumi.onboarding import DataAnalyzer, SynthConfigBuilder
    >>> analyzer = DataAnalyzer()
    >>> schema = analyzer.analyze("./customer_data.csv")
    >>> builder = SynthConfigBuilder()
    >>> config = builder.from_schema(schema, goal="qa")
"""

from oumi.onboarding.config_builder import (
    BuilderOptions,
    ConfigBuilder,
    JudgeConfigBuilder,
    SynthConfigBuilder,
    TrainConfigBuilder,
)
from oumi.onboarding.data_analyzer import (
    ColumnInfo,
    DataAnalyzer,
    DataSchema,
)
from oumi.onboarding.field_mapper import FieldMapper, FieldMapping

__all__ = [
    "BuilderOptions",
    "ColumnInfo",
    "ConfigBuilder",
    "DataAnalyzer",
    "DataSchema",
    "FieldMapper",
    "FieldMapping",
    "JudgeConfigBuilder",
    "SynthConfigBuilder",
    "TrainConfigBuilder",
]
