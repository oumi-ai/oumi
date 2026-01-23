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

"""Configuration for the typed analyzer framework.

This module provides configuration classes for the new typed analyzer
architecture, supporting both programmatic and YAML-based configuration.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from oumi.analyze.testing.engine import TestConfig, TestType
from oumi.analyze.testing.results import TestSeverity


class AnalyzerType(str, Enum):
    """Built-in analyzer types."""

    LENGTH = "length"
    QUALITY = "quality"
    FORMAT = "format"
    DIVERSITY = "diversity"
    EMBEDDING = "embedding"
    LLM_JUDGE = "llm_judge"


@dataclass
class AnalyzerConfig:
    """Configuration for a single analyzer.

    Attributes:
        id: Analyzer type identifier (e.g., "length", "quality").
        instance_id: Optional unique instance ID for multiple analyzers of same type.
        params: Analyzer-specific parameters.
    """

    id: str
    instance_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-populate instance_id if not provided."""
        if self.instance_id is None:
            self.instance_id = self.id


@dataclass
class OutputFieldSchema:
    """Schema definition for a single output field.

    Attributes:
        name: Field name (key in the returned dict).
        type: Field type ("int", "float", "bool", "str", "list").
        description: Description of the field.
    """

    name: str
    type: str = "float"
    description: str = ""


@dataclass
class CustomMetricConfig:
    """Configuration for a custom user-defined metric.

    Custom metrics allow users to define Python functions that compute
    additional metrics. These are executed during the analysis phase
    and their results are cached.

    Example YAML:
        ```yaml
        custom_metrics:
          - id: word_to_char_ratio
            scope: conversation
            description: "Ratio of words to characters"
            output_schema:
              - name: ratio
                type: float
                description: "Words divided by characters (0.15-0.20 is typical)"
            function: |
              def compute(conversation):
                  chars = sum(len(m.content) for m in conversation.messages)
                  words = sum(len(m.content.split()) for m in conversation.messages)
                  return {"ratio": words / chars if chars > 0 else 0.0}
        ```

    Attributes:
        id: Unique identifier for the metric.
        scope: Scope of the metric ("message" or "conversation").
        function: Python code defining a compute() function.
        description: Description of what the metric computes.
        output_schema: List of output field definitions.
    """

    id: str
    scope: str = "conversation"  # "message" or "conversation"
    function: str = ""
    description: str | None = None
    output_schema: list[OutputFieldSchema] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate the configuration."""
        if self.scope not in ("message", "conversation"):
            raise ValueError(
                f"Invalid scope '{self.scope}'. Must be 'message' or 'conversation'."
            )

    def get_metric_paths(self) -> list[str]:
        """Get full metric paths for all output fields.

        Returns:
            List of metric paths like ["metric_id.field_name", ...].
        """
        if self.output_schema:
            return [f"{self.id}.{f.name}" for f in self.output_schema]
        return [f"{self.id}.<field>"]

    def get_field_info(self) -> dict[str, dict[str, str]]:
        """Get field information for display.

        Returns:
            Dict mapping field names to {"type": ..., "description": ...}.
        """
        return {
            f.name: {"type": f.type, "description": f.description}
            for f in self.output_schema
        }


@dataclass
class TestConfigYAML:
    """YAML-friendly test configuration.

    This class mirrors TestConfig but uses simpler types for YAML parsing.

    Attributes:
        id: Unique identifier for the test.
        type: Test type ("threshold", "percentage", "range").
        metric: Path to the metric (e.g., "LengthAnalyzer.total_words").
        severity: Severity level ("high", "medium", "low").
        title: Human-readable title.
        description: Description of the test.
        operator: Comparison operator for threshold tests.
        value: Value to compare against.
        condition: Condition for percentage tests.
        max_percentage: Maximum allowed percentage.
        min_percentage: Minimum required percentage.
        min_value: Minimum value for range tests.
        max_value: Maximum value for range tests.
    """

    id: str
    type: str
    metric: str
    severity: str = "medium"
    title: str = ""
    description: str = ""
    operator: str | None = None
    value: float | int | str | None = None
    condition: str | None = None
    max_percentage: float | None = None
    min_percentage: float | None = None
    min_value: float | None = None
    max_value: float | None = None

    def to_test_config(self) -> TestConfig:
        """Convert to TestConfig for the test engine.

        Returns:
            TestConfig instance.
        """
        return TestConfig(
            id=self.id,
            type=TestType(self.type),
            metric=self.metric,
            severity=TestSeverity(self.severity),
            title=self.title,
            description=self.description,
            operator=self.operator,
            value=self.value,
            condition=self.condition,
            max_percentage=self.max_percentage,
            min_percentage=self.min_percentage,
            min_value=self.min_value,
            max_value=self.max_value,
        )


@dataclass
class TypedAnalyzeConfig:
    """Configuration for the typed analyzer pipeline.

    This is the main configuration class for the new typed analyzer
    architecture. It supports both programmatic construction and
    loading from YAML files.

    Example YAML:
        ```yaml
        dataset_path: /path/to/data.jsonl
        sample_count: 1000
        output_path: ./analysis_output

        analyzers:
          - id: length
            params:
              count_tokens: true
          - id: quality

        custom_metrics:
          - id: turn_pattern
            scope: conversation
            function: |
              def compute(conversation):
                  ...

        tests:
          - id: max_words
            type: threshold
            metric: LengthAnalyzer.total_words
            operator: ">"
            value: 10000
            max_percentage: 5.0
        ```

    Attributes:
        dataset_name: Name of the dataset (HuggingFace identifier).
        dataset_path: Path to local dataset file.
        split: Dataset split to use.
        sample_count: Number of samples to analyze.
        output_path: Directory for output artifacts.
        analyzers: List of analyzer configurations.
        custom_metrics: List of custom metric configurations.
        tests: List of test configurations.
        tokenizer_name: Tokenizer for token counting.
        generate_report: Whether to generate HTML report.
        report_title: Custom title for the report.
    """

    # Dataset source
    dataset_name: str | None = None
    dataset_path: str | None = None
    split: str = "train"
    subset: str | None = None
    sample_count: int | None = None

    # Output
    output_path: str = "."

    # Analyzers
    analyzers: list[AnalyzerConfig] = field(default_factory=list)

    # Custom metrics
    custom_metrics: list[CustomMetricConfig] = field(default_factory=list)

    # Tests
    tests: list[TestConfigYAML] = field(default_factory=list)

    # Tokenizer
    tokenizer_name: str | None = None
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)

    # Report
    generate_report: bool = False
    report_title: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TypedAnalyzeConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            TypedAnalyzeConfig instance.
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedAnalyzeConfig":
        """Create configuration from a dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            TypedAnalyzeConfig instance.
        """
        # Parse analyzers
        analyzers = []
        for analyzer_data in data.get("analyzers", []):
            if isinstance(analyzer_data, dict):
                analyzers.append(AnalyzerConfig(**analyzer_data))
            elif isinstance(analyzer_data, str):
                analyzers.append(AnalyzerConfig(id=analyzer_data))

        # Parse custom metrics
        custom_metrics = []
        for metric_data in data.get("custom_metrics", []):
            # Parse output_schema if present
            output_schema = []
            for field_data in metric_data.pop("output_schema", []):
                if isinstance(field_data, dict):
                    output_schema.append(OutputFieldSchema(**field_data))
            custom_metrics.append(
                CustomMetricConfig(**metric_data, output_schema=output_schema)
            )

        # Parse tests
        tests = []
        for test_data in data.get("tests", []):
            tests.append(TestConfigYAML(**test_data))

        return cls(
            dataset_name=data.get("dataset_name"),
            dataset_path=data.get("dataset_path"),
            split=data.get("split", "train"),
            subset=data.get("subset"),
            sample_count=data.get("sample_count"),
            output_path=data.get("output_path", "."),
            analyzers=analyzers,
            custom_metrics=custom_metrics,
            tests=tests,
            tokenizer_name=data.get("tokenizer_name"),
            tokenizer_kwargs=data.get("tokenizer_kwargs", {}),
            generate_report=data.get("generate_report", False),
            report_title=data.get("report_title"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "split": self.split,
            "subset": self.subset,
            "sample_count": self.sample_count,
            "output_path": self.output_path,
            "analyzers": [
                {"id": a.id, "instance_id": a.instance_id, "params": a.params}
                for a in self.analyzers
            ],
            "custom_metrics": [
                {
                    "id": m.id,
                    "scope": m.scope,
                    "function": m.function,
                    "description": m.description,
                }
                for m in self.custom_metrics
            ],
            "tests": [
                {
                    "id": t.id,
                    "type": t.type,
                    "metric": t.metric,
                    "severity": t.severity,
                    "title": t.title,
                    "description": t.description,
                    "operator": t.operator,
                    "value": t.value,
                    "condition": t.condition,
                    "max_percentage": t.max_percentage,
                    "min_percentage": t.min_percentage,
                    "min_value": t.min_value,
                    "max_value": t.max_value,
                }
                for t in self.tests
            ],
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_kwargs": self.tokenizer_kwargs,
            "generate_report": self.generate_report,
            "report_title": self.report_title,
        }

    def get_test_configs(self) -> list[TestConfig]:
        """Get test configurations for the test engine.

        Returns:
            List of TestConfig instances.
        """
        return [t.to_test_config() for t in self.tests]
