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
    TURN_STATS = "turn_stats"


@dataclass
class AnalyzerConfig:
    """Configuration for a single analyzer.

    Matches the API's ``AnalyzerConfigInput`` naming convention:

    - ``type``: registry identifier (e.g. ``"length"``, ``"quality"``).
    - ``display_name``: human-readable label shown in the UI and used as
      the results key / metric prefix.  Defaults to ``type``.
    - ``params``: analyzer-specific parameters forwarded to the factory.

    Attributes:
        type: Analyzer type identifier (registry key).
        display_name: Human-readable label used as the result key and
            metric path prefix.  Defaults to ``type`` if not set.
        params: Analyzer-specific parameters.
    """

    type: str = ""
    display_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-populate display_name if not provided."""
        if self.display_name is None:
            self.display_name = self.type


@dataclass
class TestConfigYAML:
    """YAML-friendly test configuration.

    This class mirrors TestConfig but uses simpler types for YAML parsing.

    Attributes:
        id: Unique identifier for the test.
        type: Test type ("threshold", "percentage", "range").
        metric: Path to the metric (e.g., ``"Length.total_tokens"``).
        severity: Severity level ("high", "medium", "low").
        display_name: Human-readable title shown in results.  Alias: ``title``.
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
    display_name: str = ""
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
            title=self.display_name,
            description=self.description,
            operator=self.operator,
            value=self.value,
            max_percentage=self.max_percentage,
            min_percentage=self.min_percentage,
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
          - type: length
            display_name: Length
            params:
              tokenizer_name: cl100k_base
          - type: quality
            display_name: Quality

        tests:
          - id: max_tokens
            type: threshold
            metric: Length.total_tokens
            operator: ">"
            value: 10000
            max_percentage: 5.0
            display_name: "Token count exceeds limit"
        ```

    Attributes:
        dataset_name: Name of the dataset (HuggingFace identifier).
        dataset_path: Path to local dataset file.
        split: Dataset split to use.
        sample_count: Number of samples to analyze.
        output_path: Directory for output artifacts.
        analyzers: List of analyzer configurations.
        tests: List of test configurations.
        tokenizer_name: Tokenizer for token counting.
        generate_report: Whether to generate HTML report.
        report_title: Custom title for the report.
    """

    # Eval name (optional, for web viewer)
    eval_name: str | None = None

    # Parent eval ID (for linking derived analyses)
    parent_eval_id: str | None = None

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
                analyzer_data = dict(analyzer_data)  # don't mutate caller's dict
                # Backward compat: accept "id" as alias for "type"
                if "id" in analyzer_data and "type" not in analyzer_data:
                    analyzer_data["type"] = analyzer_data.pop("id")
                elif "id" in analyzer_data:
                    analyzer_data.pop("id")  # "type" takes precedence
                # Backward compat: accept "instance_id" as alias for "display_name"
                if (
                    "instance_id" in analyzer_data
                    and "display_name" not in analyzer_data
                ):
                    analyzer_data["display_name"] = analyzer_data.pop("instance_id")
                elif "instance_id" in analyzer_data:
                    analyzer_data.pop("instance_id")  # "display_name" takes precedence
                try:
                    analyzers.append(AnalyzerConfig(**analyzer_data))
                except TypeError as e:
                    raise ValueError(
                        f"Invalid analyzer config: {e}. "
                        f"Valid fields: type, display_name, params"
                    ) from None
            elif isinstance(analyzer_data, str):
                analyzers.append(AnalyzerConfig(type=analyzer_data))

        # Validate unique display_names
        display_names = [a.display_name for a in analyzers]
        duplicates = [
            name for name in set(display_names) if display_names.count(name) > 1
        ]
        if duplicates:
            raise ValueError(
                f"Duplicate analyzer display_name values: {duplicates}. "
                "Each analyzer must have a unique display_name to avoid "
                "result collisions."
            )

        # Parse tests
        tests = []
        for test_data in data.get("tests", []):
            test_data = dict(test_data)  # don't mutate caller's dict
            # Backward compat: accept "title" as alias for "display_name"
            if "title" in test_data and "display_name" not in test_data:
                test_data["display_name"] = test_data.pop("title")
            elif "title" in test_data:
                test_data.pop("title")  # "display_name" takes precedence
            try:
                tests.append(TestConfigYAML(**test_data))
            except TypeError as e:
                raise ValueError(
                    f"Invalid test config: {e}. "
                    f"Valid fields: id, type, metric, severity, display_name, "
                    f"description, operator, value, condition, "
                    f"max_percentage, min_percentage, min_value, max_value"
                ) from None

        return cls(
            eval_name=data.get("eval_name"),
            parent_eval_id=data.get("parent_eval_id"),
            dataset_name=data.get("dataset_name"),
            dataset_path=data.get("dataset_path"),
            split=data.get("split", "train"),
            subset=data.get("subset"),
            sample_count=data.get("sample_count"),
            output_path=data.get("output_path", "."),
            analyzers=analyzers,
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
                {
                    "type": a.type,
                    "display_name": a.display_name,
                    "params": a.params,
                }
                for a in self.analyzers
            ],
            "tests": [
                {
                    "id": t.id,
                    "type": t.type,
                    "metric": t.metric,
                    "severity": t.severity,
                    "display_name": t.display_name,
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
