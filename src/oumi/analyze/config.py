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

"""Configuration for the typed analyzer framework."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from oumi.core.configs.params.test_params import TestParams


@dataclass
class AnalyzerConfig:
    """Configuration for a single analyzer instance.

    Three identity-related fields:

    * ``type`` — registry id (e.g. ``"length"``). Picks the analyzer class.
    * ``id`` — stable identity. Canonical key for results, caches, and test
      metric paths. Defaults to ``display_name`` when omitted.
    * ``display_name`` — human-readable label for UI and logs. Defaults to
      ``type`` when omitted. May repeat across analyzers.

    When callers don't set ``id`` or ``display_name``, all three collapse to
    ``type`` and today's behavior is preserved. When the API populates ``id``
    with a generated asset id, ``display_name`` becomes purely cosmetic.

    Attributes:
        type: Analyzer type (registry id, e.g. "length", "difficulty_judge").
        id: Stable identifier. Used as the results key and in test metric
            paths. Defaults to ``display_name`` when omitted.
        display_name: Human-readable label. Defaults to ``type`` when omitted.
        params: Analyzer-specific parameters.
    """

    type: str = ""
    id: str = ""
    display_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and default the analyzer configuration."""
        if not self.type:
            raise ValueError("AnalyzerConfig.type is required.")
        if not self.display_name:
            self.display_name = self.type
        if not self.id:
            self.id = self.display_name


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

    .. warning::
        **Security Warning**: The ``function`` field contains arbitrary Python
        code that is executed dynamically. Only load configurations from
        trusted sources. Never load YAML configs from untrusted users or
        external sources without review, as they could execute malicious code.

    Example YAML::

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

    Attributes:
        id: Unique identifier for the metric.
        scope: Scope of the metric ("message", "conversation", or "dataset").
        function: Python code defining a compute() function.
        description: Description of what the metric computes.
        output_schema: List of output field definitions.
    """

    id: str
    scope: str = "conversation"  # "message", "conversation", or "dataset"
    function: str = ""
    description: str | None = None
    output_schema: list[OutputFieldSchema] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate the configuration."""
        if self.scope not in ("message", "conversation", "dataset"):
            raise ValueError(
                f"Invalid scope '{self.scope}'. "
                "Must be 'message', 'conversation', or 'dataset'."
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
class TypedAnalyzeConfig:
    """Configuration for the typed analyzer pipeline.

    This is the main configuration class for the new typed analyzer
    architecture. It supports both programmatic construction and
    loading from YAML files.

    Example YAML::

        dataset_path: /path/to/data.jsonl
        sample_count: 1000
        output_path: ./analysis_output

        analyzers:
          - type: length
            params:
              count_tokens: true
          - type: quality

        custom_metrics:
          - id: turn_pattern
            scope: conversation
            function: |
              def compute(conversation):
                  ...

        tests:
          - id: max_words
            type: threshold
            metric: length.total_words
            operator: ">"
            value: 10000
            max_percentage: 5.0

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

    eval_name: str | None = None
    parent_eval_id: str | None = None
    dataset_name: str | None = None
    dataset_path: str | None = None
    split: str = "train"
    subset: str | None = None
    sample_count: int | None = None
    output_path: str | None = None
    analyzers: list[AnalyzerConfig] = field(default_factory=list)
    custom_metrics: list[CustomMetricConfig] = field(default_factory=list)
    tests: list[TestParams] = field(default_factory=list)
    tokenizer_name: str | None = None
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    generate_report: bool = False
    report_title: str | None = None

    @classmethod
    def from_yaml(
        cls, path: str | Path, allow_custom_code: bool = False
    ) -> "TypedAnalyzeConfig":
        """Load configuration from a YAML file.

        .. warning::
            **Security Warning**: If the YAML file contains ``custom_metrics``
            with ``function`` fields, arbitrary Python code will be loaded.
            Only load configurations from trusted sources. Set
            ``allow_custom_code=True`` to explicitly acknowledge this risk.

        Args:
            path: Path to YAML configuration file.
            allow_custom_code: If True, allow loading custom_metrics with
                function code. If False (default) and the config contains
                custom metrics with code, raises ValueError.

        Returns:
            TypedAnalyzeConfig instance.

        Raises:
            ValueError: If config contains custom code but allow_custom_code=False.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data, allow_custom_code=allow_custom_code)

    @classmethod
    def _parse_analyzers(cls, data: dict[str, Any]) -> list[AnalyzerConfig]:
        """Parse analyzer configurations, raising on duplicate ids.

        Accepts the legacy ``instance_id`` key as an alias for ``display_name``
        (with a ``DeprecationWarning``) for one release.
        """
        analyzers = []
        for analyzer_data in data.get("analyzers", []):
            if isinstance(analyzer_data, str):
                analyzers.append(AnalyzerConfig(type=analyzer_data))
                continue
            if not isinstance(analyzer_data, dict):
                continue

            normalized = dict(analyzer_data)
            if "instance_id" in normalized:
                warnings.warn(
                    "'instance_id' is deprecated; rename to 'display_name'.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                legacy = normalized.pop("instance_id")
                normalized.setdefault("display_name", legacy)
            if "type" not in normalized and "id" in normalized:
                raise ValueError(
                    "Legacy analyzer entry: 'id' is present without 'type'. "
                    "The 'id' field now means stable identity, not analyzer "
                    "type. Rename 'id' to 'type' (and 'instance_id' to "
                    "'display_name', if present). See "
                    "docs/user_guides/analyze/analyze_config.md."
                )
            analyzers.append(AnalyzerConfig(**normalized))

        ids = [a.id for a in analyzers]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate analyzer id values: {duplicates}. "
                "Each analyzer must have a unique id to avoid collisions."
            )

        return analyzers

    @classmethod
    def _parse_custom_metrics(
        cls, data: dict[str, Any], allow_custom_code: bool
    ) -> list[CustomMetricConfig]:
        """Parse custom metrics, raising if code is present and not allowed."""
        custom_metrics = []
        for metric_data in data.get("custom_metrics", []):
            output_schema = [
                OutputFieldSchema(**f)
                for f in metric_data.get("output_schema", [])
                if isinstance(f, dict)
            ]
            remaining = {k: v for k, v in metric_data.items() if k != "output_schema"}
            custom_metrics.append(
                CustomMetricConfig(**remaining, output_schema=output_schema)
            )

        # Security check: reject custom code unless explicitly allowed
        if not allow_custom_code:
            metrics_with_code = [m.id for m in custom_metrics if m.function.strip()]
            if metrics_with_code:
                raise ValueError(
                    f"Configuration contains custom metrics with executable code: "
                    f"{metrics_with_code}. This is a security risk if loading from "
                    f"untrusted sources. Set allow_custom_code=True to explicitly "
                    f"allow code execution, or remove the 'function' fields."
                )

        return custom_metrics

    @classmethod
    def _parse_tests(cls, data: dict[str, Any]) -> list[TestParams]:
        """Parse and validate test configurations."""
        tests = []
        for test_data in data.get("tests", []):
            test_params = TestParams(**test_data)
            test_params.finalize_and_validate()
            tests.append(test_params)
        return tests

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], allow_custom_code: bool = False
    ) -> "TypedAnalyzeConfig":
        """Create configuration from a dictionary.

        Args:
            data: Configuration dictionary.
            allow_custom_code: If True, allow custom_metrics with function code.
                If False (default) and the config contains custom metrics with
                code, raises ValueError.

        Returns:
            TypedAnalyzeConfig instance.

        Raises:
            ValueError: If config contains custom code but allow_custom_code=False,
                or if duplicate analyzer id values are found.
        """
        analyzers = cls._parse_analyzers(data)
        custom_metrics = cls._parse_custom_metrics(data, allow_custom_code)
        tests = cls._parse_tests(data)

        return cls(
            eval_name=data.get("eval_name"),
            parent_eval_id=data.get("parent_eval_id"),
            dataset_name=data.get("dataset_name"),
            dataset_path=data.get("dataset_path"),
            split=data.get("split", "train"),
            subset=data.get("subset"),
            sample_count=data.get("sample_count"),
            output_path=data.get("output_path"),
            analyzers=analyzers,
            custom_metrics=custom_metrics,
            tests=tests,
            tokenizer_name=data.get("tokenizer_name"),
            tokenizer_kwargs=data.get("tokenizer_kwargs", {}),
            generate_report=data.get("generate_report", False),
            report_title=data.get("report_title"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "eval_name": self.eval_name,
            "parent_eval_id": self.parent_eval_id,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "split": self.split,
            "subset": self.subset,
            "sample_count": self.sample_count,
            "output_path": self.output_path,
            "analyzers": [
                {
                    "type": a.type,
                    "id": a.id,
                    "display_name": a.display_name,
                    "params": a.params,
                }
                for a in self.analyzers
            ],
            "custom_metrics": [
                {
                    "id": m.id,
                    "scope": m.scope,
                    "function": m.function,
                    "description": m.description,
                    "output_schema": [
                        {
                            "name": f.name,
                            "type": f.type,
                            "description": f.description,
                        }
                        for f in m.output_schema
                    ],
                    "depends_on": m.depends_on,
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
                }
                for t in self.tests
            ],
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_kwargs": self.tokenizer_kwargs,
            "generate_report": self.generate_report,
            "report_title": self.report_title,
        }
