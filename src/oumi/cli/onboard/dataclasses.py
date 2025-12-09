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

"""Dataclasses and constants for the onboard wizard."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# Task types for synthesis
TASK_TYPES = {
    "extraction": {
        "name": "Structured Extraction",
        "description": "Extract specific information into a structured format (JSON, fields, entities)",
        "output_instruction": "Generate the extracted structured data in the expected format",
    },
    "classification": {
        "name": "Classification",
        "description": "Classify or categorize input into predefined labels/categories",
        "output_instruction": "Provide the classification label with a brief explanation",
    },
    "generation": {
        "name": "Text Generation",
        "description": "Generate new text content (summaries, responses, creative writing)",
        "output_instruction": "Generate the expected text output",
    },
    "transformation": {
        "name": "Transformation",
        "description": "Transform input from one format/style to another",
        "output_instruction": "Generate the transformed output",
    },
    "qa": {
        "name": "Question Answering",
        "description": "Answer questions based on provided context or knowledge",
        "output_instruction": "Provide a clear, accurate answer to the question",
    },
}

# Input format types
INPUT_FORMATS = {
    "single_turn": "Single question/answer",
    "multi_turn": "Multi-turn conversation",
    "document": "Document/text processing",
    "structured": "Structured data (JSON/records)",
}

# Supported file extensions
SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".xlsx", ".xls", ".docx", ".doc"}
TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl"}

# Goal and type choices for CLI
GOAL_CHOICES = ["synth", "judge", "train", "pipeline"]
SYNTH_GOAL_CHOICES = ["qa", "conversation", "augmentation", "instruction"]
JUDGE_TYPE_CHOICES = ["generic", "compliance", "relevance", "safety", "groundedness"]


@dataclass
class TaskSpec:
    """Task specification - what the model should do."""

    description: str = ""
    """What the model should do (free-form description from user)."""

    system_prompt: str = ""
    """The complete system prompt (auto-generated)."""

    task_type: str = "generation"
    """Type of task: extraction, classification, generation, transformation, qa."""


@dataclass
class InputSpec:
    """Input specification - detected from data."""

    format: str = "single_turn"
    """Detected format: single_turn, multi_turn, document, structured."""

    samples: list = field(default_factory=list)
    """Sample inputs from the data (auto-detected)."""

    source_column: str = ""
    """Column to use for inputs."""


@dataclass
class OutputSpec:
    """Output specification - quality criteria."""

    criteria: list = field(default_factory=list)
    """What makes a good response (user-provided list)."""


@dataclass
class WizardState:
    """Wizard state with serialization for caching."""

    task: TaskSpec = field(default_factory=TaskSpec)
    inputs: InputSpec = field(default_factory=InputSpec)
    outputs: OutputSpec = field(default_factory=OutputSpec)

    # File analysis results
    files: list = field(default_factory=list)
    primary_schema: Any = None

    # LLM analyzer (required, not serialized)
    llm_analyzer: Any = None
    domain_analysis: Any = None

    # Extracted use case from customer documents (if found)
    extracted_use_case: Any = None

    # Track completed steps for caching
    completed_steps: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize state to a dictionary for caching."""
        files_data = []
        for f in self.files:
            file_path = f.get("path")
            file_data = {
                "path": str(file_path) if file_path else "",
                "name": f.get("name", ""),
                "extension": f.get("extension", ""),
                "suggested_purpose": f.get("suggested_purpose", ""),
                "suggested_role": f.get("suggested_role", ""),
                "role_reason": f.get("role_reason", ""),
                "content_hash": f.get("content_hash", ""),
                "schema_cache": _serialize_schema(f.get("schema")),
            }
            files_data.append(file_data)

        domain_data = None
        if self.domain_analysis:
            domain_data = {
                "domain": self.domain_analysis.domain,
                "description": self.domain_analysis.description,
                "terminology": self.domain_analysis.terminology,
                "quality_signals": self.domain_analysis.quality_signals,
                "common_issues": self.domain_analysis.common_issues,
                "suggested_persona": self.domain_analysis.suggested_persona,
                "data_purpose": self.domain_analysis.data_purpose,
            }

        return {
            "task": {
                "description": self.task.description,
                "system_prompt": self.task.system_prompt,
                "task_type": self.task.task_type,
            },
            "inputs": {
                "format": self.inputs.format,
                "samples": self.inputs.samples,
                "source_column": self.inputs.source_column,
            },
            "outputs": {
                "criteria": self.outputs.criteria,
            },
            "files": files_data,
            "domain_analysis": domain_data,
            "completed_steps": self.completed_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WizardState":
        """Deserialize state from a dictionary."""
        state = cls()

        task_data = data.get("task", {})
        state.task = TaskSpec(
            description=task_data.get("description", ""),
            system_prompt=task_data.get("system_prompt", ""),
            task_type=task_data.get("task_type", "generation"),
        )

        inputs_data = data.get("inputs", {})
        state.inputs = InputSpec(
            format=inputs_data.get("format", "single_turn"),
            samples=inputs_data.get("samples", []),
            source_column=inputs_data.get("source_column", ""),
        )

        outputs_data = data.get("outputs", {})
        state.outputs = OutputSpec(
            criteria=outputs_data.get("criteria", []),
        )

        state.files = [
            {
                "path": Path(f.get("path", "")) if f.get("path") else None,
                "name": f.get("name", ""),
                "extension": f.get("extension", ""),
                "suggested_purpose": f.get("suggested_purpose", ""),
                "suggested_role": f.get("suggested_role", ""),
                "role_reason": f.get("role_reason", ""),
                "content_hash": f.get("content_hash", ""),
                "schema_cache": _deserialize_schema(f.get("schema_cache")),
            }
            for f in data.get("files", [])
        ]

        domain_data = data.get("domain_analysis")
        if domain_data:
            from oumi.onboarding.llm_analyzer import DomainAnalysis

            state.domain_analysis = DomainAnalysis(
                domain=domain_data.get("domain", "unknown"),
                description=domain_data.get("description", ""),
                terminology=domain_data.get("terminology", []),
                quality_signals=domain_data.get("quality_signals", []),
                common_issues=domain_data.get("common_issues", []),
                suggested_persona=domain_data.get("suggested_persona", ""),
                data_purpose=domain_data.get("data_purpose", ""),
            )

        state.completed_steps = data.get("completed_steps", [])

        return state


def _convert_to_native(obj):
    """Convert numpy/pandas types to native Python types for YAML serialization."""
    if obj is None:
        return None
    # Handle numpy types
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_native(item) for item in obj]
    # Handle pandas Timestamp
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _serialize_schema(schema) -> dict | None:
    """Serialize a DataSchema to a dictionary for caching.

    Args:
        schema: DataSchema instance or None.

    Returns:
        Dictionary representation of the schema, or None.
    """
    if schema is None:
        return None

    columns_data = []
    for col in schema.columns:
        columns_data.append({
            "name": col.name,
            "dtype": col.dtype,
            "sample_values": _convert_to_native(col.sample_values),
            "unique_count": _convert_to_native(col.unique_count),
            "null_count": _convert_to_native(col.null_count),
            "avg_length": _convert_to_native(col.avg_length),
            "is_conversation": col.is_conversation,
            "is_text": col.is_text,
            "is_categorical": col.is_categorical,
        })

    return {
        "columns": columns_data,
        "row_count": _convert_to_native(schema.row_count),
        "sample_rows": _convert_to_native(schema.sample_rows),
        "detected_format": schema.detected_format,
        "source_path": schema.source_path,
        "conversation_columns": schema.conversation_columns,
        "text_columns": schema.text_columns,
        "categorical_columns": schema.categorical_columns,
        "raw_text": schema.raw_text,
    }


def _deserialize_schema(data: dict | None):
    """Deserialize a DataSchema from a dictionary.

    Args:
        data: Dictionary representation of a schema, or None.

    Returns:
        DataSchema instance or None.
    """
    if data is None:
        return None

    from oumi.onboarding.data_analyzer import ColumnInfo, DataSchema

    columns = []
    for col_data in data.get("columns", []):
        columns.append(ColumnInfo(
            name=col_data.get("name", ""),
            dtype=col_data.get("dtype", ""),
            sample_values=col_data.get("sample_values", []),
            unique_count=col_data.get("unique_count", 0),
            null_count=col_data.get("null_count", 0),
            avg_length=col_data.get("avg_length"),
            is_conversation=col_data.get("is_conversation", False),
            is_text=col_data.get("is_text", False),
            is_categorical=col_data.get("is_categorical", False),
        ))

    return DataSchema(
        columns=columns,
        row_count=data.get("row_count", 0),
        sample_rows=data.get("sample_rows", []),
        detected_format=data.get("detected_format", "unknown"),
        source_path=data.get("source_path", ""),
        conversation_columns=data.get("conversation_columns", []),
        text_columns=data.get("text_columns", []),
        categorical_columns=data.get("categorical_columns", []),
        raw_text=data.get("raw_text"),
    )
