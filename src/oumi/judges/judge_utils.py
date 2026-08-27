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

"""Helpers shared by the LLM-based judges (SimpleJudge, RubricJudge)."""

from oumi.core.configs.params.judge_params import JudgeOutputType

# Judgment options: describing to the judge how to format its judgment.
JUDGMENT_OPTIONS_BOOL = "Your judgment should be a single word: 'Yes' or 'No'"
JUDGMENT_OPTIONS_INT = "Your judgment should be an integer value"
JUDGMENT_OPTIONS_FLOAT = "Your judgment should be a float value"
JUDGMENT_OPTIONS_ENUM_PREFIX = "Your judgment should be one of the following options: "
JUDGMENT_OPTIONS_TEXT = "Your judgment should be provided in the form of free text"


def describe_judgment_options(
    judgment_type: JudgeOutputType,
    judgment_scores: dict[str, float] | None,
) -> str:
    """Describe to the judge the values its judgment is allowed to take.

    Args:
        judgment_type: The expected type of the judgment.
        judgment_scores: Optional mapping from categorical values to numeric scores.
            When it holds more than one entry, its keys are the allowed values,
            regardless of `judgment_type`.

    Returns:
        A sentence (ending with ". ") to embed in the judge's prompt, or an empty
        string if the type carries no formatting guidance.
    """
    if judgment_scores and len(judgment_scores) > 1:
        choices_str = ", ".join(f"'{choice}'" for choice in judgment_scores.keys())
        return f"{JUDGMENT_OPTIONS_ENUM_PREFIX}{choices_str}. "
    elif judgment_type == JudgeOutputType.BOOL:
        return f"{JUDGMENT_OPTIONS_BOOL}. "
    elif judgment_type == JudgeOutputType.FLOAT:
        return f"{JUDGMENT_OPTIONS_FLOAT}. "
    elif judgment_type == JudgeOutputType.INT:
        return f"{JUDGMENT_OPTIONS_INT}. "
    elif judgment_type == JudgeOutputType.TEXT:
        return f"{JUDGMENT_OPTIONS_TEXT}. "
    return ""


def build_judgment_field_schema(
    judgment_type: JudgeOutputType,
    judgment_scores: dict[str, float] | None,
) -> dict:
    """Build the JSON schema fragment for a single judgment field.

    Args:
        judgment_type: The expected type of the judgment.
        judgment_scores: Optional mapping from categorical values to numeric scores.
            When provided, its keys become the schema's enum values.

    Returns:
        A JSON schema fragment describing the judgment value.
    """
    if judgment_scores:
        # Use the user-provided categorical values as the enum, if provided.
        # Note that these are always set for ENUM, optional for other types.
        return {"type": "string", "enum": list(judgment_scores.keys())}
    elif judgment_type == JudgeOutputType.BOOL:
        # Booleans are hardcoded to Yes/No (see JUDGMENT_OPTIONS_BOOL).
        return {"type": "string", "enum": ["Yes", "No"]}
    elif judgment_type == JudgeOutputType.INT:
        return {"type": "integer"}
    elif judgment_type == JudgeOutputType.FLOAT:
        return {"type": "number"}
    return {"type": "string"}
