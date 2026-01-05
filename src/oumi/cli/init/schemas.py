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

"""Pydantic schemas for oumi init structured outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    """Output format for generated data."""

    CONVERSATION = "conversation"
    INSTRUCTION = "instruction"
    RAW = "raw"


class AttributeType(str, Enum):
    """Type of attribute in the generation pipeline."""

    SAMPLED = "sampled"
    GENERATED = "generated"
    TRANSFORMED = "transformed"


# ============================================================================
# Conversation Phase
# ============================================================================


class TaskUnderstanding(BaseModel):
    """LLM's understanding of the user's task."""

    summary: str = Field(description="1-2 sentence summary")
    task_type: str = Field(
        description="qa_generation, augmentation, conversation, extraction, "
        "code, domain_specific"
    )
    input_sources: list[str] = Field(description="Types: documents, datasets, none")
    output_format: OutputFormat
    key_requirements: list[str] = Field(description="Critical requirements")
    confidence: Literal["high", "medium", "low"]
    unsupported_requests: list[str] = Field(default_factory=list)
    suggested_workarounds: list[str] = Field(default_factory=list)


class FollowUpQuestion(BaseModel):
    """A follow-up question for the user."""

    question: str
    question_type: Literal["multiple_choice", "free_text"]
    options: list[dict[str, str]] | None = Field(
        default=None, description="List of {label, description} for multiple_choice"
    )
    why_needed: str


class ConversationResponse(BaseModel):
    """Complete LLM response during conversation."""

    understanding: TaskUnderstanding
    follow_up_questions: list[FollowUpQuestion] = Field(default_factory=list)
    ready_to_generate: bool


# ============================================================================
# Config Generation Phase
# ============================================================================


class SampledValue(BaseModel):
    """A possible value for a sampled attribute."""

    id: str
    name: str
    description: str
    sample_rate: float | None = None


class SampledAttribute(BaseModel):
    """A sampled attribute definition."""

    id: str
    name: str
    description: str
    possible_values: list[SampledValue]


class GeneratedAttribute(BaseModel):
    """A generated attribute definition."""

    id: str
    depends_on: list[str] = Field(default_factory=list)
    system_prompt: str
    user_prompt_template: str
    postprocessing: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    """A message in chat transformation."""

    role: Literal["USER", "ASSISTANT", "SYSTEM"]
    content: str


class TransformedAttribute(BaseModel):
    """A transformed attribute definition."""

    id: str
    transformation_type: Literal["STRING", "CHAT"]
    string_template: str | None = None
    chat_messages: list[ChatMessage] | None = None


class InputDocument(BaseModel):
    """Input document configuration."""

    path: str
    id: str
    segment_length: int = 2048
    segment_overlap: int = 256


class InputDataset(BaseModel):
    """Input dataset configuration."""

    path: str
    attribute_map: dict[str, str] = Field(default_factory=dict)


class SynthConfigSpec(BaseModel):
    """Complete synth config specification."""

    input_documents: list[InputDocument] = Field(default_factory=list)
    input_data: list[InputDataset] = Field(default_factory=list)
    sampled_attributes: list[SampledAttribute] = Field(default_factory=list)
    generated_attributes: list[GeneratedAttribute] = Field(default_factory=list)
    transformed_attributes: list[TransformedAttribute] = Field(default_factory=list)
    output_path: str
    num_samples: int = 100
    passthrough_attributes: list[str] = Field(default_factory=list)


class JudgeConfigSpec(BaseModel):
    """Complete judge config specification."""

    system_instruction: str
    prompt_template: str
    judgment_type: Literal["BOOL", "ENUM", "INT", "FLOAT", "TEXT"] = "BOOL"
    judgment_scores: dict[str, float] | None = None
    include_explanation: bool = True


class ConfigGenerationResponse(BaseModel):
    """Combined response for config generation."""

    synth_config: SynthConfigSpec
    judge_config: JudgeConfigSpec


# ============================================================================
# Session State (for resume support)
# ============================================================================


class SessionState(BaseModel):
    """Saved session state for resume support."""

    task: str
    sources: list[str]
    output_format: OutputFormat
    output_dir: str
    source_analyses: list[dict[str, Any]] = Field(default_factory=list)
    understanding: dict[str, Any] | None = None
    synth_yaml: str | None = None
    judge_yaml: str | None = None
    phase: Literal["conversation", "generation", "review"] = "conversation"


# ============================================================================
# Meta-Judge Validation
# ============================================================================


class MetaJudgeResult(BaseModel):
    """Result from meta-judge validation."""

    is_coherent: bool
    issues: list[str] = Field(default_factory=list)
    judge_synth_aligned: bool
    judge_synth_aligned_reason: str
    attribute_references_valid: bool
    attribute_references_reason: str
    pipeline_logic_sound: bool
    pipeline_logic_reason: str
    prompts_well_formed: bool
    prompts_reason: str


# ============================================================================
# Edit Loop
# ============================================================================


class EditResponse(BaseModel):
    """Response from edit request."""

    changes_summary: str
    modified_sections: list[str]
    updated_synth_config: SynthConfigSpec | None = None
    updated_judge_config: JudgeConfigSpec | None = None
