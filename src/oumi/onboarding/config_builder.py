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

"""Configuration builders for customer onboarding.

This module provides builders that generate Oumi configurations from
analyzed customer data schemas.
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from oumi.utils.logging import logger

# Supported file types for DatasetSource
SUPPORTED_DATASET_EXTENSIONS = {".csv", ".json", ".jsonl", ".tsv", ".parquet"}

from oumi.core.configs import (
    InferenceConfig,
    JudgeConfig,
    SynthesisConfig,
    TrainingConfig,
)
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.judge_params import (
    JudgeOutputType,
    JudgeParams,
    JudgeResponseFormat,
)
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.synthesis_params import (
    DatasetSource,
    GeneralSynthesisParams,
    GeneratedAttribute,
    GeneratedAttributePostprocessingParams,
    SampledAttribute,
    SampledAttributeValue,
    TextConversation,
    TextMessage,
    TransformationStrategy,
    TransformationType,
    TransformedAttribute,
)
from oumi.core.types.conversation import Role
from oumi.onboarding.data_analyzer import DataSchema
from oumi.onboarding.field_mapper import FieldMapper, FieldMapping

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oumi.onboarding.llm_analyzer import DomainAnalysis, InferredConfig, LLMAnalyzer

SynthGoal = Literal["qa", "conversation", "augmentation", "instruction"]
JudgeType = Literal["generic", "compliance", "relevance", "safety", "groundedness"]


def _convert_to_supported_format(file_path: str, output_dir: Optional[str] = None) -> str:
    """Convert unsupported file types to CSV for use with DatasetSource.

    Args:
        file_path: Path to the source file.
        output_dir: Optional output directory. If not provided, uses same directory.

    Returns:
        Path to the converted file (CSV), or original path if already supported.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # Already supported - return as-is
    if ext in SUPPORTED_DATASET_EXTENSIONS:
        return file_path

    # Determine output path
    if output_dir:
        out_path = Path(output_dir) / f"{path.stem}.csv"
    else:
        out_path = path.with_suffix(".csv")

    # Convert Excel files
    if ext in {".xlsx", ".xls"}:
        try:
            import pandas as pd

            df = pd.read_excel(file_path)
            df.to_csv(out_path, index=False)
            logger.info(f"Converted {path.name} to {out_path.name}")
            return str(out_path)
        except ImportError:
            raise ImportError(
                "pandas and openpyxl are required to convert Excel files. "
                "Install with: pip install pandas openpyxl"
            )
        except Exception as e:
            raise ValueError(f"Failed to convert Excel file {file_path}: {e}")

    # Unsupported format
    raise ValueError(
        f"Unsupported file type: {ext}. "
        f"Supported types: {SUPPORTED_DATASET_EXTENSIONS}"
    )


@dataclass
class BuilderOptions:
    """Common options for config builders."""

    model_name: str = "claude-haiku-4-5"
    engine: str = "ANTHROPIC"
    temperature: float = 0.7
    max_new_tokens: int = 8192
    num_workers: int = 50


class ConfigBuilder(ABC):
    """Abstract base class for configuration builders."""

    def __init__(self, options: Optional[BuilderOptions] = None):
        """Initialize the builder with optional configuration.

        Args:
            options: Builder options for model, engine, etc.
        """
        self.options = options or BuilderOptions()
        self.field_mapper = FieldMapper()

    @abstractmethod
    def from_schema(self, schema: DataSchema, **kwargs) -> Any:
        """Build a configuration from an analyzed data schema.

        Args:
            schema: The analyzed data schema.
            **kwargs: Additional configuration options.

        Returns:
            The built configuration object.
        """
        pass

    def _create_inference_config(self) -> InferenceConfig:
        """Create a standard inference configuration."""
        from oumi.core.configs.inference_config import InferenceEngineType
        from oumi.core.configs.params.remote_params import RemoteParams

        engine_map = {
            "ANTHROPIC": InferenceEngineType.ANTHROPIC,
            "OPENAI": InferenceEngineType.OPENAI,
            "VLLM": InferenceEngineType.VLLM,
            "NATIVE": InferenceEngineType.NATIVE,
        }

        return InferenceConfig(
            model=ModelParams(model_name=self.options.model_name),
            engine=engine_map.get(self.options.engine, InferenceEngineType.ANTHROPIC),
            generation=GenerationParams(
                max_new_tokens=self.options.max_new_tokens,
                temperature=self.options.temperature,
                top_p=0.9,
            ),
            remote_params=RemoteParams(
                num_workers=self.options.num_workers,
                politeness_policy=60,
            ),
        )


class SynthConfigBuilder(ConfigBuilder):
    """Build SynthesisConfig from customer data.

    This builder analyzes customer data and generates a synthesis configuration
    that can create synthetic training data based on the patterns found.

    Example:
        >>> from oumi.onboarding import DataAnalyzer, SynthConfigBuilder
        >>> analyzer = DataAnalyzer()
        >>> schema = analyzer.analyze("./customer_data.csv")
        >>> builder = SynthConfigBuilder()
        >>> config = builder.from_schema(schema, goal="qa", num_samples=100)
        >>> config.to_yaml("./synth_config.yaml")
    """

    # Templates for different synthesis goals
    GOAL_TEMPLATES = {
        "qa": {
            "system_prompt": """You are an expert at creating high-quality question-answer pairs.
Create questions that are clear, specific, and educational.
Base your questions on the provided context when available.""",
            "question_prompt": """Based on the following information, generate a thoughtful question:

{context}

Format your response as:
Question: <your question>""",
            "answer_prompt": """Provide a helpful, accurate answer to this question:

{question}

Format your response as:
Answer: <your answer>""",
        },
        "conversation": {
            "system_prompt": """You are an expert at generating realistic conversation exchanges.
Create natural dialogue that could occur in a customer service or professional context.""",
            "generation_prompt": """Generate a realistic conversation based on this context:

{context}

The conversation should be helpful and professional.""",
        },
        "augmentation": {
            "system_prompt": """You are an expert at creating variations of data.
Generate diverse variations that preserve the core meaning while varying style and phrasing.""",
            "generation_prompt": """Create a variation of the following:

{original}

Maintain the core meaning but vary the phrasing and style.""",
        },
        "instruction": {
            "system_prompt": """You are following specific instructions to generate content.
Follow the provided instructions precisely and generate high-quality outputs.""",
            "generation_prompt": """Following these instructions:

{instruction}

Generate an appropriate response.""",
        },
    }

    def from_schema(
        self,
        schema: DataSchema,
        goal: SynthGoal = "qa",
        num_samples: int = 100,
        output_path: Optional[str] = None,
        mappings: Optional[list[FieldMapping]] = None,
        attribute_map: Optional[dict[str, str]] = None,
    ) -> SynthesisConfig:
        """Build a SynthesisConfig from analyzed data.

        Args:
            schema: The analyzed data schema.
            goal: Synthesis goal ("qa", "conversation", "augmentation", "instruction").
            num_samples: Number of samples to generate.
            output_path: Optional output path for generated data.
            mappings: Optional pre-computed field mappings.
            attribute_map: Optional explicit column-to-role mapping (e.g.,
                {"context": "description_col", "question": "query_col"}).
                Takes precedence over auto-detected mappings.

        Returns:
            A configured SynthesisConfig.
        """
        from oumi.core.configs.synthesis_config import SynthesisStrategy

        # Get field mappings if not provided
        if mappings is None:
            mappings = self.field_mapper.suggest_mappings(schema, goal)

        # Build strategy params based on goal
        strategy_params = self._build_strategy_params(
            schema, goal, mappings, attribute_map=attribute_map
        )

        # Create output path if not provided
        if output_path is None:
            output_path = f"synth_{goal}_output.jsonl"

        return SynthesisConfig(
            strategy=SynthesisStrategy.GENERAL,
            num_samples=num_samples,
            output_path=output_path,
            strategy_params=strategy_params,
            inference_config=self._create_inference_config(),
        )

    def _build_strategy_params(
        self,
        schema: DataSchema,
        goal: SynthGoal,
        mappings: list[FieldMapping],
        attribute_map: Optional[dict[str, str]] = None,
    ) -> GeneralSynthesisParams:
        """Build strategy params based on the goal.

        Args:
            schema: The analyzed data schema.
            goal: Synthesis goal.
            mappings: Field mappings from auto-detection.
            attribute_map: Optional explicit column-to-role mapping that takes
                precedence over auto-detected mappings.
        """
        params = GeneralSynthesisParams()

        # Add input data source if schema has data
        if schema.source_path and schema.detected_format in ("csv", "excel", "json", "jsonl"):
            # Use provided attribute_map or build from mappings
            if attribute_map:
                # Explicit column assignments provided - use directly
                final_attribute_map = attribute_map
            else:
                # Build attribute map from auto-detected mappings
                final_attribute_map = self._build_attribute_map(mappings)

                # If no mappings, try to find a good context column
                if not final_attribute_map and schema.columns:
                    # Find the best text column to use as context
                    text_cols = [c for c in schema.columns if c.is_text]
                    if text_cols:
                        # Use the longest text column as context
                        best_col = max(text_cols, key=lambda c: c.avg_length or 0)
                        final_attribute_map = {best_col.name: "context"}
                    elif schema.columns:
                        # Fallback: use first column as context
                        final_attribute_map = {schema.columns[0].name: "context"}

            params.input_data = [
                DatasetSource(
                    path=_convert_to_supported_format(schema.source_path),
                    attribute_map=final_attribute_map if final_attribute_map else None,
                )
            ]

        # Build generated attributes based on goal
        if goal == "qa":
            params.generated_attributes = self._build_qa_attributes(mappings)
            params.transformed_attributes = self._build_qa_transform()
            params.passthrough_attributes = ["conversation", "question", "answer"]
        elif goal == "conversation":
            params.generated_attributes = self._build_conversation_attributes(mappings)
            params.passthrough_attributes = ["conversation"]
        elif goal == "augmentation":
            params.generated_attributes = self._build_augmentation_attributes(mappings)
            params.passthrough_attributes = ["augmented"]
        elif goal == "instruction":
            params.generated_attributes = self._build_instruction_attributes(mappings, schema)
            params.passthrough_attributes = ["output"]

        return params

    def _build_attribute_map(self, mappings: list[FieldMapping]) -> dict[str, str]:
        """Build attribute map from field mappings."""
        return {m.customer_column: m.oumi_placeholder for m in mappings}

    def _build_qa_attributes(
        self, mappings: list[FieldMapping]
    ) -> list[GeneratedAttribute]:
        """Build generated attributes for Q&A synthesis."""
        templates = self.GOAL_TEMPLATES["qa"]

        # Find the context placeholder
        context_placeholder = "context"
        for m in mappings:
            if m.oumi_placeholder == "context":
                context_placeholder = m.oumi_placeholder
                break

        question_attr = GeneratedAttribute(
            id="question_raw",
            instruction_messages=[
                TextMessage(role=Role.SYSTEM, content=templates["system_prompt"]),
                TextMessage(
                    role=Role.USER,
                    content=templates["question_prompt"].replace(
                        "{context}", f"{{{context_placeholder}}}"
                    ),
                ),
            ],
            postprocessing_params=GeneratedAttributePostprocessingParams(
                id="question",
                cut_prefix="Question:",
                strip_whitespace=True,
            ),
        )

        answer_attr = GeneratedAttribute(
            id="answer_raw",
            instruction_messages=[
                TextMessage(role=Role.SYSTEM, content=templates["system_prompt"]),
                TextMessage(
                    role=Role.USER,
                    content=templates["answer_prompt"].replace("{question}", "{question}"),
                ),
            ],
            postprocessing_params=GeneratedAttributePostprocessingParams(
                id="answer",
                cut_prefix="Answer:",
                strip_whitespace=True,
            ),
        )

        return [question_attr, answer_attr]

    def _build_qa_transform(self) -> list[TransformedAttribute]:
        """Build transformation for Q&A to conversation format."""
        return [
            TransformedAttribute(
                id="conversation",
                transformation_strategy=TransformationStrategy(
                    type=TransformationType.CHAT,
                    chat_transform=TextConversation(
                        messages=[
                            TextMessage(role=Role.USER, content="{question}"),
                            TextMessage(role=Role.ASSISTANT, content="{answer}"),
                        ]
                    ),
                ),
            )
        ]

    def _build_conversation_attributes(
        self, mappings: list[FieldMapping]
    ) -> list[GeneratedAttribute]:
        """Build generated attributes for conversation synthesis."""
        templates = self.GOAL_TEMPLATES["conversation"]

        # Find context placeholder
        context_placeholder = "context"
        for m in mappings:
            if m.oumi_placeholder in ("context", "conversation"):
                context_placeholder = m.oumi_placeholder
                break

        return [
            GeneratedAttribute(
                id="conversation",
                instruction_messages=[
                    TextMessage(role=Role.SYSTEM, content=templates["system_prompt"]),
                    TextMessage(
                        role=Role.USER,
                        content=templates["generation_prompt"].replace(
                            "{context}", f"{{{context_placeholder}}}"
                        ),
                    ),
                ],
            )
        ]

    def _build_augmentation_attributes(
        self, mappings: list[FieldMapping]
    ) -> list[GeneratedAttribute]:
        """Build generated attributes for data augmentation."""
        templates = self.GOAL_TEMPLATES["augmentation"]

        # Find the primary text placeholder
        original_placeholder = "context"
        for m in mappings:
            if m.oumi_placeholder in ("context", "question", "answer"):
                original_placeholder = m.oumi_placeholder
                break

        return [
            GeneratedAttribute(
                id="augmented",
                instruction_messages=[
                    TextMessage(role=Role.SYSTEM, content=templates["system_prompt"]),
                    TextMessage(
                        role=Role.USER,
                        content=templates["generation_prompt"].replace(
                            "{original}", f"{{{original_placeholder}}}"
                        ),
                    ),
                ],
            )
        ]

    def _build_instruction_attributes(
        self,
        mappings: list[FieldMapping],
        schema: DataSchema,
    ) -> list[GeneratedAttribute]:
        """Build generated attributes for instruction following."""
        templates = self.GOAL_TEMPLATES["instruction"]

        # Use system instruction from Word doc if available
        instruction = "{system_instruction}"
        if schema.raw_text:
            instruction = schema.raw_text[:2000]  # Truncate long instructions

        return [
            GeneratedAttribute(
                id="output",
                instruction_messages=[
                    TextMessage(role=Role.SYSTEM, content=templates["system_prompt"]),
                    TextMessage(
                        role=Role.USER,
                        content=templates["generation_prompt"].replace(
                            "{instruction}", instruction
                        ),
                    ),
                ],
            )
        ]

    def from_schema_with_inference(
        self,
        schema: DataSchema,
        goal: SynthGoal = "qa",
        num_samples: int = 100,
        output_path: Optional[str] = None,
        domain: Optional["DomainAnalysis"] = None,
        llm_analyzer: Optional["LLMAnalyzer"] = None,
        attribute_map: Optional[dict[str, str]] = None,
    ) -> SynthesisConfig:
        """Build a SynthesisConfig using LLM-inferred domain knowledge.

        This method uses an LLM to analyze the data and generate domain-specific
        prompts and configuration, rather than using generic templates.

        Args:
            schema: The analyzed data schema.
            goal: Synthesis goal ("qa", "conversation", "augmentation", "instruction").
            num_samples: Number of samples to generate.
            output_path: Optional output path for generated data.
            domain: Optional pre-computed DomainAnalysis. If not provided,
                will be computed using the LLM analyzer.
            llm_analyzer: Optional LLMAnalyzer instance. If not provided,
                a new one will be created.
            attribute_map: Optional explicit column-to-role mapping (e.g.,
                {"context": "description_col", "question": "query_col"}).

        Returns:
            A configured SynthesisConfig with domain-specific prompts.
        """
        from oumi.core.configs.synthesis_config import SynthesisStrategy
        from oumi.onboarding.llm_analyzer import LLMAnalyzer as LLMAnalyzerClass

        # Create LLM analyzer if not provided
        if llm_analyzer is None:
            llm_analyzer = LLMAnalyzerClass()

        # Get domain analysis if not provided
        if domain is None:
            domain = llm_analyzer.analyze(schema)

        # Get LLM-inferred config
        inferred = llm_analyzer.infer_synth_config(schema, goal, domain)

        # Build strategy params using inferred config
        strategy_params = self._build_inferred_strategy_params(
            schema, goal, domain, inferred, attribute_map=attribute_map
        )

        # Create output path if not provided
        if output_path is None:
            output_path = f"synth_{goal}_output.jsonl"

        return SynthesisConfig(
            strategy=SynthesisStrategy.GENERAL,
            num_samples=num_samples,
            output_path=output_path,
            strategy_params=strategy_params,
            inference_config=self._create_inference_config(),
        )

    def from_schema_with_custom_prompts(
        self,
        schema: DataSchema,
        goal: SynthGoal = "qa",
        num_samples: int = 100,
        output_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        question_template: Optional[str] = None,
        answer_template: Optional[str] = None,
        postprocessing: Optional[dict] = None,
        attribute_map: Optional[dict[str, str]] = None,
    ) -> SynthesisConfig:
        """Build a SynthesisConfig using custom prompts provided by the user.

        This method allows for iteratively refined prompts from the wizard
        to be used directly in the config.

        Args:
            schema: The analyzed data schema.
            goal: Synthesis goal ("qa", "conversation", "augmentation", "instruction").
            num_samples: Number of samples to generate.
            output_path: Optional output path for generated data.
            system_prompt: Custom system prompt for the AI persona.
            question_template: Custom question/instruction template with {placeholders}.
            answer_template: Custom answer generation template.
            postprocessing: Postprocessing config dict (cut_prefix, strip_whitespace).
            attribute_map: Optional explicit column-to-role mapping (e.g.,
                {"context": "description_col", "question": "query_col"}).

        Returns:
            A configured SynthesisConfig with custom prompts.
        """
        from oumi.core.configs.synthesis_config import SynthesisStrategy

        # Build strategy params using custom prompts
        strategy_params = self._build_custom_strategy_params(
            schema,
            goal,
            system_prompt or self.GOAL_TEMPLATES.get(goal, {}).get("system_prompt", ""),
            question_template or "",
            answer_template or "",
            postprocessing or {},
            attribute_map=attribute_map,
        )

        # Create output path if not provided
        if output_path is None:
            output_path = f"synth_{goal}_output.jsonl"

        return SynthesisConfig(
            strategy=SynthesisStrategy.GENERAL,
            num_samples=num_samples,
            output_path=output_path,
            strategy_params=strategy_params,
            inference_config=self._create_inference_config(),
        )

    def _build_custom_strategy_params(
        self,
        schema: DataSchema,
        goal: SynthGoal,
        system_prompt: str,
        question_template: str,
        answer_template: str,
        postprocessing: dict,
        attribute_map: Optional[dict[str, str]] = None,
    ) -> GeneralSynthesisParams:
        """Build strategy params using custom prompts.

        Args:
            schema: The analyzed data schema.
            goal: Synthesis goal.
            system_prompt: Custom system prompt.
            question_template: Question/instruction template.
            answer_template: Answer generation template.
            postprocessing: Postprocessing config dict.
            attribute_map: Optional explicit column-to-role mapping that takes
                precedence over auto-detected mappings.
        """
        params = GeneralSynthesisParams()

        # Add input data source if schema has data
        if schema.source_path and schema.detected_format in (
            "csv",
            "excel",
            "json",
            "jsonl",
        ):
            # Use provided attribute_map or build from columns
            if attribute_map:
                # Explicit column assignments provided - use directly
                final_attribute_map = attribute_map
            else:
                # Build attribute map from columns
                final_attribute_map = {}
                if schema.columns:
                    text_cols = [c for c in schema.columns if c.is_text]
                    if text_cols:
                        best_col = max(text_cols, key=lambda c: c.avg_length or 0)
                        final_attribute_map = {best_col.name: "context"}
                    elif schema.columns:
                        final_attribute_map = {schema.columns[0].name: "context"}

            params.input_data = [
                DatasetSource(
                    path=_convert_to_supported_format(schema.source_path),
                    attribute_map=final_attribute_map if final_attribute_map else None,
                )
            ]

        # Build generated attributes using custom prompts
        cut_prefix = postprocessing.get("cut_prefix", "Answer:")
        strip_ws = postprocessing.get("strip_whitespace", True)

        question_attr = GeneratedAttribute(
            id="question_raw",
            instruction_messages=[
                TextMessage(role=Role.SYSTEM, content=system_prompt),
                TextMessage(role=Role.USER, content=question_template),
            ],
            postprocessing_params=GeneratedAttributePostprocessingParams(
                id="question",
                cut_prefix="Question:",
                strip_whitespace=True,
            ),
        )

        answer_attr = GeneratedAttribute(
            id="answer_raw",
            instruction_messages=[
                TextMessage(role=Role.SYSTEM, content=system_prompt),
                TextMessage(role=Role.USER, content=answer_template),
            ],
            postprocessing_params=GeneratedAttributePostprocessingParams(
                id="answer",
                cut_prefix=cut_prefix,
                strip_whitespace=strip_ws,
            ),
        )

        params.generated_attributes = [question_attr, answer_attr]
        params.transformed_attributes = self._build_qa_transform()
        params.passthrough_attributes = ["conversation", "question", "answer"]

        return params

    def _build_inferred_strategy_params(
        self,
        schema: DataSchema,
        goal: SynthGoal,
        domain: "DomainAnalysis",
        inferred: "InferredConfig",
        attribute_map: Optional[dict[str, str]] = None,
    ) -> GeneralSynthesisParams:
        """Build strategy params using LLM-inferred configuration.

        Args:
            schema: The analyzed data schema.
            goal: Synthesis goal.
            domain: Domain analysis from LLM.
            inferred: Inferred config from LLM.
            attribute_map: Optional explicit column-to-role mapping that takes
                precedence over both inferred and auto-detected mappings.
        """
        params = GeneralSynthesisParams()

        # Add input data source if schema has data
        if schema.source_path and schema.detected_format in (
            "csv",
            "excel",
            "json",
            "jsonl",
        ):
            # Priority: explicit attribute_map > inferred mappings > auto-detect
            if attribute_map:
                # User provided explicit column assignments
                final_attribute_map = attribute_map
            elif inferred.field_mappings:
                # Use LLM-inferred field mappings, stripping any braces from values
                # LLM sometimes returns {col} instead of col
                final_attribute_map = {
                    k: v.strip("{}") if isinstance(v, str) else v
                    for k, v in inferred.field_mappings.items()
                }
            else:
                # Fallback to auto-detect
                final_attribute_map = None
                if schema.columns:
                    text_cols = [c for c in schema.columns if c.is_text]
                    if text_cols:
                        best_col = max(text_cols, key=lambda c: c.avg_length or 0)
                        final_attribute_map = {best_col.name: "context"}
                    elif schema.columns:
                        final_attribute_map = {schema.columns[0].name: "context"}

            params.input_data = [
                DatasetSource(
                    path=_convert_to_supported_format(schema.source_path),
                    attribute_map=final_attribute_map if final_attribute_map else None,
                )
            ]

        # Build generated attributes using inferred prompts
        if goal == "qa":
            params.generated_attributes = self._build_inferred_qa_attributes(
                domain, inferred
            )
            params.transformed_attributes = self._build_qa_transform()
            params.passthrough_attributes = ["conversation", "question", "answer"]
        elif goal == "conversation":
            params.generated_attributes = self._build_inferred_conversation_attributes(
                domain, inferred
            )
            params.passthrough_attributes = ["conversation"]
        elif goal == "augmentation":
            params.generated_attributes = self._build_inferred_augmentation_attributes(
                domain, inferred
            )
            params.passthrough_attributes = ["augmented"]
        elif goal == "instruction":
            params.generated_attributes = self._build_inferred_instruction_attributes(
                domain, inferred, schema
            )
            params.passthrough_attributes = ["output"]

        return params

    def _build_inferred_qa_attributes(
        self,
        domain: "DomainAnalysis",
        inferred: "InferredConfig",
    ) -> list[GeneratedAttribute]:
        """Build Q&A attributes using LLM-inferred prompts."""
        # Use inferred system prompt or fall back to domain-enhanced default
        system_prompt = inferred.system_prompt or domain.suggested_persona
        if not system_prompt:
            system_prompt = self.GOAL_TEMPLATES["qa"]["system_prompt"]

        # Use inferred instruction template or build from domain knowledge
        question_template = inferred.instruction_template
        if not question_template:
            terminology_str = ", ".join(domain.terminology[:5]) if domain.terminology else ""
            question_template = f"""Based on the following information about {domain.domain}, generate a thoughtful question.

Domain terminology to use: {terminology_str}

{{context}}

Format your response as:
Question: <your question>"""

        # Build postprocessing params from inferred config
        postproc = inferred.postprocessing or {}
        cut_prefix = postproc.get("cut_prefix", "Question:")

        question_attr = GeneratedAttribute(
            id="question_raw",
            instruction_messages=[
                TextMessage(role=Role.SYSTEM, content=system_prompt),
                TextMessage(role=Role.USER, content=question_template),
            ],
            postprocessing_params=GeneratedAttributePostprocessingParams(
                id="question",
                cut_prefix=cut_prefix,
                strip_whitespace=postproc.get("strip_whitespace", True),
            ),
        )

        # Build answer attribute
        answer_template = f"""Provide a helpful, accurate answer to this question about {domain.domain}:

{{question}}

Use appropriate domain terminology and be specific.

Format your response as:
Answer: <your answer>"""

        answer_attr = GeneratedAttribute(
            id="answer_raw",
            instruction_messages=[
                TextMessage(role=Role.SYSTEM, content=system_prompt),
                TextMessage(role=Role.USER, content=answer_template),
            ],
            postprocessing_params=GeneratedAttributePostprocessingParams(
                id="answer",
                cut_prefix="Answer:",
                strip_whitespace=True,
            ),
        )

        return [question_attr, answer_attr]

    def _build_inferred_conversation_attributes(
        self,
        domain: "DomainAnalysis",
        inferred: "InferredConfig",
    ) -> list[GeneratedAttribute]:
        """Build conversation attributes using LLM-inferred prompts."""
        system_prompt = inferred.system_prompt or domain.suggested_persona
        if not system_prompt:
            system_prompt = f"""You are an expert at generating realistic conversation exchanges in {domain.domain}.
Create natural dialogue that demonstrates expertise and helpfulness."""

        instruction = inferred.instruction_template
        if not instruction:
            instruction = f"""Generate a realistic conversation based on this context about {domain.domain}:

{{context}}

The conversation should be professional and demonstrate domain knowledge."""

        return [
            GeneratedAttribute(
                id="conversation",
                instruction_messages=[
                    TextMessage(role=Role.SYSTEM, content=system_prompt),
                    TextMessage(role=Role.USER, content=instruction),
                ],
            )
        ]

    def _build_inferred_augmentation_attributes(
        self,
        domain: "DomainAnalysis",
        inferred: "InferredConfig",
    ) -> list[GeneratedAttribute]:
        """Build augmentation attributes using LLM-inferred prompts."""
        system_prompt = inferred.system_prompt or domain.suggested_persona
        if not system_prompt:
            system_prompt = f"""You are an expert at creating variations of {domain.domain} content.
Generate diverse variations that preserve the core meaning while varying style and phrasing."""

        instruction = inferred.instruction_template
        if not instruction:
            terminology_str = ", ".join(domain.terminology[:5]) if domain.terminology else ""
            instruction = f"""Create a variation of the following {domain.domain} content:

{{context}}

Maintain the core meaning but vary the phrasing and style.
Use domain terminology: {terminology_str}"""

        return [
            GeneratedAttribute(
                id="augmented",
                instruction_messages=[
                    TextMessage(role=Role.SYSTEM, content=system_prompt),
                    TextMessage(role=Role.USER, content=instruction),
                ],
            )
        ]

    def _build_inferred_instruction_attributes(
        self,
        domain: "DomainAnalysis",
        inferred: "InferredConfig",
        schema: DataSchema,
    ) -> list[GeneratedAttribute]:
        """Build instruction attributes using LLM-inferred prompts."""
        system_prompt = inferred.system_prompt or domain.suggested_persona
        if not system_prompt:
            system_prompt = f"""You are following specific instructions to generate {domain.domain} content.
Follow the provided instructions precisely and generate high-quality outputs."""

        # Use document content or inferred instruction
        instruction = "{instruction}"
        if schema.raw_text:
            instruction = schema.raw_text[:2000]
        elif inferred.instruction_template:
            instruction = inferred.instruction_template

        return [
            GeneratedAttribute(
                id="output",
                instruction_messages=[
                    TextMessage(role=Role.SYSTEM, content=system_prompt),
                    TextMessage(role=Role.USER, content=instruction),
                ],
            )
        ]

    def from_template(
        self,
        template_name: str,
        num_samples: int = 100,
        output_path: Optional[str] = None,
        **overrides,
    ) -> SynthesisConfig:
        """Build a SynthesisConfig from a pre-built template.

        Args:
            template_name: Name of the template to use.
            num_samples: Number of samples to generate.
            output_path: Optional output path.
            **overrides: Additional overrides for the config.

        Returns:
            A configured SynthesisConfig.
        """
        # Load template from configs/templates/synth/{template_name}.yaml
        template_path = Path(__file__).parent.parent.parent.parent / "configs" / "templates" / "synth" / f"{template_name}.yaml"

        if template_path.exists():
            config = SynthesisConfig.from_yaml(str(template_path))
            config.num_samples = num_samples
            if output_path:
                config.output_path = output_path
            return config
        else:
            raise ValueError(f"Template not found: {template_name}")


class JudgeConfigBuilder(ConfigBuilder):
    """Build JudgeConfig from customer data.

    This builder creates judge configurations for evaluating data quality,
    compliance, relevance, and other criteria.

    Example:
        >>> from oumi.onboarding import DataAnalyzer, JudgeConfigBuilder
        >>> analyzer = DataAnalyzer()
        >>> schema = analyzer.analyze("./conversations.csv")
        >>> builder = JudgeConfigBuilder()
        >>> config = builder.from_schema(schema, judge_type="compliance")
    """

    # Templates for different judge types
    JUDGE_TEMPLATES = {
        "generic": {
            "system_instruction": "You are an expert evaluator. Assess the quality of the provided content objectively.",
            "prompt_template": """Evaluate the following content:

{context}

Provide your assessment.""",
            "judgment_type": JudgeOutputType.FLOAT,
        },
        "compliance": {
            "system_instruction": """You are an expert compliance evaluator.
Assess whether the response adheres to the specified guidelines and policies.
Be thorough and objective in your evaluation.""",
            "prompt_template": """Evaluate whether this response is compliant:

Response: {response}

Guidelines: {context}

Is this response compliant with the guidelines?""",
            "judgment_type": JudgeOutputType.BOOL,
        },
        "relevance": {
            "system_instruction": """You are an expert at evaluating relevance.
Assess how well the answer addresses the question based on the given context.""",
            "prompt_template": """Context: {context}
Question: {question}
Answer: {answer}

Is the answer relevant and accurate given the context and question?""",
            "judgment_type": JudgeOutputType.BOOL,
        },
        "safety": {
            "system_instruction": """You are an expert at evaluating content safety.
Assess whether the content is safe, appropriate, and free from harmful material.""",
            "prompt_template": """Evaluate the safety of this content:

{response}

Is this content safe and appropriate?""",
            "judgment_type": JudgeOutputType.BOOL,
        },
        "groundedness": {
            "system_instruction": """You are an expert at evaluating factual accuracy.
Assess whether claims in the response are supported by the provided context.""",
            "prompt_template": """Context: {context}
Response: {response}

Is the response grounded in and supported by the context?""",
            "judgment_type": JudgeOutputType.BOOL,
        },
    }

    def from_schema(
        self,
        schema: DataSchema,
        judge_type: JudgeType = "generic",
        custom_criteria: Optional[str] = None,
        mappings: Optional[list[FieldMapping]] = None,
    ) -> JudgeConfig:
        """Build a JudgeConfig from analyzed data.

        Args:
            schema: The analyzed data schema.
            judge_type: Type of judge ("generic", "compliance", "relevance", etc.).
            custom_criteria: Optional custom evaluation criteria.
            mappings: Optional pre-computed field mappings.

        Returns:
            A configured JudgeConfig.
        """
        # Get field mappings if not provided
        if mappings is None:
            mappings = self.field_mapper.suggest_mappings(schema)

        template = self.JUDGE_TEMPLATES[judge_type]

        # Build prompt template with mapped fields
        prompt_template = self._adapt_prompt_template(
            template["prompt_template"],
            mappings,
            custom_criteria,
        )

        judge_params = JudgeParams(
            system_instruction=template["system_instruction"],
            prompt_template=prompt_template,
            response_format=JudgeResponseFormat.XML,
            judgment_type=template["judgment_type"],
            include_explanation=True,
        )

        return JudgeConfig(
            judge_params=judge_params,
            inference_config=self._create_inference_config(),
        )

    def _adapt_prompt_template(
        self,
        template: str,
        mappings: list[FieldMapping],
        custom_criteria: Optional[str],
    ) -> str:
        """Adapt prompt template to use customer column names."""
        result = template

        # Replace standard placeholders with mapped column names
        mapping_dict = {m.oumi_placeholder: m.customer_column for m in mappings}

        for placeholder, column in mapping_dict.items():
            result = result.replace(f"{{{placeholder}}}", f"{{{column}}}")

        # Add custom criteria if provided
        if custom_criteria:
            result = f"Evaluation Criteria: {custom_criteria}\n\n{result}"

        return result

    def from_schema_with_inference(
        self,
        schema: DataSchema,
        judge_type: JudgeType = "generic",
        domain: Optional["DomainAnalysis"] = None,
        llm_analyzer: Optional["LLMAnalyzer"] = None,
    ) -> JudgeConfig:
        """Build a JudgeConfig using LLM-inferred domain knowledge.

        This method uses an LLM to analyze the data and generate domain-specific
        evaluation criteria and prompts, rather than using generic templates.

        Args:
            schema: The analyzed data schema.
            judge_type: Type of judge ("generic", "compliance", "relevance", etc.).
            domain: Optional pre-computed DomainAnalysis. If not provided,
                will be computed using the LLM analyzer.
            llm_analyzer: Optional LLMAnalyzer instance. If not provided,
                a new one will be created.

        Returns:
            A configured JudgeConfig with domain-specific evaluation criteria.
        """
        from oumi.onboarding.llm_analyzer import LLMAnalyzer as LLMAnalyzerClass

        # Create LLM analyzer if not provided
        if llm_analyzer is None:
            llm_analyzer = LLMAnalyzerClass()

        # Get domain analysis if not provided
        if domain is None:
            domain = llm_analyzer.analyze(schema)

        # Get LLM-inferred judge config
        inferred = llm_analyzer.infer_judge_config(schema, judge_type, domain)

        # Build judge params using inferred config
        system_instruction = inferred.system_prompt
        if not system_instruction:
            system_instruction = (
                f"You are an expert evaluator for {domain.domain} content. "
                f"Assess the quality objectively, watching for: {', '.join(domain.common_issues[:3])}."
            )

        prompt_template = inferred.instruction_template
        if not prompt_template:
            quality_signals = ", ".join(domain.quality_signals[:3]) if domain.quality_signals else "accuracy, clarity, completeness"
            prompt_template = f"""Evaluate the following {domain.domain} content:

{{context}}

Assessment criteria: {quality_signals}

Provide your evaluation."""

        # Determine judgment type based on judge_type
        judgment_type = self.JUDGE_TEMPLATES.get(
            judge_type, self.JUDGE_TEMPLATES["generic"]
        )["judgment_type"]

        judge_params = JudgeParams(
            system_instruction=system_instruction,
            prompt_template=prompt_template,
            response_format=JudgeResponseFormat.XML,
            judgment_type=judgment_type,
            include_explanation=True,
        )

        return JudgeConfig(
            judge_params=judge_params,
            inference_config=self._create_inference_config(),
        )

    def from_custom_criteria(
        self,
        schema: DataSchema,
        judge_name: str,
        criteria: str,
        description: str = "",
        judgment_type: JudgeOutputType = JudgeOutputType.FLOAT,
    ) -> JudgeConfig:
        """Build a JudgeConfig from custom evaluation criteria.

        This method creates a judge configuration based on user-defined criteria,
        allowing for flexible evaluation beyond the predefined judge templates.

        Args:
            schema: The analyzed data schema.
            judge_name: A short name for this judge (e.g., "accuracy", "helpfulness").
            criteria: The specific evaluation criteria as a detailed description.
            description: Optional additional context about what this judge evaluates.
            judgment_type: The type of judgment output (FLOAT, BOOL, or LIKERT).

        Returns:
            A configured JudgeConfig with custom evaluation criteria.

        Example:
            >>> builder = JudgeConfigBuilder()
            >>> config = builder.from_custom_criteria(
            ...     schema=schema,
            ...     judge_name="domain_expertise",
            ...     criteria="Evaluate whether the response demonstrates deep knowledge "
            ...              "of machine learning concepts and uses correct terminology.",
            ...     description="Assesses ML domain expertise in responses",
            ...     judgment_type=JudgeOutputType.LIKERT,
            ... )
        """
        # Build system instruction based on the judge name and description
        system_instruction = f"""You are an expert evaluator assessing '{judge_name}'.
{description}

Your task is to objectively evaluate content based on the following criteria:
{criteria}

Be thorough, fair, and consistent in your evaluations. Provide clear explanations for your judgments."""

        # Build prompt template that can handle various input formats
        prompt_template = f"""Evaluate the following content for '{judge_name}':

**Criteria**: {criteria}

**Content to evaluate**:
{{context}}

{{{{#if response}}}}
**Response**: {{response}}
{{{{/if}}}}

{{{{#if question}}}}
**Question**: {{question}}
{{{{/if}}}}

Provide your evaluation based on the criteria above."""

        judge_params = JudgeParams(
            system_instruction=system_instruction,
            prompt_template=prompt_template,
            response_format=JudgeResponseFormat.XML,
            judgment_type=judgment_type,
            include_explanation=True,
        )

        return JudgeConfig(
            judge_params=judge_params,
            inference_config=self._create_inference_config(),
        )


class TrainConfigBuilder(ConfigBuilder):
    """Build TrainingConfig from customer data.

    This builder creates training configurations for fine-tuning models
    on customer data or synthetically generated data.

    Example:
        >>> from oumi.onboarding import TrainConfigBuilder
        >>> builder = TrainConfigBuilder()
        >>> config = builder.from_data_path(
        ...     "./training_data.jsonl",
        ...     base_model="meta-llama/Llama-3.2-1B-Instruct",
        ...     use_lora=True,
        ... )
    """

    DEFAULT_MODELS = {
        "small": "meta-llama/Llama-3.2-1B-Instruct",
        "medium": "meta-llama/Llama-3.2-3B-Instruct",
        "large": "meta-llama/Llama-3.1-8B-Instruct",
    }

    def from_schema(
        self,
        schema: DataSchema,
        base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
        use_lora: bool = True,
        output_dir: str = "./output/training",
        max_steps: int = 1000,
    ) -> TrainingConfig:
        """Build a TrainingConfig from analyzed data.

        Args:
            schema: The analyzed data schema (usually from synth output).
            base_model: Base model to fine-tune.
            use_lora: Whether to use LoRA for efficient fine-tuning.
            output_dir: Directory to save model outputs.
            max_steps: Maximum training steps.

        Returns:
            A configured TrainingConfig.
        """
        return self.from_data_path(
            schema.source_path,
            base_model=base_model,
            use_lora=use_lora,
            output_dir=output_dir,
            max_steps=max_steps,
        )

    def from_data_path(
        self,
        data_path: str,
        base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
        use_lora: bool = True,
        output_dir: str = "./output/training",
        max_steps: int = 1000,
        learning_rate: float = 5e-6,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
    ) -> TrainingConfig:
        """Build a TrainingConfig from a data path.

        Args:
            data_path: Path to training data (JSONL format).
            base_model: Base model to fine-tune.
            use_lora: Whether to use LoRA.
            output_dir: Output directory.
            max_steps: Maximum training steps.
            learning_rate: Learning rate.
            batch_size: Per-device batch size.
            gradient_accumulation_steps: Gradient accumulation steps.

        Returns:
            A configured TrainingConfig.
        """
        from oumi.core.configs.params.data_params import (
            DataParams,
            DatasetParams,
            DatasetSplitParams,
        )
        from oumi.core.configs.params.peft_params import PeftParams
        from oumi.core.configs.params.training_params import (
            TrainerType,
            TrainingParams,
        )

        # Build data params
        data = DataParams(
            train=DatasetSplitParams(
                datasets=[DatasetParams(dataset_path=data_path)],
                collator_name="text_sft",
            )
        )

        # Build model params
        # Note: When using mixed precision training, model must be loaded in fp32
        model = ModelParams(
            model_name=base_model,
            model_max_length=4096,
            torch_dtype_str="float32",  # Required for mixed precision training
            attn_implementation="sdpa",
        )

        # Build training params
        training = TrainingParams(
            trainer_type=TrainerType.TRL_SFT,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            save_steps=100,
            logging_steps=10,
            output_dir=output_dir,
            mixed_precision_dtype="bf16",
        )

        # Build PEFT params if using LoRA
        peft = None
        if use_lora:
            peft = PeftParams(
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.05,
            )

        return TrainingConfig(
            data=data,
            model=model,
            training=training,
            peft=peft,
        )
