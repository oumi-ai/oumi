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
analyzed customer data schemas using LLM-powered prompt generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from oumi.cli.onboard.prompts import load_prompt
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
from oumi.utils.logging import logger

if TYPE_CHECKING:
    from oumi.onboarding.llm_analyzer import DomainAnalysis

# Type aliases
SynthGoal = Literal["qa", "conversation", "augmentation", "instruction"]
JudgeType = Literal["generic", "compliance", "relevance", "safety", "groundedness"]

# Constants
TABULAR_FORMATS = frozenset({"csv", "excel", "json", "jsonl"})
SUPPORTED_DATASET_EXTENSIONS = {".csv", ".json", ".jsonl", ".tsv", ".parquet"}


class SynthPlaceholder:
    """Standard placeholder names for synthesis configs."""

    QUESTION = "synth_question"
    ANSWER = "synth_answer"
    CONVERSATION = "synth_conversation"
    CONTEXT = "context"


def _convert_to_supported_format(
    file_path: str, output_dir: Optional[str] = None
) -> str:
    """Convert unsupported file types to CSV for use with DatasetSource.

    Args:
        file_path: Path to the source file.
        output_dir: Optional output directory. If not provided, uses same directory.

    Returns:
        Path to the converted file (CSV), or original path if already supported.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in SUPPORTED_DATASET_EXTENSIONS:
        return file_path

    if output_dir:
        out_path = Path(output_dir) / f"{path.stem}.csv"
    else:
        out_path = path.with_suffix(".csv")

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

    raise ValueError(
        f"Unsupported file type: {ext}. Supported types: {SUPPORTED_DATASET_EXTENSIONS}"
    )


@dataclass
class BuilderOptions:
    """Common options for config builders."""

    model_name: str = "claude-sonnet-4-20250514"
    engine: str = "ANTHROPIC"
    temperature: float = 1.0
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

    @abstractmethod
    def build(self, schema: DataSchema, **kwargs) -> Any:
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
            ),
            remote_params=RemoteParams(
                num_workers=self.options.num_workers,
                politeness_policy=60,
            ),
        )


class SynthConfigBuilder(ConfigBuilder):
    """Build SynthesisConfig from customer data using LLM-powered prompts.

    This builder analyzes customer data and generates a synthesis configuration
    with task-specific prompts based on domain analysis and wizard inputs.

    Example:
        >>> from oumi.onboarding import DataAnalyzer, SynthConfigBuilder
        >>> analyzer = DataAnalyzer()
        >>> schema = analyzer.analyze("./customer_data.csv")
        >>> builder = SynthConfigBuilder()
        >>> config = builder.build(
        ...     schema,
        ...     task_type="extraction",
        ...     task_description="Extract product info from reviews",
        ...     system_prompt="You are a product data extractor.",
        ... )
        >>> config.to_yaml("./synth_config.yaml")
    """

    def build(
        self,
        schema: DataSchema,
        goal: SynthGoal = "qa",
        task_type: str = "generation",
        task_description: str = "",
        system_prompt: str = "",
        output_format: Optional[str] = None,
        domain: Optional["DomainAnalysis"] = None,
        num_samples: int = 100,
        output_path: Optional[str] = None,
        attribute_map: Optional[dict[str, str]] = None,
        generation_mode: str = "synthesis",
        labeled_examples: Optional[list[dict]] = None,
        unlabeled_prompts: Optional[list[str]] = None,
        seed_data: Optional[dict[str, list[str]]] = None,
        user_prompt_template: Optional[str] = None,
        template_mapping: Optional[dict[str, str]] = None,
    ) -> SynthesisConfig:
        """Build a SynthesisConfig with task-specific prompts.

        Args:
            schema: The analyzed data schema.
            goal: Synthesis goal ("qa", "conversation", "augmentation", "instruction").
            task_type: Task type from wizard (extraction, classification, generation, etc.).
            task_description: User's task description from wizard.
            system_prompt: Generated system prompt from wizard.
            output_format: Description of expected output format.
        domain: Optional DomainAnalysis for domain-specific terminology.
        num_samples: Number of samples to generate.
        output_path: Optional output path for generated data.
        attribute_map: Optional explicit column-to-role mapping.
        generation_mode: "synthesis" | "augmentation" | "teacher_labeling".
        labeled_examples: Labeled input/output pairs when augmenting.
        unlabeled_prompts: Prompts without outputs when teacher labeling.
        seed_data: Optional seed values for diversity sampling.
        user_prompt_template: Optional user prompt template to follow.
        template_mapping: Mapping of template variables to columns.

        Returns:
            A configured SynthesisConfig with task-specific prompts.
        """
        from oumi.core.configs.synthesis_config import SynthesisStrategy

        labeled_examples = labeled_examples or []
        unlabeled_prompts = unlabeled_prompts or []
        seed_data = seed_data or {}

        valid_modes = {"synthesis", "augmentation", "teacher_labeling"}
        generation_mode_normalized = generation_mode or "synthesis"
        if generation_mode_normalized not in valid_modes:
            logger.warning(
                f"Unknown generation_mode '{generation_mode_normalized}', defaulting to 'synthesis'. "
                f"Valid modes: {', '.join(sorted(valid_modes))}"
            )
            generation_mode_normalized = "synthesis"

        strategy_params = self._build_strategy_params(
            schema=schema,
            goal=goal,
            task_type=task_type,
            task_description=task_description,
            system_prompt=system_prompt,
            output_format=output_format,
            domain=domain,
            attribute_map=attribute_map,
            generation_mode=generation_mode_normalized,
            labeled_examples=labeled_examples,
            unlabeled_prompts=unlabeled_prompts,
            seed_data=seed_data,
            user_prompt_template=user_prompt_template,
            template_mapping=template_mapping,
        )

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
        task_type: str,
        task_description: str,
        system_prompt: str,
        output_format: Optional[str],
        domain: Optional["DomainAnalysis"],
        attribute_map: Optional[dict[str, str]],
        generation_mode: str,
        labeled_examples: list[dict],
        unlabeled_prompts: list[str],
        seed_data: dict[str, list[str]],
        user_prompt_template: Optional[str],
        template_mapping: Optional[dict[str, str]],
    ) -> GeneralSynthesisParams:
        """Build strategy params with task-specific prompts from Jinja templates.

        Supports three modes:
        - synthesis: generate both inputs and outputs
        - augmentation: augment existing labeled examples
        - teacher_labeling: generate outputs for unlabeled prompts
        """
        from oumi.core.configs.params.synthesis_params import ExampleSource

        params = GeneralSynthesisParams()

        seed_attributes = self._build_seed_sampled_attributes(seed_data)
        if seed_attributes:
            params.sampled_attributes = seed_attributes

        if generation_mode == "augmentation":
            examples = [
                {
                    "seed_question": ex.get("input", ""),
                    "seed_answer": ex.get("output", ""),
                }
                for ex in labeled_examples
                if ex.get("input") and ex.get("output")
            ]
            if not examples:
                examples = self._build_inline_examples(schema, task_description)

            params.input_examples = [ExampleSource(examples=examples)]
            params.generated_attributes = self._build_augmentation_attributes(
                task_description=task_description,
                system_prompt=system_prompt,
                domain=domain,
                labeled_examples=labeled_examples,
                seed_data=seed_data,
                user_prompt_template=user_prompt_template,
            )
        elif generation_mode == "teacher_labeling":
            prompt_examples = [
                {SynthPlaceholder.QUESTION: prompt}
                for prompt in unlabeled_prompts
                if prompt
            ]
            if not prompt_examples:
                inline_examples = self._build_inline_examples(schema, task_description)
                prompt_examples = [
                    {SynthPlaceholder.QUESTION: example.get(SynthPlaceholder.CONTEXT, "")}
                    for example in inline_examples
                ]
                if not prompt_examples:
                    prompt_examples = [
                        {SynthPlaceholder.QUESTION: task_description or "Generate a prompt"}
                    ]

            params.input_examples = [ExampleSource(examples=prompt_examples)]
            params.generated_attributes = self._build_teacher_labeling_attributes(
                task_description=task_description,
                system_prompt=system_prompt,
                domain=domain,
                user_prompt_template=user_prompt_template,
                template_mapping=template_mapping,
                seed_data=seed_data,
            )
        else:
            # Build input data source
            if schema and schema.source_path and schema.detected_format in TABULAR_FORMATS:
                final_attribute_map = self._build_attribute_map(schema, attribute_map)
                params.input_data = [
                    DatasetSource(
                        path=_convert_to_supported_format(schema.source_path),
                        attribute_map=final_attribute_map if final_attribute_map else None,
                    )
                ]
            else:
                # Fallback to inline examples
                examples = self._build_inline_examples(schema, task_description)
                params.input_examples = [ExampleSource(examples=examples)]

            # Build generated attributes using Jinja templates
            params.generated_attributes = self._build_generated_attributes(
                task_type=task_type,
                task_description=task_description,
                system_prompt=system_prompt,
                domain=domain,
                schema=schema,
                output_format=output_format,
                seed_data=seed_data,
                user_prompt_template=user_prompt_template,
                template_mapping=template_mapping,
            )

        # Build transformation to conversation format
        params.transformed_attributes = self._build_conversation_transform(system_prompt)

        # Set passthrough attributes
        params.passthrough_attributes = [
            SynthPlaceholder.CONVERSATION,
            SynthPlaceholder.QUESTION,
            SynthPlaceholder.ANSWER,
        ]

        return params

    def _build_attribute_map(
        self,
        schema: DataSchema,
        attribute_map: Optional[dict[str, str]],
    ) -> dict[str, str]:
        """Build attribute map, ensuring context mapping exists."""
        if attribute_map:
            final_map = dict(attribute_map)
        else:
            final_map = {}

        # Ensure we have a context mapping
        if SynthPlaceholder.CONTEXT not in final_map.values() and schema.columns:
            # Find best column for context
            text_cols = [c for c in schema.columns if c.is_text]
            if text_cols:
                best_col = max(text_cols, key=lambda c: c.avg_length or 0)
                final_map[best_col.name] = SynthPlaceholder.CONTEXT
            elif schema.columns:
                final_map[schema.columns[0].name] = SynthPlaceholder.CONTEXT

        return final_map

    def _build_inline_examples(
        self,
        schema: DataSchema,
        task_description: str,
    ) -> list[dict]:
        """Build inline examples when no tabular data source is available."""
        examples = []

        if schema and schema.sample_rows:
            for row in schema.sample_rows[:5]:
                text_parts = [str(v) for v in row.values() if v and str(v).strip()]
                if text_parts:
                    examples.append({SynthPlaceholder.CONTEXT: " | ".join(text_parts)})
        elif schema and schema.raw_text:
            examples.append({SynthPlaceholder.CONTEXT: schema.raw_text[:2000]})

        if not examples:
            examples.append(
                {SynthPlaceholder.CONTEXT: task_description or "Generate a sample input"}
            )

        return examples

    def _build_generated_attributes(
        self,
        task_type: str,
        task_description: str,
        system_prompt: str,
        domain: Optional["DomainAnalysis"],
        schema: DataSchema,
        output_format: Optional[str],
        seed_data: dict[str, list[str]],
        user_prompt_template: Optional[str],
        template_mapping: Optional[dict[str, str]],
    ) -> list[GeneratedAttribute]:
        """Build generated attributes using task-specific Jinja templates."""
        seed_context = self._format_seed_context(seed_data)
        template_vars = {
            "task_description": task_description,
            "system_prompt": system_prompt,
            "domain": domain,
            "sample_rows": schema.sample_rows[:3] if schema and schema.sample_rows else [],
            "output_format": output_format,
            "seed_context": seed_context,
            "user_prompt_template": user_prompt_template or "",
            "template_mapping": template_mapping or {},
        }

        # Map task_type to template name (default to generation if unknown)
        valid_types = {"extraction", "classification", "qa", "transformation", "generation"}
        if task_type not in valid_types:
            logger.warning(
                f"Unknown task_type '{task_type}', defaulting to 'generation'. "
                f"Valid types: {', '.join(sorted(valid_types))}"
            )
        template_type = task_type if task_type in valid_types else "generation"

        # Load question and answer templates
        question_prompt = load_prompt(f"synth/{template_type}_question", **template_vars)
        answer_prompt = load_prompt(f"synth/{template_type}_answer", **template_vars)

        question_attr = GeneratedAttribute(
            id=f"{SynthPlaceholder.QUESTION}_raw",
            instruction_messages=[
                TextMessage(role=Role.SYSTEM, content=system_prompt),
                TextMessage(role=Role.USER, content=question_prompt),
            ],
            postprocessing_params=GeneratedAttributePostprocessingParams(
                id=SynthPlaceholder.QUESTION,
                cut_prefix="Question:",
                strip_whitespace=True,
            ),
        )

        answer_attr = GeneratedAttribute(
            id=f"{SynthPlaceholder.ANSWER}_raw",
            instruction_messages=[
                TextMessage(role=Role.SYSTEM, content=system_prompt),
                TextMessage(role=Role.USER, content=answer_prompt),
            ],
            postprocessing_params=GeneratedAttributePostprocessingParams(
                id=SynthPlaceholder.ANSWER,
                cut_prefix="Answer:",
                strip_whitespace=True,
            ),
        )

        return [question_attr, answer_attr]

    def _build_augmentation_attributes(
        self,
        task_description: str,
        system_prompt: str,
        domain: Optional["DomainAnalysis"],
        labeled_examples: list[dict],
        seed_data: dict[str, list[str]],
        user_prompt_template: Optional[str],
    ) -> list[GeneratedAttribute]:
        """Build generated attributes for augmentation mode."""
        seed_context = self._format_seed_context(seed_data)
        template_vars = {
            "task_description": task_description,
            "system_prompt": system_prompt,
            "domain": domain,
            "examples_json": json.dumps(labeled_examples[:5], indent=2),
            "seed_context": seed_context,
            "user_prompt_template": user_prompt_template or "",
        }

        question_prompt = load_prompt("synth/augmentation_question", **template_vars)
        answer_prompt = load_prompt("synth/augmentation_answer", **template_vars)

        question_attr = GeneratedAttribute(
            id=f"{SynthPlaceholder.QUESTION}_raw",
            instruction_messages=[
                TextMessage(role=Role.SYSTEM, content=system_prompt),
                TextMessage(role=Role.USER, content=question_prompt),
            ],
            postprocessing_params=GeneratedAttributePostprocessingParams(
                id=SynthPlaceholder.QUESTION,
                cut_prefix="Question:",
                strip_whitespace=True,
            ),
        )

        answer_attr = GeneratedAttribute(
            id=f"{SynthPlaceholder.ANSWER}_raw",
            instruction_messages=[
                TextMessage(role=Role.SYSTEM, content=system_prompt),
                TextMessage(role=Role.USER, content=answer_prompt),
            ],
            postprocessing_params=GeneratedAttributePostprocessingParams(
                id=SynthPlaceholder.ANSWER,
                cut_prefix="Answer:",
                strip_whitespace=True,
            ),
        )

        return [question_attr, answer_attr]

    def _build_teacher_labeling_attributes(
        self,
        task_description: str,
        system_prompt: str,
        domain: Optional["DomainAnalysis"],
        user_prompt_template: Optional[str],
        template_mapping: Optional[dict[str, str]],
        seed_data: dict[str, list[str]],
    ) -> list[GeneratedAttribute]:
        """Build generated attributes for teacher labeling mode (answers only)."""
        seed_context = self._format_seed_context(seed_data)
        template_vars = {
            "task_description": task_description,
            "system_prompt": system_prompt,
            "domain": domain,
            "user_prompt_template": user_prompt_template or "",
            "template_mapping": template_mapping or {},
            "seed_context": seed_context,
        }

        answer_prompt = load_prompt("synth/teacher_labeling_answer", **template_vars)

        answer_attr = GeneratedAttribute(
            id=f"{SynthPlaceholder.ANSWER}_raw",
            instruction_messages=[
                TextMessage(role=Role.SYSTEM, content=system_prompt),
                TextMessage(role=Role.USER, content=answer_prompt),
            ],
            postprocessing_params=GeneratedAttributePostprocessingParams(
                id=SynthPlaceholder.ANSWER,
                cut_prefix="Answer:",
                strip_whitespace=True,
            ),
        )

        return [answer_attr]

    def _build_seed_sampled_attributes(
        self, seed_data: dict[str, list[str]]
    ) -> list[SampledAttribute]:
        """Convert seed data into sampled attributes for diversity."""
        sampled: list[SampledAttribute] = []

        for col, values in seed_data.items():
            if not values:
                continue
            seed_id = self._normalize_seed_id(col)
            possible_values = []
            for idx, val in enumerate(values[:8]):
                val_str = str(val)
                possible_values.append(
                    SampledAttributeValue(
                        id=f"{seed_id}_value_{idx}",
                        name=val_str,
                        description=val_str,
                    )
                )

            sampled.append(
                SampledAttribute(
                    id=seed_id,
                    name=col,
                    description=f"Seed attribute from column '{col}'",
                    possible_values=possible_values,
                )
            )

        return sampled

    def _format_seed_context(self, seed_data: dict[str, list[str]]) -> str:
        """Format seed data into a readable string for prompts."""
        if not seed_data:
            return ""

        lines = []
        for col, values in seed_data.items():
            preview = ", ".join(str(v) for v in values[:5])
            seed_id = self._normalize_seed_id(col)
            lines.append(f"{col} ({{{seed_id}}}): {preview}")

        return "\n".join(lines)

    def _normalize_seed_id(self, name: str) -> str:
        """Normalize a seed column name into a safe attribute id."""
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip())
        safe = safe.strip("_") or "seed"
        return f"seed_{safe}"

    def _build_conversation_transform(
        self,
        system_prompt: Optional[str] = None,
    ) -> list[TransformedAttribute]:
        """Build transformation for Q&A to conversation format."""
        messages = []

        if system_prompt:
            messages.append(TextMessage(role=Role.SYSTEM, content=system_prompt))

        messages.append(
            TextMessage(role=Role.USER, content=f"{{{SynthPlaceholder.QUESTION}}}")
        )
        messages.append(
            TextMessage(role=Role.ASSISTANT, content=f"{{{SynthPlaceholder.ANSWER}}}")
        )

        return [
            TransformedAttribute(
                id=SynthPlaceholder.CONVERSATION,
                transformation_strategy=TransformationStrategy(
                    type=TransformationType.CHAT,
                    chat_transform=TextConversation(messages=messages),
                ),
            )
        ]


class JudgeConfigBuilder(ConfigBuilder):
    """Build JudgeConfig from customer data using LLM-powered prompts.

    This builder creates judge configurations for evaluating data quality,
    with task-specific evaluation criteria loaded from Jinja templates.

    Example:
        >>> from oumi.onboarding import JudgeConfigBuilder
        >>> builder = JudgeConfigBuilder()
        >>> config = builder.build(
        ...     schema=schema,
        ...     judge_name="accuracy",
        ...     criteria="Evaluate factual accuracy",
        ...     task_type="qa",
        ...     task_description="Answer customer questions",
        ... )
    """

    def build(
        self,
        schema: DataSchema,
        judge_name: str,
        criteria: str,
        task_type: str = "generation",
        task_description: str = "",
        domain: Optional["DomainAnalysis"] = None,
        output_format: Optional[str] = None,
        judgment_type: JudgeOutputType = JudgeOutputType.BOOL,
        llm_analyzer: Optional[Any] = None,
    ) -> JudgeConfig:
        """Build a JudgeConfig with LLM-generated evaluation prompts.

        Args:
            schema: The analyzed data schema.
            judge_name: A short name for this judge (e.g., "accuracy", "helpfulness").
            criteria: The specific evaluation criteria description.
            task_type: Task type (extraction, classification, qa, etc.).
            task_description: User's task description.
            domain: Optional DomainAnalysis for domain-specific evaluation.
            output_format: Expected output format for evaluation.
            judgment_type: The type of judgment output (BOOL, FLOAT, or LIKERT).
            llm_analyzer: Optional LLMAnalyzer for generating custom prompts.

        Returns:
            A configured JudgeConfig with generated evaluation prompts.
        """
        # Generate custom judge prompt using LLM if available
        if llm_analyzer:
            try:
                template_vars = {
                    "task_description": task_description,
                    "criterion_name": judge_name,
                    "criterion_description": criteria,
                    "task_type": task_type,
                    "domain": domain.domain if domain else None,
                }

                generation_prompt = load_prompt("generate_judge_prompt", **template_vars)
                prompt_template = llm_analyzer._invoke(generation_prompt).strip()

                # Validate that required placeholders are present
                if "{input}" not in prompt_template or "{output}" not in prompt_template:
                    logger.warning(
                        f"Generated judge prompt missing required placeholders, using fallback"
                    )
                    prompt_template = self._fallback_prompt_template(
                        judge_name, criteria, task_description
                    )
            except Exception as e:
                logger.warning(f"Failed to generate judge prompt: {e}, using fallback")
                prompt_template = self._fallback_prompt_template(
                    judge_name, criteria, task_description
                )
        else:
            # No LLM available, use simple fallback
            prompt_template = self._fallback_prompt_template(
                judge_name, criteria, task_description
            )

        system_instruction = f"""You are an expert evaluator assessing '{judge_name}'.

Your task is to objectively evaluate content based on the provided criteria.
Be thorough, fair, and consistent in your evaluations."""

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

    def _fallback_prompt_template(
        self, judge_name: str, criteria: str, task_description: str
    ) -> str:
        """Generate a simple fallback prompt when LLM generation fails.

        Args:
            judge_name: Name of the criterion being evaluated.
            criteria: Description of what to evaluate.
            task_description: Description of the task.

        Returns:
            A simple but functional judge prompt template.
        """
        return f"""You are evaluating: {judge_name}

## Task Context
{task_description}

## Evaluation Criterion
{criteria}

## What to Check
Based on the criterion above, evaluate whether the output meets the requirements for production quality.

## Content to Evaluate
Input: {{input}}
Output: {{output}}

Does the output satisfy this criterion? Provide a clear yes/no assessment."""


class TrainConfigBuilder(ConfigBuilder):
    """Build TrainingConfig from customer data.

    This builder creates training configurations for fine-tuning models
    on customer data or synthetically generated data.

    Example:
        >>> from oumi.onboarding import TrainConfigBuilder
        >>> builder = TrainConfigBuilder()
        >>> config = builder.build(
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

    def build(
        self,
        schema: DataSchema,
        base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
        use_lora: bool = True,
        output_dir: str = "./output/training",
        max_steps: int = 1000,
        **kwargs,
    ) -> TrainingConfig:
        """Build a TrainingConfig from analyzed data.

        Args:
            schema: The analyzed data schema (usually from synth output).
            base_model: Base model to fine-tune.
            use_lora: Whether to use LoRA for efficient fine-tuning.
            output_dir: Directory to save model outputs.
            max_steps: Maximum training steps.
            **kwargs: Additional training parameters.

        Returns:
            A configured TrainingConfig.
        """
        return self.from_data_path(
            schema.source_path if schema else kwargs.get("data_path", ""),
            base_model=base_model,
            use_lora=use_lora,
            output_dir=output_dir,
            max_steps=max_steps,
            **kwargs,
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
        **kwargs,
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
            **kwargs: Additional parameters.

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

        data = DataParams(
            train=DatasetSplitParams(
                datasets=[DatasetParams(dataset_path=data_path)],
                collator_name="text_sft",
            )
        )

        model = ModelParams(
            model_name=base_model,
            model_max_length=4096,
            torch_dtype_str="float32",
            attn_implementation="sdpa",
        )

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
