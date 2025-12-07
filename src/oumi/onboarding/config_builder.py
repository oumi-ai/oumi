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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

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

SynthGoal = Literal["qa", "conversation", "augmentation", "instruction"]
JudgeType = Literal["generic", "compliance", "relevance", "safety", "groundedness"]


@dataclass
class BuilderOptions:
    """Common options for config builders."""

    model_name: str = "claude-3-5-sonnet-20240620"
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
    ) -> SynthesisConfig:
        """Build a SynthesisConfig from analyzed data.

        Args:
            schema: The analyzed data schema.
            goal: Synthesis goal ("qa", "conversation", "augmentation", "instruction").
            num_samples: Number of samples to generate.
            output_path: Optional output path for generated data.
            mappings: Optional pre-computed field mappings.

        Returns:
            A configured SynthesisConfig.
        """
        from oumi.core.configs.synthesis_config import SynthesisStrategy

        # Get field mappings if not provided
        if mappings is None:
            mappings = self.field_mapper.suggest_mappings(schema, goal)

        # Build strategy params based on goal
        strategy_params = self._build_strategy_params(schema, goal, mappings)

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
    ) -> GeneralSynthesisParams:
        """Build strategy params based on the goal."""
        params = GeneralSynthesisParams()

        # Add input data source if schema has data
        if schema.source_path and schema.detected_format in ("csv", "excel", "json", "jsonl"):
            # Build attribute map - map source columns to standard placeholders
            attribute_map = self._build_attribute_map(mappings)

            # If no mappings, try to find a good context column
            if not attribute_map and schema.columns:
                # Find the best text column to use as context
                text_cols = [c for c in schema.columns if c.is_text]
                if text_cols:
                    # Use the longest text column as context
                    best_col = max(text_cols, key=lambda c: c.avg_length or 0)
                    attribute_map = {best_col.name: "context"}
                elif schema.columns:
                    # Fallback: use first column as context
                    attribute_map = {schema.columns[0].name: "context"}

            params.input_data = [
                DatasetSource(
                    path=schema.source_path,
                    attribute_map=attribute_map if attribute_map else None,
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
