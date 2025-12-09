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

"""LLM-based analysis for customer data onboarding.

This module uses Oumi inference engines to analyze customer data samples and
automatically infer:
- Domain and industry context
- Terminology and key concepts
- Quality signals and common issues
- System prompts and evaluation criteria

Example:
    >>> from oumi.onboarding import DataAnalyzer, LLMAnalyzer
    >>> analyzer = DataAnalyzer()
    >>> schema = analyzer.analyze("./customer_data.csv")
    >>> llm = LLMAnalyzer()
    >>> domain = llm.analyze(schema)
    >>> print(f"Domain: {domain.domain}")
    >>> print(f"Terminology: {domain.terminology}")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.configs.inference_config import InferenceEngineType
from oumi.core.configs.params.remote_params import RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.onboarding.data_analyzer import DataSchema
from oumi.utils.logging import logger


@dataclass
class DomainAnalysis:
    """Analysis of the semantic domain from customer data.

    This dataclass captures what the LLM infers about the customer's data,
    including the industry/domain, key terminology, and quality patterns.
    """

    domain: str
    """The identified domain or industry (e.g., 'plumbing services', 'medical Q&A')."""

    description: str
    """A brief description of what the data represents."""

    terminology: list[str] = field(default_factory=list)
    """Domain-specific terms found in the data."""

    quality_signals: list[str] = field(default_factory=list)
    """Indicators of high-quality data in this domain."""

    common_issues: list[str] = field(default_factory=list)
    """Common problems or issues to watch for."""

    suggested_persona: str = ""
    """Suggested system prompt persona for this domain."""

    data_purpose: str = ""
    """What the data appears to be used for."""


@dataclass
class InferredConfig:
    """Configuration elements inferred by the LLM.

    This dataclass contains the LLM-generated configuration suggestions
    that can be used to build synthesis, judge, or training configs.
    """

    system_prompt: str
    """Generated system prompt for the task."""

    instruction_template: str
    """Template for user instructions with {placeholders}."""

    output_format: dict[str, Any] = field(default_factory=dict)
    """Expected output format structure."""

    evaluation_criteria: list[dict[str, Any]] = field(default_factory=list)
    """List of evaluation criteria with descriptions and weights."""

    field_mappings: dict[str, str] = field(default_factory=dict)
    """Suggested mappings from data columns to config placeholders."""

    postprocessing: dict[str, Any] = field(default_factory=dict)
    """Suggested postprocessing parameters."""


@dataclass
class FileContext:
    """Context about a single file for multi-file analysis."""

    path: str
    """Path to the file."""

    role: str
    """Role of the file: primary, reference, rules, examples, context."""

    schema: Optional[DataSchema] = None
    """Analyzed schema if available."""

    summary: str = ""
    """Brief summary of the file contents."""


@dataclass
class MultiFileAnalysis:
    """Analysis of relationships between multiple files."""

    primary_purpose: str
    """The main purpose of the data collection."""

    file_roles: dict[str, str] = field(default_factory=dict)
    """Mapping of filename to role."""

    relationships: list[dict[str, str]] = field(default_factory=list)
    """Relationships between files (e.g., lookup, validate, join)."""

    extracted_rules: list[str] = field(default_factory=list)
    """Rules extracted from rules documents."""

    quality_patterns: dict[str, list[str]] = field(default_factory=dict)
    """Patterns for good vs bad data."""

    suggested_pipeline: dict[str, Any] = field(default_factory=dict)
    """Suggested pipeline configuration."""


class LLMAnalyzer:
    """Use an LLM to analyze customer data and infer configuration.

    This class uses Oumi inference engines to analyze sample data and automatically
    generate domain-specific configurations for synthesis, evaluation, and training.

    Supports multiple inference engines: ANTHROPIC, OPENAI, VLLM, etc.

    Example:
        >>> llm = LLMAnalyzer()
        >>> domain = llm.analyze(schema)
        >>> synth_config = llm.infer_synth_config(schema, goal="qa", domain=domain)

        >>> # Use a different engine
        >>> llm = LLMAnalyzer(engine="OPENAI", model="gpt-4o")
    """

    # Default models per engine (prefer fast/cheap models for analysis tasks)
    DEFAULT_MODELS: dict[str, str] = {
        "ANTHROPIC": "claude-haiku-4-5",
        "OPENAI": "gpt-5-mini",
        "DEEPSEEK": "deepseek-chat",
        "TOGETHER": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    }

    # API key environment variables per engine
    API_KEY_ENV_VARS: dict[str, str] = {
        "ANTHROPIC": "ANTHROPIC_API_KEY",
        "OPENAI": "OPENAI_API_KEY",
        "DEEPSEEK": "DEEPSEEK_API_KEY",
        "TOGETHER": "TOGETHER_API_KEY",
    }

    def __init__(
        self,
        model: Optional[str] = None,
        engine: Literal["ANTHROPIC", "OPENAI", "DEEPSEEK", "TOGETHER"] = "ANTHROPIC",
        api_key: Optional[str] = None,
        api_key_env_var: Optional[str] = None,
        num_workers: int = 1,
    ):
        """Initialize the LLM analyzer.

        Args:
            model: Model name to use. If not provided, uses default for the engine.
            engine: Inference engine to use (ANTHROPIC, OPENAI, DEEPSEEK, TOGETHER).
            api_key: Optional API key. If not provided, uses environment variable.
            api_key_env_var: Environment variable name for the API key.
                If not provided, uses default for the engine.
            num_workers: Number of concurrent workers for inference.
        """
        self.engine_type = engine
        self.model = model or self.DEFAULT_MODELS.get(engine, "claude-haiku-4-5")
        self._api_key = api_key
        self._api_key_env_var = api_key_env_var or self.API_KEY_ENV_VARS.get(
            engine, "ANTHROPIC_API_KEY"
        )
        self._num_workers = num_workers
        self._inference_engine = None

    def _get_engine_type(self) -> InferenceEngineType:
        """Convert string engine type to InferenceEngineType enum."""
        engine_map = {
            "ANTHROPIC": InferenceEngineType.ANTHROPIC,
            "OPENAI": InferenceEngineType.OPENAI,
            "DEEPSEEK": InferenceEngineType.DEEPSEEK,
            "TOGETHER": InferenceEngineType.TOGETHER,
        }
        return engine_map.get(self.engine_type, InferenceEngineType.ANTHROPIC)

    @property
    def inference_engine(self):
        """Lazy-load the Oumi inference engine."""
        if self._inference_engine is None:
            from oumi.builders.inference_engines import build_inference_engine

            model_params = ModelParams(model_name=self.model)
            generation_params = GenerationParams(
                max_new_tokens=4096,
                temperature=1.0,
            )
            remote_params = RemoteParams(
                api_key=self._api_key,
                api_key_env_varname=self._api_key_env_var,
                num_workers=self._num_workers,
                max_retries=3,
            )

            self._inference_engine = build_inference_engine(
                engine_type=self._get_engine_type(),
                model_params=model_params,
                generation_params=generation_params,
                remote_params=remote_params,
            )

        return self._inference_engine

    def _invoke(self, prompt: str, system: str = "") -> str:
        """Invoke the LLM with a prompt using Oumi inference engine.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.

        Returns:
            The LLM response text.
        """
        # Build conversation
        messages = []
        if system:
            messages.append(Message(content=system, role=Role.SYSTEM))
        messages.append(Message(content=prompt, role=Role.USER))

        conversation = Conversation(messages=messages)

        logger.debug(f"LLM Request: {conversation}")

        # Create inference config
        inference_config = InferenceConfig(
            model=ModelParams(model_name=self.model),
            generation=GenerationParams(
                max_new_tokens=4096,
                temperature=1.0,
            ),
            engine=self._get_engine_type(),
        )

        try:
            # Run inference
            results = self.inference_engine.infer(
                input=[conversation],
                inference_config=inference_config,
            )

            logger.debug(f"LLM Response: {results}")

            # Extract response text from the last message
            if results and results[0].messages:
                last_message = results[0].messages[-1]
                if last_message.role == Role.ASSISTANT:
                    return str(last_message.content)

            logger.warning("No response from inference engine")
            return ""

        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise

    def _invoke_json(self, prompt: str, system: str = "") -> dict[str, Any]:
        """Invoke the LLM and parse the response as JSON.

        Args:
            prompt: The user prompt (should request JSON output).
            system: Optional system prompt.

        Returns:
            Parsed JSON response as a dictionary.
        """
        response = self._invoke(prompt, system)

        # Try to extract JSON from the response
        try:
            # First, try direct JSON parsing
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find a JSON object in the response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from LLM response: {response[:200]}...")
        return {}

    def _prepare_samples(self, schema: DataSchema) -> dict[str, Any]:
        """Prepare optimal samples for LLM analysis.

        Selects the most informative samples to send to the LLM while
        keeping the context size reasonable.

        Args:
            schema: The data schema to prepare samples from.

        Returns:
            Dictionary with sample information for the LLM.
        """
        # Basic column info
        columns_info = [
            {
                "name": col.name,
                "type": col.dtype,
                "is_text": col.is_text,
                "is_categorical": col.is_categorical,
                "avg_length": col.avg_length,
                "sample_values": col.sample_values[:2] if col.sample_values else [],
            }
            for col in schema.columns
        ]

        # Select diverse text samples (longest columns for richest context)
        text_samples = {}
        text_cols = sorted(
            [c for c in schema.columns if c.is_text],
            key=lambda c: c.avg_length or 0,
            reverse=True,
        )
        for col in text_cols[:3]:
            if schema.sample_rows:
                text_samples[col.name] = schema.sample_rows[0].get(col.name, "")

        # Categorical values
        categories = {
            col.name: col.sample_values
            for col in schema.columns
            if col.is_categorical and col.sample_values
        }

        # Full example rows (limit size)
        example_rows = schema.sample_rows[:2] if schema.sample_rows else []

        return {
            "columns": columns_info,
            "text_samples": text_samples,
            "categories": categories,
            "example_rows": example_rows,
            "row_count": schema.row_count,
            "format": schema.detected_format,
        }

    def analyze(self, schema: DataSchema) -> DomainAnalysis:
        """Analyze data samples to understand the semantic domain.

        Uses the LLM to examine sample data and infer:
        - The domain/industry
        - Key terminology
        - Quality signals
        - Common issues

        Args:
            schema: The analyzed data schema.

        Returns:
            DomainAnalysis with inferred domain information.
        """
        samples = self._prepare_samples(schema)

        # Handle word documents differently
        if schema.detected_format == "word" and schema.raw_text:
            doc_size = len(schema.raw_text)
            prompt = f"""Analyze this document to understand what the customer wants to build.

DOCUMENT INFO:
- Size: {doc_size:,} characters
- Sections/paragraphs: {schema.row_count}

DOCUMENT CONTENT:
{schema.raw_text}

This document may be one of:
1. **Use case specification**: Contains system prompts, output schemas, prompt templates, or examples of what the customer wants their model to do
2. **Reference material**: Background knowledge, FAQs, or documentation the model should use
3. **Training data**: Examples of inputs and outputs for training

Analyze this document and return a JSON object with:
{{
    "domain": "the industry or field (e.g., 'B2B sales', 'customer support', 'healthcare')",
    "description": "what this document is about and what the customer wants to achieve",
    "terminology": ["key", "domain", "specific", "terms"],
    "quality_signals": ["what indicates a good output for this use case"],
    "common_issues": ["potential problems or edge cases to handle"],
    "suggested_persona": "A system prompt persona based on what the document describes",
    "data_purpose": "specification/reference/training - what role this document plays"
}}

Return ONLY the JSON object, no other text."""
        else:
            prompt = f"""Analyze this data sample to understand the semantic domain.

DATA OVERVIEW:
- Format: {samples["format"]}
- Row count: {samples["row_count"]}
- Columns: {json.dumps(samples["columns"], indent=2)}

EXAMPLE ROWS:
{json.dumps(samples["example_rows"], indent=2)}

TEXT SAMPLES (longest text columns):
{json.dumps(samples["text_samples"], indent=2)}

CATEGORICAL VALUES:
{json.dumps(samples["categories"], indent=2)}

Analyze this data and return a JSON object with:
{{
    "domain": "the industry or field (e.g., 'plumbing services', 'customer support')",
    "description": "brief description of what this data represents",
    "terminology": ["key", "domain", "specific", "terms", "found in the data"],
    "quality_signals": ["what indicates good/high-quality data in this domain"],
    "common_issues": ["problems or issues to watch for in this type of data"],
    "suggested_persona": "A system prompt persona for an expert in this domain",
    "data_purpose": "what this data appears to be used for (training, validation, etc.)"
}}

Return ONLY the JSON object, no other text."""

        # Use appropriate system prompt based on content type
        if schema.detected_format == "word" and schema.raw_text:
            system = (
                "You are an ML engineer helping customers build AI models. "
                "Analyze customer-provided documents to understand what they want to build. "
                "Look for explicit specifications like system prompts, output schemas, and examples."
            )
        else:
            system = (
                "You are a data analysis expert. Analyze the provided data samples "
                "to understand the semantic domain, terminology, and data quality patterns. "
                "Be specific and practical in your analysis."
            )

        result = self._invoke_json(prompt, system)

        return DomainAnalysis(
            domain=result.get("domain", "unknown"),
            description=result.get("description", ""),
            terminology=result.get("terminology", []),
            quality_signals=result.get("quality_signals", []),
            common_issues=result.get("common_issues", []),
            suggested_persona=result.get("suggested_persona", ""),
            data_purpose=result.get("data_purpose", ""),
        )

    def infer_synth_config(
        self,
        schema: DataSchema,
        goal: str,
        domain: Optional[DomainAnalysis] = None,
    ) -> InferredConfig:
        """Generate synthesis configuration from domain analysis.

        Uses the LLM to create domain-specific prompts and configuration
        for synthetic data generation.

        Args:
            schema: The analyzed data schema.
            goal: The synthesis goal (qa, conversation, augmentation, instruction).
            domain: Optional pre-computed domain analysis.

        Returns:
            InferredConfig with generated configuration elements.
        """
        if domain is None:
            domain = self.analyze(schema)

        samples = self._prepare_samples(schema)
        column_names = [col.name for col in schema.columns]

        prompt = f"""Generate synthesis configuration for creating training data.

DOMAIN CONTEXT:
- Domain: {domain.domain}
- Description: {domain.description}
- Terminology: {domain.terminology}
- Quality signals: {domain.quality_signals}

DATA STRUCTURE:
- Columns: {column_names}
- Example row: {json.dumps(samples["example_rows"][0] if samples["example_rows"] else {}, indent=2)}

SYNTHESIS GOAL: {goal}
- qa: Generate question-answer pairs
- conversation: Generate multi-turn dialogues
- augmentation: Create variations of existing data
- instruction: Generate instruction-following data

Create synthesis prompts that are specific to the {domain.domain} domain.
Use the actual column names as placeholders in the format {{column_name}} in instruction_template.

Return a JSON object with:
{{
    "system_prompt": "Expert persona system prompt specific to this domain",
    "instruction_template": "User instruction with {{column_name}} placeholders for available columns",
    "output_format": {{"field1": "description", "field2": "description"}},
    "field_mappings": {{"column_name": "column_name"}},
    "postprocessing": {{
        "cut_prefix": "prefix to remove from output or null",
        "strip_whitespace": true
    }}
}}

IMPORTANT: In field_mappings, use plain column names WITHOUT braces.
Example: {{"LINE_ID": "LINE_ID", "DESCRIPTION": "DESCRIPTION"}} NOT {{"LINE_ID": "{{LINE_ID}}"}}

Make the prompts domain-specific using the terminology: {domain.terminology}

Return ONLY the JSON object, no other text."""

        system = (
            "You are an expert at creating training data synthesis configurations. "
            "Generate high-quality, domain-specific prompts that will produce "
            "realistic and useful training data."
        )

        result = self._invoke_json(prompt, system)

        return InferredConfig(
            system_prompt=result.get("system_prompt", ""),
            instruction_template=result.get("instruction_template", ""),
            output_format=result.get("output_format", {}),
            evaluation_criteria=[],  # Not relevant for synth
            field_mappings=result.get("field_mappings", {}),
            postprocessing=result.get("postprocessing", {}),
        )

    def infer_judge_config(
        self,
        schema: DataSchema,
        judge_type: str,
        domain: Optional[DomainAnalysis] = None,
    ) -> InferredConfig:
        """Generate judge/evaluation configuration from domain analysis.

        Uses the LLM to create domain-specific evaluation criteria
        and scoring rubrics.

        Args:
            schema: The analyzed data schema.
            judge_type: Type of evaluation (generic, compliance, relevance, safety).
            domain: Optional pre-computed domain analysis.

        Returns:
            InferredConfig with generated evaluation configuration.
        """
        if domain is None:
            domain = self.analyze(schema)

        samples = self._prepare_samples(schema)

        prompt = f"""Generate evaluation/judge configuration for assessing data quality.

DOMAIN CONTEXT:
- Domain: {domain.domain}
- Description: {domain.description}
- Terminology: {domain.terminology}
- Quality signals: {domain.quality_signals}
- Common issues: {domain.common_issues}

EXAMPLE DATA:
{json.dumps(samples["example_rows"][0] if samples["example_rows"] else {}, indent=2)}

JUDGE TYPE: {judge_type}
- generic: General quality assessment
- compliance: Check adherence to rules/policies
- relevance: Evaluate if responses address the question
- safety: Content safety evaluation
- groundedness: Check if claims are supported by context

Create evaluation criteria specific to the {domain.domain} domain.

Return a JSON object with:
{{
    "system_prompt": "Expert evaluator persona for this domain",
    "instruction_template": "Evaluation prompt template with {{placeholders}}",
    "evaluation_criteria": [
        {{
            "name": "criterion_name",
            "description": "What this criterion measures",
            "weight": 0.3,
            "scoring": {{"excellent": 1.0, "good": 0.7, "fair": 0.4, "poor": 0.0}}
        }}
    ],
    "field_mappings": {{"source_column": "evaluation_placeholder"}},
    "output_format": {{
        "score": "float 0-1",
        "explanation": "reasoning for the score",
        "issues": ["list of identified issues"]
    }}
}}

Use domain terminology: {domain.terminology}
Watch for common issues: {domain.common_issues}

Return ONLY the JSON object, no other text."""

        system = (
            "You are an expert at creating evaluation rubrics and quality assessment "
            "criteria. Generate specific, measurable criteria that will effectively "
            "assess data quality in the given domain."
        )

        result = self._invoke_json(prompt, system)

        return InferredConfig(
            system_prompt=result.get("system_prompt", ""),
            instruction_template=result.get("instruction_template", ""),
            output_format=result.get("output_format", {}),
            evaluation_criteria=result.get("evaluation_criteria", []),
            field_mappings=result.get("field_mappings", {}),
            postprocessing={},
        )

    def analyze_multi_file(self, file_contexts: list[FileContext]) -> MultiFileAnalysis:
        """Analyze relationships between multiple files.

        Examines multiple files together to understand how they relate
        and suggest an optimal processing pipeline.

        Args:
            file_contexts: List of FileContext objects with file information.

        Returns:
            MultiFileAnalysis with inferred relationships and pipeline.
        """
        # Build file summaries
        file_summaries = []
        for ctx in file_contexts:
            summary = {
                "path": ctx.path,
                "role": ctx.role,
            }

            if ctx.schema:
                summary["format"] = ctx.schema.detected_format
                summary["row_count"] = ctx.schema.row_count
                summary["columns"] = [c.name for c in ctx.schema.columns]
                if ctx.schema.sample_rows:
                    summary["sample"] = ctx.schema.sample_rows[0]
                if ctx.schema.raw_text:
                    summary["text_preview"] = ctx.schema.raw_text[:500]

            file_summaries.append(summary)

        prompt = f"""Analyze relationships between these files for a data processing task.

FILES PROVIDED:
{json.dumps(file_summaries, indent=2)}

FILE ROLES:
- primary: Main data to process (synthesize from, evaluate, train on)
- reference: Valid values to match against (lookup tables, catalogs)
- rules: Guidelines and evaluation criteria
- examples: Labeled good/bad samples
- context: Background information

ANALYSIS TASKS:
1. What is the PRIMARY purpose of this data collection?
2. How do the files relate to each other?
3. What lookup/validation relationships exist?
4. What rules or criteria are described in documents?
5. What patterns distinguish good vs bad examples?
6. What would be the ideal processing pipeline?

Return a JSON object with:
{{
    "primary_purpose": "description of what user is trying to accomplish",
    "file_roles": {{
        "filename": "primary|reference|rules|examples|context"
    }},
    "relationships": [
        {{"from_file": "file1", "from_col": "col1", "to_file": "file2", "to_col": "col2", "type": "lookup|validate|join"}}
    ],
    "extracted_rules": ["rule 1 from documents", "rule 2"],
    "quality_patterns": {{
        "good": ["patterns indicating good data"],
        "bad": ["patterns indicating bad data"]
    }},
    "suggested_pipeline": {{
        "synth": {{"enabled": true, "goal": "qa|conversation|augmentation", "source_file": "..."}},
        "judge": {{"enabled": true, "type": "compliance|relevance|...", "criteria_source": "..."}},
        "train": {{"enabled": true, "task": "classification|generation", "labels": ["label1", "label2"]}}
    }}
}}

Return ONLY the JSON object, no other text."""

        system = (
            "You are a data pipeline architect. Analyze the provided files "
            "to understand their relationships and suggest an optimal processing "
            "pipeline for the user's needs."
        )

        result = self._invoke_json(prompt, system)

        return MultiFileAnalysis(
            primary_purpose=result.get("primary_purpose", ""),
            file_roles=result.get("file_roles", {}),
            relationships=result.get("relationships", []),
            extracted_rules=result.get("extracted_rules", []),
            quality_patterns=result.get("quality_patterns", {}),
            suggested_pipeline=result.get("suggested_pipeline", {}),
        )

    def suggest_improvements(
        self,
        schema: DataSchema,
        current_config: dict[str, Any],
        domain: Optional[DomainAnalysis] = None,
    ) -> dict[str, Any]:
        """Suggest improvements to an existing configuration.

        Analyzes the current configuration and suggests domain-specific
        improvements based on the data and domain analysis.

        Args:
            schema: The analyzed data schema.
            current_config: The current configuration as a dictionary.
            domain: Optional pre-computed domain analysis.

        Returns:
            Dictionary with suggested improvements.
        """
        if domain is None:
            domain = self.analyze(schema)

        prompt = f"""Review this configuration and suggest improvements.

DOMAIN CONTEXT:
- Domain: {domain.domain}
- Terminology: {domain.terminology}
- Quality signals: {domain.quality_signals}

CURRENT CONFIGURATION:
{json.dumps(current_config, indent=2)}

DATA SAMPLE:
{json.dumps(self._prepare_samples(schema)["example_rows"][0] if self._prepare_samples(schema)["example_rows"] else {}, indent=2)}

Suggest improvements to make this configuration more effective for the {domain.domain} domain.

Return a JSON object with:
{{
    "suggestions": [
        {{
            "field": "path.to.config.field",
            "current_value": "current value",
            "suggested_value": "improved value",
            "reason": "why this improvement helps"
        }}
    ],
    "missing_elements": ["elements that should be added"],
    "overall_assessment": "brief assessment of the config quality"
}}

Return ONLY the JSON object, no other text."""

        system = (
            "You are a configuration optimization expert. Analyze the provided "
            "configuration and suggest specific, actionable improvements."
        )

        return self._invoke_json(prompt, system)
