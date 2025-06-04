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

from typing import Optional

from typing_extensions import override

from oumi.core.configs.judge_config import JudgeConfig, JudgeOutputType
from oumi.core.inference import BaseInferenceEngine
from oumi.judges.base_judge import (
    BaseJudge,
    JudgeOutputField,
    JudgeResponseFormat,
)
from oumi.utils.logging import logger

# Field names for judge outputs.
EXPLANATION_KEY = "explanation"
JUDGMENT_KEY = "judgment"

# Prompt suffixes for XML, JSON, RAW formats.
XML_SUFFIX = (
    "\nProvide your response in XML format only. Include your judgment enclosed within "
    f"<{JUDGMENT_KEY}> and </{JUDGMENT_KEY}> tags. Do not include any text outside "
    "the XML. Ensure that all tags are properly closed and that the XML is well-formed."
)
XML_SUFFIX_WITH_EXPLANATION = (
    "\nProvide your response in XML format only. Begin with an explanation justifying "
    f"your judgment, enclosed within <{EXPLANATION_KEY}> and </{EXPLANATION_KEY}> tags."
    f" Follow this with your judgment, enclosed within <{JUDGMENT_KEY}> and "
    f"</{JUDGMENT_KEY}> tags. Do not include any text outside the XML. "
    "Ensure that all tags are properly closed and that the XML is well-formed."
)
JSON_SUFFIX = (
    "\nProvide your response in JSON format only. Include your judgment as the value "
    f"of a single key named '{JUDGMENT_KEY}'. Do not include any text outside the JSON."
    " Ensure the JSON is properly formatted and valid."
)
JSON_SUFFIX_WITH_EXPLANATION = (
    "\nProvide your response in JSON format only. Begin with an explanation justifying "
    f"your judgment, using the key '{EXPLANATION_KEY}'. Then include your judgment "
    f"using the key '{JUDGMENT_KEY}'. Do not include any text outside the JSON. "
    "Ensure the JSON is properly formatted and valid."
)
RAW_SUFFIX_WITH_EXPLANATION = "\nExplain your reasoning before providing your judgment."


class OumiJudge(BaseJudge):
    """Judge class for evaluating outputs based on a given configuration."""

    def __init__(
        self,
        config: JudgeConfig,
        inference_engine: Optional[BaseInferenceEngine] = None,
    ):
        """Initialize the Judge."""
        self._config = config
        if not config.prompt_template:
            raise ValueError(
                "prompt_template must be specified in the judge configuration."
            )

        # Generate an inference engine if not provided
        if inference_engine is None:
            logger.debug("Initializing a new inference engine.")
            inference_engine = self._create_inference_engine(config)

        # Create output fields based on configuration
        output_fields = [self._create_judgment_output_field(config)]

        # Add explanation field, if explanations are enabled
        if config.include_explanation:
            output_fields.append(self._create_explanation_output_field())

        super().__init__(
            prompt_template=config.prompt_template,
            response_format=config.response_format,
            output_fields=output_fields,
            inference_engine=inference_engine,
        )

    @override
    def build_judgement_prompt(self, judge_input: dict[str, str]) -> str:
        """Generate judge prompts using the template."""
        prompt_content = super().build_judgement_prompt(judge_input)

        # Append format-specific instructions to the prompt
        if format_suffix := self._get_format_suffix():
            prompt_content += format_suffix

        return prompt_content

    def _get_format_suffix(self) -> str:
        """Get the appropriate format suffix based on response format and explanation.

        Returns:
            Format-specific instruction suffix to append to prompts
        """
        response_format = self._config.response_format
        include_explanation = self._config.include_explanation

        if response_format == JudgeResponseFormat.XML:
            return XML_SUFFIX_WITH_EXPLANATION if include_explanation else XML_SUFFIX
        elif response_format == JudgeResponseFormat.JSON:
            return JSON_SUFFIX_WITH_EXPLANATION if include_explanation else JSON_SUFFIX
        elif response_format == JudgeResponseFormat.RAW:
            return RAW_SUFFIX_WITH_EXPLANATION if include_explanation else ""
        else:
            return ""

    def _create_judgment_output_field(self, config: JudgeConfig) -> JudgeOutputField:
        """Create the main judgment output field."""
        return JudgeOutputField(
            field_key=JUDGMENT_KEY,
            field_type=config.judgment_type,
            field_scores=config.judgment_scores,
        )

    def _create_explanation_output_field(self) -> JudgeOutputField:
        """Create the explanation output field."""
        return JudgeOutputField(
            field_key=EXPLANATION_KEY,
            field_type=JudgeOutputType.TEXT,
            field_scores=None,
        )

    def _create_inference_engine(self, config: JudgeConfig) -> BaseInferenceEngine:
        """Create the inference engine based on the provided configuration."""
        from oumi.builders.inference_engines import build_inference_engine

        return build_inference_engine(
            engine_type=config.engine,
            model_params=config.model,
            remote_params=config.remote_params,
            generation_params=config.generation,
        )
