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

# Field name constants (for judge outputs).
EXPLANATION_KEY = "explanation"
JUDGEMENT_KEY = "judgement"

# Prompt suffixes for XML, JSON, RAW formats.
XML_SUFFIX = (
    "\nProvide your response in XML format only. Include your judgment enclosed within "
    f"<{JUDGEMENT_KEY}> and </{JUDGEMENT_KEY}> tags. Do not include any text outside "
    "the XML. Ensure that all tags are properly closed and that the XML is well-formed."
)
XML_SUFFIX_WITH_EXPLANATION = (
    "\nProvide your response in XML format only. Begin with an explanation justifying "
    f"your judgment, enclosed within <{EXPLANATION_KEY}> and </{EXPLANATION_KEY}> tags."
    f" Follow this with your judgment, enclosed within <{JUDGEMENT_KEY}> and "
    f"</{JUDGEMENT_KEY}> tags. Do not include any text outside the XML. "
    "Ensure that all tags are properly closed and that the XML is well-formed."
)
JSON_SUFFIX = (
    "\nProvide your response in JSON format only. Include your judgment as the value of "
    "a single key named '{JUDGEMENT_KEY}'. Do not include any text outside the JSON. "
    "Ensure the JSON is properly formatted and valid."
)
JSON_SUFFIX_WITH_EXPLANATION = (
    "\nProvide your response in JSON format only. Begin with an explanation justifying "
    f"your judgment, using the key '{EXPLANATION_KEY}'. Then include your judgment "
    "using the key '{JUDGEMENT_KEY}'. Do not include any text outside the JSON. "
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

        if inference_engine is None:
            logger.debug("Initializing inference engine.")
            inference_engine = self._create_inference_engine(config)

        output_fields = [
            JudgeOutputField(
                field_key=JUDGEMENT_KEY,
                field_type=config.judgment_type,
                field_scores=config.judgment_scores,
            ),
            JudgeOutputField(
                field_key=EXPLANATION_KEY,
                field_type=JudgeOutputType.TEXT,
                field_scores=None,
            ),
        ]

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

        # Add instructions based on the response format
        if self._config.response_format == JudgeResponseFormat.XML:
            if self._config.include_explanation:
                prompt_content += XML_SUFFIX_WITH_EXPLANATION
            else:
                prompt_content += XML_SUFFIX
        elif self._config.response_format == JudgeResponseFormat.JSON:
            if self._config.include_explanation:
                prompt_content += JSON_SUFFIX_WITH_EXPLANATION
            else:
                prompt_content += JSON_SUFFIX
        elif self._config.response_format == JudgeResponseFormat.RAW:
            if self._config.include_explanation:
                prompt_content += RAW_SUFFIX_WITH_EXPLANATION

        return prompt_content

    def _create_inference_engine(self, config: JudgeConfig) -> BaseInferenceEngine:
        """Create the inference engine."""
        from oumi.builders.inference_engines import build_inference_engine

        return build_inference_engine(
            engine_type=config.engine,
            model_params=config.model,
            remote_params=config.remote_params,
            generation_params=config.generation,
        )
