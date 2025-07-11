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

from typing import Union

from typing_extensions import override

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.judge_config_v2 import JudgeConfig
from oumi.core.configs.params.judge_params import (
    JudgeOutputType,
    JudgeParams,
    JudgeResponseFormat,
)
from oumi.core.inference import BaseInferenceEngine
from oumi.judges_v2.base_judge import (
    BaseJudge,
    JudgeOutputField,
)

# Expected field/key names in the judge's output.
EXPLANATION_KEY = "explanation"
JUDGMENT_KEY = "judgment"

# Judgment options: describing to the judge how to format its judgment.
JUDGMENT_OPTIONS_BOOL = "Your judgment should be a single word: 'Yes' or 'No'"
JUDGMENT_OPTIONS_INT = "Your judgment should be an integer value"
JUDGMENT_OPTIONS_FLOAT = "Your judgment should be a float value"
JUDGMENT_OPTIONS_ENUM_PREFIX = "Your judgment should be one of the following options: "
JUDGMENT_OPTIONS_TEXT = "Your judgment should be provided in the form of free text"

# Prompt suffix: describing to the judge how to format its response (XML, JSON, or RAW).
XML_SUFFIX = (
    "\n\nProvide your response in XML format only. Include your judgment enclosed "
    "within <{judgment_key}> and </{judgment_key}> tags. {judgment_options}Do not  "
    "include any text outside the XML. Ensure that all tags are properly closed and "
    "that the XML is well-formed."
)
XML_SUFFIX_WITH_EXPLANATION = (
    "\n\nProvide your response in XML format only. Begin with an explanation "
    "justifying your judgment, enclosed within <{explanation_key}> and "
    "</{explanation_key}> tags. Follow this with your judgment, enclosed within "
    "<{judgment_key}> and </{judgment_key}> tags. {judgment_options}Do not include any "
    "text outside the XML. Ensure that all tags are properly closed and that the XML "
    "is well-formed."
)
JSON_SUFFIX = (
    "\n\nProvide your response in JSON format only. Include your judgment as the value "
    "of a single key named '{judgment_key}'. {judgment_options}Do not include any "
    "text outside the JSON. Ensure the JSON is properly formatted and valid."
)
JSON_SUFFIX_WITH_EXPLANATION = (
    "\n\nProvide your response in JSON format only. Begin with an explanation "
    "justifying your judgment, using the key '{explanation_key}'. Then include your "
    "judgment using the key '{judgment_key}'. {judgment_options}Do not include any "
    "text outside the JSON. Ensure the JSON is properly formatted and valid."
)
RAW_SUFFIX_WITH_EXPLANATION = (
    "\n\nExplain your reasoning before providing your judgment."
)


class SimpleJudge(BaseJudge):
    """Judge class for evaluating outputs based on a given configuration."""

    def __init__(
        self,
        judge_config: Union[JudgeConfig, str],
    ):
        """Initialize the Judge.

        Args:
            judge_config: JudgeConfig object or a path to a judge configuration file.
                Contains both judge parameters and inference configuration.
        """
        if isinstance(judge_config, str):
            judge_config = JudgeConfig.from_path(judge_config)

        self._judge_params = judge_config.judge_params
        self._judge_params.replace_template_variables()
        self._inference_config = judge_config.inference_config

        # Create output fields based on judge configuration
        output_fields = []
        if self._judge_params.include_explanation:
            output_fields.append(self._create_explanation_output_field())
        output_fields.append(self._create_judgment_output_field(self._judge_params))

        # Generate an inference engine from inference config
        inference_engine = self._create_inference_engine(self._inference_config)

        # Append format suffix to system instruction if it exists
        system_instruction = self._judge_params.system_instruction
        if system_instruction:
            system_instruction = f"{system_instruction}{self._get_format_suffix()}"

        super().__init__(
            prompt_template=self._judge_params.prompt_template,
            system_instruction=system_instruction,
            example_field_values=self._judge_params.examples,
            response_format=self._judge_params.response_format,
            output_fields=output_fields,
            inference_engine=inference_engine,
        )

    @override
    def _build_judgment_prompt(self, judge_input: dict[str, str]) -> str:
        """Generate judge prompts using the template."""
        prompt_content = super()._build_judgment_prompt(judge_input)

        # Only append format suffix to judgment prompt if no system instruction exists
        # (otherwise it was already appended to system instruction in __init__)
        if not self._judge_params.system_instruction:
            prompt_content += self._get_format_suffix()

        return prompt_content

    def _get_format_suffix(self) -> str:
        """Get the appropriate format suffix based on response format and explanation.

        Returns:
            Format-specific instruction suffix to append to prompts
        """
        response_format = self._judge_params.response_format
        include_explanation = self._judge_params.include_explanation

        # Describe the expected judgment options to the judge
        if (
            self._judge_params.judgment_scores
            and len(self._judge_params.judgment_scores) > 1
        ):
            choices = [f"'{c}'" for c in self._judge_params.judgment_scores.keys()]
            choices_str = ", ".join(choices)
            judgment_options = f"{JUDGMENT_OPTIONS_ENUM_PREFIX}{choices_str}. "
        elif self._judge_params.judgment_type == JudgeOutputType.BOOL:
            judgment_options = f"{JUDGMENT_OPTIONS_BOOL}. "
        elif self._judge_params.judgment_type == JudgeOutputType.FLOAT:
            judgment_options = f"{JUDGMENT_OPTIONS_FLOAT}. "
        elif self._judge_params.judgment_type == JudgeOutputType.INT:
            judgment_options = f"{JUDGMENT_OPTIONS_INT}. "
        elif self._judge_params.judgment_type == JudgeOutputType.TEXT:
            judgment_options = f"{JUDGMENT_OPTIONS_TEXT}. "
        else:
            judgment_options = ""

        # Describe the expected response format to the judge
        if response_format == JudgeResponseFormat.XML:
            suffix = XML_SUFFIX_WITH_EXPLANATION if include_explanation else XML_SUFFIX
        elif response_format == JudgeResponseFormat.JSON:
            suffix = (
                JSON_SUFFIX_WITH_EXPLANATION if include_explanation else JSON_SUFFIX
            )
        elif response_format == JudgeResponseFormat.RAW:
            suffix = RAW_SUFFIX_WITH_EXPLANATION if include_explanation else ""
        else:
            suffix = ""

        return suffix.format(
            judgment_key=JUDGMENT_KEY,
            explanation_key=EXPLANATION_KEY,
            judgment_options=judgment_options,
        )

    def _create_judgment_output_field(self, params: JudgeParams) -> JudgeOutputField:
        """Create the main judgment output field."""
        return JudgeOutputField(
            field_key=JUDGMENT_KEY,
            field_type=params.judgment_type,
            field_scores=params.judgment_scores,
        )

    def _create_explanation_output_field(self) -> JudgeOutputField:
        """Create the explanation output field."""
        return JudgeOutputField(
            field_key=EXPLANATION_KEY,
            field_type=JudgeOutputType.TEXT,
            field_scores=None,
        )

    def _create_inference_engine(
        self, inference_config: InferenceConfig
    ) -> BaseInferenceEngine:
        """Create the inference engine based on the provided configuration."""
        from oumi.builders.inference_engines import build_inference_engine

        if inference_config.engine is None:
            raise ValueError("Inference engine not specified in the configuration.")
        elif inference_config.input_path or inference_config.output_path:
            raise ValueError(
                "Input and output paths are not supported in inference_config, when "
                "instantiating the SimpleJudge. Please set both to None."
            )

        return build_inference_engine(
            engine_type=inference_config.engine,
            model_params=inference_config.model,
            remote_params=inference_config.remote_params,
            generation_params=inference_config.generation,
        )
