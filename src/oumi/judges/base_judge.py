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

import json
import re
from typing import Optional, Union

import pydantic
from typing_extensions import Self

from oumi.core.configs.judge_config import JudgeOutputType, JudgeResponseFormat
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role


class JudgeOutputField(pydantic.BaseModel):
    """Represents a single output field that a judge can produce.

    Attributes:
        field_key: The key/name for this field in the judge's output
        field_type: The data type expected for this field's value
        field_scores: Optional mapping from categorical values to numeric scores
    """

    field_key: str
    field_type: JudgeOutputType
    field_scores: Optional[dict[str, float]]

    def get_typed_value(self, raw_value: str) -> Optional[Union[float, int, str, bool]]:
        """Convert the field's raw string value to the appropriate type.

        Args:
            raw_value: The raw string value from the judge's output

        Returns:
            The typed value, or None if conversion fails

        Raises:
            ValueError: If the field_type is not supported
        """
        if self.field_type == JudgeOutputType.BOOL:
            from oumi.utils.str_utils import try_str_to_bool

            return try_str_to_bool(raw_value)

        elif self.field_type == JudgeOutputType.INT:
            try:
                return int(raw_value)
            except ValueError:
                return None

        elif self.field_type == JudgeOutputType.FLOAT:
            try:
                return float(raw_value)
            except ValueError:
                return None

        elif self.field_type == JudgeOutputType.ENUM:
            if self.field_scores:
                # Return None for unmapped enum values
                return raw_value if raw_value in self.field_scores else None
            else:
                return raw_value

        elif self.field_type == JudgeOutputType.TEXT:
            return raw_value

        else:
            raise ValueError(
                f"Unsupported field type: {self.field_type}. "
                "Supported types are: BOOL, INT, FLOAT, ENUM, TEXT."
            )


class JudgeOutput(pydantic.BaseModel):
    """Represents the output from a judge evaluation.

    Attributes:
        raw_output: The original unprocessed output from the judge
        parsed_output: Structured data (fields & their values) extracted from raw output
        field_values: Typed values for each expected output field
        field_scores: Numeric scores for each expected output field (if applicable)
    """

    raw_output: str
    parsed_output: dict[str, str] = {}
    field_values: dict[str, Optional[Union[float, int, str, bool]]] = {}
    field_scores: dict[str, Optional[float]] = {}

    @classmethod
    def from_raw_output(
        cls,
        raw_output: str,
        response_format: JudgeResponseFormat,
        output_fields: list[JudgeOutputField],
    ) -> Self:
        """Generate a structured judge output from a raw model output."""
        field_values = {}
        field_scores = {}

        # Parse the judge's response based on the expected format
        if response_format == JudgeResponseFormat.XML:
            parsed_output = cls._parse_xml_output(raw_output)
        elif response_format == JudgeResponseFormat.JSON:
            parsed_output = cls._parse_json_output(raw_output)
        else:  # JudgeResponseFormat.RAW
            parsed_output = {}

        # Process each expected output field
        for field in output_fields:
            if field.field_key not in parsed_output:
                field_values[field.field_key] = None
                field_scores[field.field_key] = None
                continue

            # Extract and clean the raw value
            raw_value = parsed_output[field.field_key].strip()

            # Convert to the appropriate type
            typed_value = field.get_typed_value(raw_value)
            field_values[field.field_key] = typed_value

            # Extract numeric score if field has score mapping
            if field.field_scores:
                field_scores[field.field_key] = field.field_scores.get(raw_value)
            elif field.field_type == JudgeOutputType.BOOL:
                # For boolean fields, scores can be inferred
                field_scores[field.field_key] = 1.0 if typed_value else 0.0
            else:
                field_scores[field.field_key] = None

        return cls(
            raw_output=raw_output,
            parsed_output=parsed_output,
            field_values=field_values,
            field_scores=field_scores,
        )

    @classmethod
    def _parse_xml_output(cls, xml_output: Optional[str]) -> dict[str, str]:
        """Parses an XML judge output."""
        if not xml_output:
            return {}

        # Regex pattern to match XML-like tags and their content
        # Captures the tag name in group 1 and the content between tags in group 2
        # For example, "<label>True</label>" would match as ("label", "True")
        pattern = r"<(\w+)>(.*?)</\1>"
        matches = re.findall(pattern, xml_output, re.DOTALL)

        return {field_name: field_value.strip() for field_name, field_value in matches}

    # TODO: Consider leveraging structured-outputs for better JSON parsing
    # https://oumi.ai/docs/en/latest/user_guides/infer/common_workflows.html#structured-outputs
    @classmethod
    def _parse_json_output(cls, json_output: Optional[str]) -> dict[str, str]:
        """Parse judgment data from JSON format.

        Args:
            json_output: Raw JSON string from the judge

        Returns:
            Dictionary of field names to values, empty dict if parsing fails
        """
        if not json_output:
            return {}
        try:
            parsed = json.loads(json_output)
            # Ensure all values are strings for consistent processing
            return {k: str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            return {}


class BaseJudge:
    """Base class for implementing judges that evaluate model outputs.

    A judge takes structured inputs, formats them using a prompt template,
    runs inference to get judgments, and parses the results into structured outputs.
    """

    def __init__(
        self,
        prompt_template: str,
        response_format: JudgeResponseFormat,
        output_fields: list[JudgeOutputField],
        inference_engine: BaseInferenceEngine,
    ):
        """Initialize the judge.

        Args:
            prompt_template: Template string with placeholders for input data
            response_format: Expected format of judge responses (XML, JSON, or RAW)
            output_fields: List of fields expected in judge outputs
            inference_engine: Engine for running model inference
        """
        self.prompt_template = prompt_template
        self.response_format = response_format
        self.output_fields = output_fields
        self.inference_engine = inference_engine

    def judge(
        self,
        inputs: list[dict[str, str]],
    ) -> list[JudgeOutput]:
        """Evaluate a batch of inputs and return structured judgments.

        Args:
            inputs: List of dictionaries containing input data for evaluation

        Returns:
            List of structured judge outputs with parsed results
        """
        # Build prompts and conversations for all inputs
        judgment_prompts = [
            self._build_judgement_prompt(input_data) for input_data in inputs
        ]
        judge_conversations = [
            self._build_judge_conversation(prompt) for prompt in judgment_prompts
        ]

        # Run inference for all conversations in batch
        completed_conversations = self._infer(judge_conversations)

        # Extract and parse the judgment outputs
        judge_outputs = []
        for conversation in completed_conversations:
            if len(conversation.messages) != 2:
                raise ValueError("Expected conversation to have precisely 2 messages")

            raw_output = str(conversation.messages[-1].content)
            parsed_output = self._transform_judge_output(raw_output)
            judge_outputs.append(parsed_output)

        return judge_outputs

    def _build_judgement_prompt(self, judge_input: dict[str, str]) -> str:
        """Generate a judge prompt by filling the template with input data.

        Args:
            judge_input: Dictionary mapping placeholder names to values

        Returns:
            Formatted prompt string ready for inference

        Raises:
            ValueError: If required placeholders are missing from judge_input
        """
        # Extract all placeholders from the template (e.g., {question}, {answer})
        required_placeholders = set(re.findall(r"\{(\w+)\}", self.prompt_template))

        # Validate that all required data is provided
        provided_keys = set(judge_input.keys())
        if missing_keys := required_placeholders - provided_keys:
            raise ValueError(
                f"Missing values for template placeholders: {sorted(missing_keys)}. "
                f"Required: {sorted(required_placeholders)}, "
                f"Provided: {sorted(provided_keys)}"
            )

        # Format the template with the provided data
        return self.prompt_template.format(**judge_input)

    def _build_judge_conversation(self, judgment_prompt: str) -> Conversation:
        """Create a conversation object from a formatted judge prompt.

        Args:
            judgment_prompt: The formatted prompt string

        Returns:
            Conversation object ready for inference
        """
        messages = [Message(content=judgment_prompt, role=Role.USER)]
        return Conversation(messages=messages)

    def _infer(self, conversations: list[Conversation]) -> list[Conversation]:
        """Run inference on judge conversations and preserve metadata.

        Args:
            conversations: List of conversations to run inference on

        Returns:
            List of conversations with model responses added
        """
        # Preserve original metadata from input conversations
        original_metadata = [conv.metadata for conv in conversations]

        # Run batch inference
        response_conversations = self.inference_engine.infer(input=conversations)

        if len(response_conversations) != len(original_metadata):
            raise ValueError(
                f"Inference engine returned {len(response_conversations)} responses "
                f"but expected {len(original_metadata)}"
            )

        # Restore original metadata to response conversations
        for response_conv, metadata in zip(response_conversations, original_metadata):
            response_conv.metadata.update(metadata)

        return response_conversations

    def _transform_judge_output(self, raw_output: str) -> JudgeOutput:
        """Parse raw model output into structured judge output.

        Args:
            raw_output: The raw string output from the judge model

        Returns:
            Structured judge output with parsed fields and values
        """
        return JudgeOutput.from_raw_output(
            raw_output=raw_output,
            response_format=self.response_format,
            output_fields=self.output_fields,
        )
