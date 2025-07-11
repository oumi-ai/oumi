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

from oumi.core.configs.params.judge_params import (
    JudgeOutputType,
    JudgeResponseFormat,
)
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.placeholders import resolve_placeholders


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
            if not self.field_scores or not isinstance(self.field_scores, dict):
                raise ValueError(
                    "ENUM type requires field_scores to map values to scores."
                )
            # Only return the raw value if it exists in the scores mapping
            return raw_value if raw_value in self.field_scores else None

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
        output_fields: List of expected output fields for this judge
        field_values: Typed values for each expected output field
        field_scores: Numeric scores for each expected output field (if applicable)
        response_format: Format used for generating output (XML, JSON, or RAW)
    """

    raw_output: str
    parsed_output: dict[str, str] = {}
    output_fields: Optional[list[JudgeOutputField]] = None
    field_values: dict[str, Optional[Union[float, int, str, bool]]] = {}
    field_scores: dict[str, Optional[float]] = {}
    response_format: Optional[JudgeResponseFormat] = None

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
            response_format=response_format,
            output_fields=output_fields,
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

        # Remove any API formatting
        if json_output.startswith("```json"):
            json_output = json_output[len("```json") :].lstrip()
        if json_output.endswith("```"):
            json_output = json_output[:-3].rstrip()

        try:
            parsed = json.loads(json_output)
            # Ensure all values are strings for consistent processing
            return {k: str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            return {}

    def generate_raw_output(self, field_values: dict[str, str]) -> str:
        """Generate raw output string from field values in the specified format.

        Args:
            field_values: Dictionary mapping field keys to their string values.
                            Must contain values for all required output fields.

        Returns:
            Formatted raw output string ready for use as assistant response.

        Raises:
            ValueError: If required output fields are missing from field_values,
                            if response_format/output_fields are not set, or if
                            response_format is not supported.
        """
        if not self.response_format:
            raise ValueError("response_format must be set before generating output")
        if not self.output_fields:
            raise ValueError("output_fields must be set before generating output")

        # Extract required field keys from output_fields
        required_field_keys = {field.field_key for field in self.output_fields}

        # Validate that all required fields are provided
        provided_keys = set(field_values.keys())
        if missing_keys := required_field_keys - provided_keys:
            raise ValueError(
                f"Missing values for required output fields: {sorted(missing_keys)}. "
                f"Required: {sorted(required_field_keys)}, "
                f"Provided: {sorted(provided_keys)}"
            )

        # Filter field_values to only include required output fields
        filtered_field_values = {
            key: value
            for key, value in field_values.items()
            if key in required_field_keys
        }

        if self.response_format == JudgeResponseFormat.XML:
            return self._generate_xml_output(filtered_field_values)
        elif self.response_format == JudgeResponseFormat.JSON:
            return self._generate_json_output(filtered_field_values)
        elif self.response_format == JudgeResponseFormat.RAW:
            # For RAW format, concatenate values in the order of output_fields
            ordered_values = [
                filtered_field_values[field.field_key] for field in self.output_fields
            ]
            return "\n".join(ordered_values)
        else:
            raise ValueError(f"Unsupported response format: {self.response_format}")

    @classmethod
    def _generate_xml_output(cls, field_values: dict[str, str]) -> str:
        """Generate XML formatted output from field values.

        Args:
            field_values: Dictionary mapping field keys to their string values.

        Returns:
            XML formatted string with each field as a tag.
        """
        xml_parts = [f"<{key}>{value}</{key}>" for key, value in field_values.items()]
        return "\n".join(xml_parts)

    @classmethod
    def _generate_json_output(cls, field_values: dict[str, str]) -> str:
        """Generate JSON formatted output from field values.

        Args:
            field_values: Dictionary mapping field keys to their string values.

        Returns:
            JSON formatted string.
        """
        return json.dumps(field_values, indent=2)

    def to_json(self) -> str:
        """Convert the JudgeOutput to a JSON string.

        Returns:
            JSON string representation of the JudgeOutput data.
        """
        return json.dumps(self.model_dump())


class BaseJudge:
    """Base class for implementing judges that evaluate model outputs.

    A judge takes structured inputs, formats them using a prompt template,
    runs inference to get judgments, and parses the results into structured outputs.
    """

    def __init__(
        self,
        prompt_template: str,
        system_instruction: Optional[str],
        example_field_values: list[dict[str, str]],
        response_format: JudgeResponseFormat,
        output_fields: list[JudgeOutputField],
        inference_engine: BaseInferenceEngine,
    ):
        """Initialize the judge.

        Args:
            prompt_template: Template string with placeholders for input data
            system_instruction: Optional system message to guide judge behavior
            example_field_values: List of field value dicts for few-shot learning
            response_format: Expected format of judge responses (XML, JSON, or RAW)
            output_fields: List of fields expected in judge outputs
            inference_engine: Engine for running model inference
        """
        self.prompt_template = prompt_template
        self.system_instruction = system_instruction
        self.example_field_values = example_field_values
        self.response_format = response_format
        self.output_fields = output_fields
        self.inference_engine = inference_engine

        # Validate the configuration
        if prompt_template is None or not prompt_template.strip():
            raise ValueError("Prompt template cannot be empty or None")
        self._validate_output_fields(output_fields)

    def judge(
        self,
        inputs: list[dict[str, str]],
    ) -> list[JudgeOutput]:
        """Evaluate a batch of inputs and return structured judgments.

        Args:
            inputs: List of dictionaries containing input data for evaluation.
                    Each dict must contain values for all prompt_template placeholders.

        Returns:
            List of structured judge outputs with parsed results

        Raises:
            ValueError: If inference returns unexpected number of conversations
        """
        # Build few-shot examples: convert field values to prompts and responses
        example_user_prompts = [
            self._build_judgment_prompt(example_fields)
            for example_fields in self.example_field_values
        ]
        example_assistant_responses = [
            self._build_assistant_response(example_fields)
            for example_fields in self.example_field_values
        ]

        # Build a judgment prompt for each dataset input
        judgment_prompts = [
            self._build_judgment_prompt(input_data) for input_data in inputs
        ]

        # Create a conversation for each judgment prompt
        judge_conversations = [
            self._build_judge_conversation(
                system_instruction=self.system_instruction,
                example_user_prompts=example_user_prompts,
                example_assistant_responses=example_assistant_responses,
                judgment_prompt=judgment_prompt,
            )
            for judgment_prompt in judgment_prompts
        ]

        # Run inference for all conversations in batch
        completed_conversations = self._infer(judge_conversations)

        # Extract and parse the judgment outputs
        judge_outputs = []
        for conversation in completed_conversations:
            self._validate_completed_conversation(conversation)

            raw_output = str(conversation.messages[-1].content)
            judge_output = self._transform_judge_output(raw_output)
            judge_outputs.append(judge_output)

        return judge_outputs

    def _validate_output_fields(self, output_fields: list[JudgeOutputField]) -> None:
        """Ensure all output fields are properly defined."""
        if not output_fields:
            raise ValueError("Output fields cannot be empty")

        for field in self.output_fields:
            if field.field_key is None or not field.field_key.strip():
                raise ValueError(
                    f"Output field `field_key` cannot be None or empty: {field}"
                )
            if field.field_type == JudgeOutputType.ENUM and not field.field_scores:
                raise ValueError(
                    f"ENUM field type requires `field_scores` to be defined: {field}"
                )

    def _build_judgment_prompt(self, judge_input: dict[str, str]) -> str:
        """Generate a judge prompt by filling the template with input data.

        Args:
            judge_input: Dictionary mapping placeholder names to values

        Returns:
            Formatted prompt string ready for inference

        Raises:
            ValueError: If required placeholders are missing from judge_input
        """
        return resolve_placeholders(self.prompt_template, judge_input)

    def _build_assistant_response(self, field_values: dict[str, str]) -> str:
        """Generate the expected assistant response for few-shot examples.

        Args:
            field_values: Dictionary mapping field keys to their expected values.

        Returns:
            Formatted response string in the judge's expected output format.
        """
        # Create JudgeOutput instance with format configuration
        judge_output = JudgeOutput(
            raw_output="",
            response_format=self.response_format,
            output_fields=self.output_fields,
        )
        return judge_output.generate_raw_output(field_values=field_values)

    @classmethod
    def _build_judge_conversation(
        cls,
        system_instruction: Optional[str],
        example_user_prompts: list[str],
        example_assistant_responses: list[str],
        judgment_prompt: str,
    ) -> Conversation:
        """Create a conversation object.

        The conversation structure is:
            1. System instruction (if provided)
            2. Few-shot examples: alternating user prompts and assistant responses
            3. Final user prompt for the actual judgment

        Args:
            system_instruction: Optional system message to guide judge behavior
            example_user_prompts: List of example user prompts for few-shot learning
            example_assistant_responses: List of corresponding assistant responses
            judgment_prompt: The final user prompt for the actual judgment task

        Returns:
            Conversation object ready for inference

        Raises:
            ValueError: If number of example prompts doesn't match number of responses
        """
        # Validate that examples are properly paired
        if len(example_user_prompts) != len(example_assistant_responses):
            raise ValueError(
                f"Number of prompts ({len(example_user_prompts)}) must match "
                f"number of responses ({len(example_assistant_responses)})"
            )

        messages = []

        # Add system instruction, if provided
        if system_instruction:
            messages.append(Message(content=system_instruction, role=Role.SYSTEM))

        # Add few-shot examples as alternating user/assistant pairs
        for prompt, response in zip(example_user_prompts, example_assistant_responses):
            messages.extend(
                [
                    Message(content=prompt, role=Role.USER),
                    Message(content=response, role=Role.ASSISTANT),
                ]
            )

        # Add the actual judgment prompt
        messages.append(Message(content=judgment_prompt, role=Role.USER))

        return Conversation(messages=messages)

    def _validate_completed_conversation(self, conversation: Conversation) -> None:
        """Validate that completed conversation has the expected structure."""
        # Calculate expected message count
        expected_messages = 2  # Judgment user prompt + assistant response
        if self.system_instruction:
            expected_messages += 1
        expected_messages += 2 * len(self.example_field_values)

        if len(conversation.messages) != expected_messages:
            raise ValueError(
                f"Expected {expected_messages} messages, "
                f"got {len(conversation.messages)}"
            )

        # Validate that the last message is from the assistant
        if conversation.messages[-1].role != Role.ASSISTANT:
            raise ValueError(
                f"Expected last message from assistant, "
                f"got {conversation.messages[-1].role}"
            )

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
