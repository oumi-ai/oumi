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

from unittest.mock import Mock, patch

import pytest

from oumi.core.configs.params.judge_params import JudgeOutputType, JudgeResponseFormat
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.judges_v2.base_judge import BaseJudge, JudgeOutput, JudgeOutputField


class TestJudgeOutputField:
    """Test cases for the JudgeOutputField class."""

    def test_get_typed_value_bool_true(self):
        field = JudgeOutputField(
            field_key="test",
            field_type=JudgeOutputType.BOOL,
            field_scores=None,
        )

        assert field.get_typed_value("True") is True
        assert field.get_typed_value("true") is True
        assert field.get_typed_value("Yes") is True
        assert field.get_typed_value("yes") is True
        assert field.get_typed_value("1") is True

    def test_get_typed_value_bool_false(self):
        field = JudgeOutputField(
            field_key="test",
            field_type=JudgeOutputType.BOOL,
            field_scores=None,
        )

        assert field.get_typed_value("False") is False
        assert field.get_typed_value("false") is False
        assert field.get_typed_value("No") is False
        assert field.get_typed_value("no") is False
        assert field.get_typed_value("0") is False

    def test_get_typed_value_bool_invalid(self):
        field = JudgeOutputField(
            field_key="test",
            field_type=JudgeOutputType.BOOL,
            field_scores=None,
        )

        assert field.get_typed_value("maybe") is None
        assert field.get_typed_value("") is None

    def test_get_typed_value_int_valid(self):
        field = JudgeOutputField(
            field_key="test",
            field_type=JudgeOutputType.INT,
            field_scores=None,
        )

        assert field.get_typed_value("42") == 42
        assert field.get_typed_value("-10") == -10
        assert field.get_typed_value("0") == 0

    def test_get_typed_value_int_invalid(self):
        field = JudgeOutputField(
            field_key="test",
            field_type=JudgeOutputType.INT,
            field_scores=None,
        )

        assert field.get_typed_value("not_a_number") is None
        assert field.get_typed_value("3.14") is None

    def test_get_typed_value_float_valid(self):
        field = JudgeOutputField(
            field_key="test",
            field_type=JudgeOutputType.FLOAT,
            field_scores=None,
        )

        assert field.get_typed_value("3.14") == 3.14
        assert field.get_typed_value("42") == 42.0
        assert field.get_typed_value("-2.5") == -2.5

    def test_get_typed_value_float_invalid(self):
        field = JudgeOutputField(
            field_key="test",
            field_type=JudgeOutputType.FLOAT,
            field_scores=None,
        )

        assert field.get_typed_value("not_a_number") is None

    def test_get_typed_value_enum_with_scores(self):
        field = JudgeOutputField(
            field_key="test",
            field_type=JudgeOutputType.ENUM,
            field_scores={"excellent": 1.0, "good": 0.7, "poor": 0.3},
        )

        assert field.get_typed_value("excellent") == "excellent"
        assert field.get_typed_value("good") == "good"
        assert field.get_typed_value("poor") == "poor"
        assert field.get_typed_value("unmapped") is None

    def test_get_typed_value_enum_no_scores(self):
        field = JudgeOutputField(
            field_key="test",
            field_type=JudgeOutputType.ENUM,
            field_scores=None,
        )

        with pytest.raises(
            ValueError, match="ENUM type requires field_scores to map values to scores."
        ):
            field.get_typed_value("something")

    def test_get_typed_value_text(self):
        field = JudgeOutputField(
            field_key="test",
            field_type=JudgeOutputType.TEXT,
            field_scores=None,
        )

        assert field.get_typed_value("Hello, world!") == "Hello, world!"
        assert field.get_typed_value("") == ""


class TestJudgeOutput:
    """Test cases for the JudgeOutput class."""

    def test_parse_xml_output_simple(self):
        xml_output = "<judgment>True</judgment>"
        parsed = JudgeOutput._parse_xml_output(xml_output)
        assert parsed == {"judgment": "True"}

    def test_parse_xml_output_multiple_fields(self):
        xml_output = """
        <explanation>This is helpful</explanation>
        <judgment>True</judgment>
        """
        parsed = JudgeOutput._parse_xml_output(xml_output)
        assert parsed == {
            "explanation": "This is helpful",
            "judgment": "True",
        }

    def test_parse_xml_output_with_whitespace(self):
        xml_output = "<judgment>  True  </judgment>"
        parsed = JudgeOutput._parse_xml_output(xml_output)
        assert parsed == {"judgment": "True"}

    def test_parse_xml_output_empty(self):
        assert JudgeOutput._parse_xml_output("") == {}
        assert JudgeOutput._parse_xml_output(None) == {}

    def test_parse_json_output_simple(self):
        json_output = '{"judgment": "True"}'
        parsed = JudgeOutput._parse_json_output(json_output)
        assert parsed == {"judgment": "True"}

    def test_parse_json_output_multiple_fields(self):
        json_output = '{"explanation": "This is helpful", "judgment": true}'
        parsed = JudgeOutput._parse_json_output(json_output)
        assert parsed == {"explanation": "This is helpful", "judgment": "True"}

    def test_parse_json_output_empty(self):
        assert JudgeOutput._parse_json_output("") == {}
        assert JudgeOutput._parse_json_output(None) == {}

    def test_parse_json_output_malformed(self):
        json_output = '{"judgment": "True"'  # Missing closing brace
        parsed = JudgeOutput._parse_json_output(json_output)
        assert parsed == {}

    def test_from_raw_output_bool_no_scores(self):
        raw_output = "<judgment>True</judgment>"
        output_fields = [
            JudgeOutputField(
                field_key="judgment",
                field_type=JudgeOutputType.BOOL,
                field_scores=None,
            )
        ]

        judge_output = JudgeOutput.from_raw_output(
            raw_output=raw_output,
            response_format=JudgeResponseFormat.XML,
            output_fields=output_fields,
        )

        assert judge_output.raw_output == raw_output
        assert judge_output.parsed_output == {"judgment": "True"}
        assert judge_output.field_values == {"judgment": True}
        assert judge_output.field_scores == {"judgment": 1.0}

    def test_from_raw_output_bool_with_scores(self):
        raw_output = "<judgment>False</judgment>"
        output_fields = [
            JudgeOutputField(
                field_key="judgment",
                field_type=JudgeOutputType.BOOL,
                field_scores={"True": 1.0, "False": -0.5},
            )
        ]

        judge_output = JudgeOutput.from_raw_output(
            raw_output=raw_output,
            response_format=JudgeResponseFormat.XML,
            output_fields=output_fields,
        )

        assert judge_output.raw_output == raw_output
        assert judge_output.parsed_output == {"judgment": "False"}
        assert judge_output.field_values == {"judgment": False}
        assert judge_output.field_scores == {"judgment": -0.5}

    def test_from_raw_output_enum_with_scores(self):
        raw_output = '{"judgment": "good"}'
        output_fields = [
            JudgeOutputField(
                field_key="judgment",
                field_type=JudgeOutputType.ENUM,
                field_scores={"excellent": 1.0, "good": 0.7, "poor": 0.3},
            )
        ]

        judge_output = JudgeOutput.from_raw_output(
            raw_output=raw_output,
            response_format=JudgeResponseFormat.JSON,
            output_fields=output_fields,
        )

        assert judge_output.raw_output == raw_output
        assert judge_output.parsed_output == {"judgment": "good"}
        assert judge_output.field_values == {"judgment": "good"}
        assert judge_output.field_scores == {"judgment": 0.7}

    def test_from_raw_output_enum_with_scores_unmapped(self):
        raw_output = '{"judgment": "mediocre"}'
        output_fields = [
            JudgeOutputField(
                field_key="judgment",
                field_type=JudgeOutputType.ENUM,
                field_scores={"excellent": 1.0, "good": 0.7, "poor": 0.3},
            )
        ]

        judge_output = JudgeOutput.from_raw_output(
            raw_output=raw_output,
            response_format=JudgeResponseFormat.JSON,
            output_fields=output_fields,
        )

        assert judge_output.raw_output == raw_output
        assert judge_output.parsed_output == {"judgment": "mediocre"}
        assert judge_output.field_values == {"judgment": None}
        assert judge_output.field_scores == {"judgment": None}

    def test_from_raw_output_enum_no_scores(self):
        raw_output = '{"judgment": "mediocre"}'
        output_fields = [
            JudgeOutputField(
                field_key="judgment",
                field_type=JudgeOutputType.ENUM,
                field_scores=None,
            )
        ]

        with pytest.raises(
            ValueError,
            match="ENUM type requires field_scores to map values to scores.",
        ):
            JudgeOutput.from_raw_output(
                raw_output=raw_output,
                response_format=JudgeResponseFormat.JSON,
                output_fields=output_fields,
            )

    def test_from_raw_output_missing_field(self):
        raw_output = "<something_else>True</something_else>"
        output_fields = [
            JudgeOutputField(
                field_key="judgment",
                field_type=JudgeOutputType.BOOL,
                field_scores=None,
            )
        ]

        judge_output = JudgeOutput.from_raw_output(
            raw_output=raw_output,
            response_format=JudgeResponseFormat.XML,
            output_fields=output_fields,
        )

        assert judge_output.raw_output == raw_output
        assert judge_output.parsed_output == {"something_else": "True"}
        assert judge_output.field_values == {"judgment": None}
        assert judge_output.field_scores == {"judgment": None}

    def test_generate_raw_output_xml(self):
        output_fields = [
            JudgeOutputField(
                field_key="judgment",
                field_type=JudgeOutputType.BOOL,
                field_scores=None,
            ),
            JudgeOutputField(
                field_key="explanation",
                field_type=JudgeOutputType.TEXT,
                field_scores=None,
            ),
        ]

        judge_output = JudgeOutput(
            raw_output="",
            response_format=JudgeResponseFormat.XML,
            output_fields=output_fields,
        )

        field_values = {"judgment": "True", "explanation": "This is helpful"}
        result = judge_output.generate_raw_output(field_values)

        expected = (
            "<judgment>True</judgment>\n<explanation>This is helpful</explanation>"
        )
        assert result == expected

    def test_generate_raw_output_json(self):
        output_fields = [
            JudgeOutputField(
                field_key="judgment",
                field_type=JudgeOutputType.BOOL,
                field_scores=None,
            )
        ]

        judge_output = JudgeOutput(
            raw_output="",
            response_format=JudgeResponseFormat.JSON,
            output_fields=output_fields,
        )

        field_values = {"judgment": "True"}
        result = judge_output.generate_raw_output(field_values)

        expected = '{\n  "judgment": "True"\n}'
        assert result == expected

    def test_generate_raw_output_raw(self):
        output_fields = [
            JudgeOutputField(
                field_key="judgment",
                field_type=JudgeOutputType.BOOL,
                field_scores=None,
            ),
            JudgeOutputField(
                field_key="explanation",
                field_type=JudgeOutputType.TEXT,
                field_scores=None,
            ),
        ]

        judge_output = JudgeOutput(
            raw_output="",
            response_format=JudgeResponseFormat.RAW,
            output_fields=output_fields,
        )

        field_values = {"judgment": "True", "explanation": "This is helpful"}
        result = judge_output.generate_raw_output(field_values)

        expected = "True\nThis is helpful"
        assert result == expected

    def test_generate_raw_output_missing_fields(self):
        output_fields = [
            JudgeOutputField(
                field_key="judgment",
                field_type=JudgeOutputType.BOOL,
                field_scores=None,
            ),
            JudgeOutputField(
                field_key="explanation",
                field_type=JudgeOutputType.TEXT,
                field_scores=None,
            ),
        ]

        judge_output = JudgeOutput(
            raw_output="",
            response_format=JudgeResponseFormat.XML,
            output_fields=output_fields,
        )

        field_values = {"judgment": "True"}  # Missing explanation

        with pytest.raises(
            ValueError, match="Missing values for required output fields"
        ):
            judge_output.generate_raw_output(field_values)

    def test_generate_raw_output_no_format(self):
        judge_output = JudgeOutput(raw_output="")

        with pytest.raises(
            ValueError, match="response_format must be set before generating output"
        ):
            judge_output.generate_raw_output({"judgment": "True"})

    def test_generate_raw_output_no_fields(self):
        judge_output = JudgeOutput(
            raw_output="", response_format=JudgeResponseFormat.XML
        )

        with pytest.raises(
            ValueError, match="output_fields must be set before generating output"
        ):
            judge_output.generate_raw_output({"judgment": "True"})

    def test_to_json(self):
        """Test converting JudgeOutput to JSON string."""
        judge_output = JudgeOutput(
            raw_output=(
                "<judgment>True</judgment><explanation>This is helpful</explanation>"
            ),
            parsed_output={"judgment": "True", "explanation": "This is helpful"},
            output_fields=[
                JudgeOutputField(
                    field_key="judgment",
                    field_type=JudgeOutputType.BOOL,
                    field_scores=None,
                ),
                JudgeOutputField(
                    field_key="explanation",
                    field_type=JudgeOutputType.TEXT,
                    field_scores=None,
                ),
            ],
            field_values={"judgment": True, "explanation": "This is helpful"},
            field_scores={"judgment": 1.0, "explanation": None},
            response_format=JudgeResponseFormat.XML,
        )

        # Test JSON conversion
        json_output = judge_output.to_json()

        # Parse and verify the JSON contains expected data
        import json

        parsed_output = json.loads(json_output)
        assert parsed_output["raw_output"] == (
            "<judgment>True</judgment><explanation>This is helpful</explanation>"
        )
        assert parsed_output["parsed_output"] == {
            "judgment": "True",
            "explanation": "This is helpful",
        }

        assert len(parsed_output["output_fields"]) == 2
        assert parsed_output["output_fields"][0]["field_key"] == "judgment"
        assert parsed_output["output_fields"][0]["field_type"] == "bool"
        assert parsed_output["output_fields"][0]["field_scores"] is None
        assert parsed_output["output_fields"][1]["field_key"] == "explanation"
        assert parsed_output["output_fields"][1]["field_type"] == "text"
        assert parsed_output["output_fields"][1]["field_scores"] is None

        assert parsed_output["field_values"]["judgment"] is True
        assert parsed_output["field_values"]["explanation"] == "This is helpful"
        assert parsed_output["field_scores"]["judgment"] == 1.0
        assert parsed_output["field_scores"]["explanation"] is None
        assert parsed_output["response_format"] == "xml"


class TestBaseJudge:
    """Test cases for the BaseJudge class."""

    @pytest.fixture
    def mock_inference_engine(self):
        return Mock()

    @pytest.fixture
    def sample_output_fields(self):
        return [
            JudgeOutputField(
                field_key="judgment",
                field_type=JudgeOutputType.BOOL,
                field_scores=None,
            )
        ]

    @pytest.fixture
    def base_judge(self, mock_inference_engine, sample_output_fields):
        return BaseJudge(
            prompt_template="Is this helpful? Question: {question}, Answer: {answer}",
            system_instruction=None,
            example_field_values=[],
            response_format=JudgeResponseFormat.XML,
            output_fields=sample_output_fields,
            inference_engine=mock_inference_engine,
        )

    def test_init(self, base_judge, mock_inference_engine, sample_output_fields):
        assert (
            base_judge.prompt_template
            == "Is this helpful? Question: {question}, Answer: {answer}"
        )
        assert base_judge.system_instruction is None
        assert base_judge.example_field_values == []
        assert base_judge.response_format == JudgeResponseFormat.XML
        assert base_judge.output_fields == sample_output_fields
        assert base_judge.inference_engine == mock_inference_engine

    def test_build_judgment_prompt(self, base_judge):
        judge_input = {"question": "What is 2+2?", "answer": "4"}
        prompt = base_judge._build_judgment_prompt(judge_input)
        expected = "Is this helpful? Question: What is 2+2?, Answer: 4"
        assert prompt == expected

    def test_build_judgment_prompt_missing_placeholder(self, base_judge):
        judge_input = {"question": "What is 2+2?"}  # Missing 'answer'

        with pytest.raises(ValueError, match="Missing value for placeholder: answer"):
            base_judge._build_judgment_prompt(judge_input)

    def test_build_judgment_prompt_extra_data(self, base_judge):
        judge_input = {"question": "What is 2+2?", "answer": "4", "extra": "ignored"}
        prompt = base_judge._build_judgment_prompt(judge_input)
        expected = "Is this helpful? Question: What is 2+2?, Answer: 4"
        assert prompt == expected

    def test_build_judge_conversation_simple(self, base_judge):
        conversation = base_judge._build_judge_conversation(
            system_instruction=None,
            example_user_prompts=[],
            example_assistant_responses=[],
            judgment_prompt="Test prompt",
        )

        assert len(conversation.messages) == 1
        assert conversation.messages[0].content == "Test prompt"
        assert conversation.messages[0].role == Role.USER

    def test_build_judge_conversation_with_system(self, base_judge):
        conversation = base_judge._build_judge_conversation(
            system_instruction="You are a helpful judge",
            example_user_prompts=[],
            example_assistant_responses=[],
            judgment_prompt="Test prompt",
        )

        assert len(conversation.messages) == 2
        assert conversation.messages[0].content == "You are a helpful judge"
        assert conversation.messages[0].role == Role.SYSTEM
        assert conversation.messages[1].content == "Test prompt"
        assert conversation.messages[1].role == Role.USER

    def test_build_judge_conversation_with_examples(self, base_judge):
        conversation = base_judge._build_judge_conversation(
            system_instruction=None,
            example_user_prompts=["Example question?"],
            example_assistant_responses=["<judgment>True</judgment>"],
            judgment_prompt="Test prompt",
        )

        assert len(conversation.messages) == 3
        assert conversation.messages[0].content == "Example question?"
        assert conversation.messages[0].role == Role.USER
        assert conversation.messages[1].content == "<judgment>True</judgment>"
        assert conversation.messages[1].role == Role.ASSISTANT
        assert conversation.messages[2].content == "Test prompt"
        assert conversation.messages[2].role == Role.USER

    def test_build_judge_conversation_mismatched_examples(self, base_judge):
        with pytest.raises(
            ValueError,
            match=r"Number of prompts \(2\) must match number of responses \(1\)",
        ):
            base_judge._build_judge_conversation(
                system_instruction=None,
                example_user_prompts=["Example 1?", "Example 2?"],
                example_assistant_responses=["<judgment>True</judgment>"],
                judgment_prompt="Test prompt",
            )

    def test_build_assistant_response(self, base_judge):
        field_values = {"judgment": "True"}
        response = base_judge._build_assistant_response(field_values)
        expected = "<judgment>True</judgment>"
        assert response == expected

    def test_infer_preserves_metadata(self, base_judge, mock_inference_engine):
        # Setup input conversations with metadata
        input_convs = [
            Conversation(
                messages=[Message(content="test1", role=Role.USER)],
                metadata={"id": "conv1", "custom": "data1"},
            ),
            Conversation(
                messages=[Message(content="test2", role=Role.USER)],
                metadata={"id": "conv2", "custom": "data2"},
            ),
        ]

        # Setup mock to return conversations with responses
        output_convs = [
            Conversation(
                messages=[
                    Message(content="test1", role=Role.USER),
                    Message(content="response1", role=Role.ASSISTANT),
                ]
            ),
            Conversation(
                messages=[
                    Message(content="test2", role=Role.USER),
                    Message(content="response2", role=Role.ASSISTANT),
                ]
            ),
        ]
        mock_inference_engine.infer.return_value = output_convs

        result = base_judge._infer(input_convs)

        # Check that metadata was preserved
        assert len(result) == 2
        assert result[0].metadata == {"id": "conv1", "custom": "data1"}
        assert result[1].metadata == {"id": "conv2", "custom": "data2"}

        mock_inference_engine.infer.assert_called_once_with(input=input_convs)

    def test_infer_length_mismatch(self, base_judge, mock_inference_engine):
        input_convs = [Conversation(messages=[Message(content="test", role=Role.USER)])]
        mock_inference_engine.infer.return_value = []  # Wrong length

        with pytest.raises(
            ValueError, match="Inference engine returned 0 responses but expected 1"
        ):
            base_judge._infer(input_convs)

    def test_transform_judge_output(self, base_judge):
        raw_output = "<judgment>True</judgment>"

        with patch.object(JudgeOutput, "from_raw_output") as mock_from_raw:
            mock_judge_output = Mock()
            mock_from_raw.return_value = mock_judge_output

            result = base_judge._transform_judge_output(raw_output)

            mock_from_raw.assert_called_once_with(
                raw_output=raw_output,
                response_format=JudgeResponseFormat.XML,
                output_fields=base_judge.output_fields,
            )
            assert result == mock_judge_output

    def test_judge_end_to_end(self, base_judge, mock_inference_engine):
        # Setup input data
        inputs = [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 2+2?", "answer": "3"},
        ]

        # Setup mock inference engine to return response
        response_conv_1 = Conversation(
            messages=[
                Message(
                    content="Is this helpful? Question: What is 1+1?, Answer: 2",
                    role=Role.USER,
                ),
                Message(content="<judgment>True</judgment>", role=Role.ASSISTANT),
            ]
        )
        response_conv_2 = Conversation(
            messages=[
                Message(
                    content="Is this helpful? Question: What is 2+2?, Answer: 3",
                    role=Role.USER,
                ),
                Message(content="<judgment>False</judgment>", role=Role.ASSISTANT),
            ]
        )
        mock_inference_engine.infer.return_value = [response_conv_1, response_conv_2]

        # Execute judge
        results = base_judge.judge(inputs)

        # Verify results
        assert len(results) == 2
        assert results[0].raw_output == "<judgment>True</judgment>"
        assert results[0].field_values == {"judgment": True}
        assert results[0].field_scores == {"judgment": 1.0}
        assert results[1].raw_output == "<judgment>False</judgment>"
        assert results[1].field_values == {"judgment": False}
        assert results[1].field_scores == {"judgment": 0.0}

    def test_judge_invalid_conversation_length(self, base_judge, mock_inference_engine):
        inputs = [{"question": "What is 2+2?", "answer": "4"}]

        # Return conversation with wrong number of messages
        response_conv = Conversation(
            messages=[Message(content="single message", role=Role.USER)]
        )
        mock_inference_engine.infer.return_value = [response_conv]

        with pytest.raises(ValueError, match="Expected 2 messages, got 1"):
            base_judge.judge(inputs)
