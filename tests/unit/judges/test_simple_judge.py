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

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.judge_config_v2 import (
    JudgeConfig,
)
from oumi.core.configs.params.judge_params import (
    JudgeOutputType,
    JudgeParams,
    JudgeResponseFormat,
)
from oumi.core.configs.params.model_params import ModelParams
from oumi.judges_v2.simple_judge import (
    EXPLANATION_KEY,
    JSON_SUFFIX,
    JSON_SUFFIX_WITH_EXPLANATION,
    JUDGMENT_KEY,
    RAW_SUFFIX_WITH_EXPLANATION,
    XML_SUFFIX,
    XML_SUFFIX_WITH_EXPLANATION,
    SimpleJudge,
)


class TestSimpleJudge:
    """Test cases for the SimpleJudge class."""

    # Test constants
    TEST_PROMPT_TEMPLATE = "Test: {input}"
    TEST_ENUM_SCORES = {"excellent": 1.0, "good": 0.7, "poor": 0.3}

    # Test judge input data
    XML_JUDGE_INPUT = {"question": "What is 2+2?", "answer": "4"}
    JSON_JUDGE_INPUT = {"answer": "Some good answer"}
    RAW_JUDGE_INPUT = {"content": "Some content"}

    # Pre-formatted suffix constants for tests
    BOOL_JUDGMENT_OPTIONS = "Your judgment should be a single word: 'Yes' or 'No'. "
    ENUM_JUDGMENT_OPTIONS = (
        "Your judgment should be one of the following options: "
        "'excellent', 'good', 'poor'. "
    )
    TEXT_JUDGMENT_OPTIONS = (
        "Your judgment should be provided in the form of free text. "
    )

    XML_SUFFIX_FORMATTED_BOOL = XML_SUFFIX.format(
        judgment_key=JUDGMENT_KEY,
        explanation_key=EXPLANATION_KEY,
        judgment_options=BOOL_JUDGMENT_OPTIONS,
    )
    XML_SUFFIX_WITH_EXPLANATION_FORMATTED_BOOL = XML_SUFFIX_WITH_EXPLANATION.format(
        judgment_key=JUDGMENT_KEY,
        explanation_key=EXPLANATION_KEY,
        judgment_options=BOOL_JUDGMENT_OPTIONS,
    )
    JSON_SUFFIX_FORMATTED_ENUM = JSON_SUFFIX.format(
        judgment_key=JUDGMENT_KEY,
        explanation_key=EXPLANATION_KEY,
        judgment_options=ENUM_JUDGMENT_OPTIONS,
    )
    JSON_SUFFIX_WITH_EXPLANATION_FORMATTED_ENUM = JSON_SUFFIX_WITH_EXPLANATION.format(
        judgment_key=JUDGMENT_KEY,
        explanation_key=EXPLANATION_KEY,
        judgment_options=ENUM_JUDGMENT_OPTIONS,
    )
    RAW_SUFFIX_WITH_EXPLANATION_FORMATTED_TEXT = RAW_SUFFIX_WITH_EXPLANATION.format(
        judgment_key=JUDGMENT_KEY,
        explanation_key=EXPLANATION_KEY,
        judgment_options=TEXT_JUDGMENT_OPTIONS,
    )

    @pytest.fixture
    def xml_config_no_explanation(self):
        return JudgeConfig(
            judge_params=JudgeParams(
                prompt_template=(
                    "Is this helpful? Question: {question}, Answer: {answer}"
                ),
                response_format=JudgeResponseFormat.XML,
                judgment_type=JudgeOutputType.BOOL,
                include_explanation=False,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )

    @pytest.fixture
    def xml_config_with_explanation(self):
        return JudgeConfig(
            judge_params=JudgeParams(
                prompt_template=(
                    "Is this helpful? Question: {question}, Answer: {answer}"
                ),
                response_format=JudgeResponseFormat.XML,
                judgment_type=JudgeOutputType.BOOL,
                include_explanation=True,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )

    @pytest.fixture
    def json_config_no_explanation(self):
        return JudgeConfig(
            judge_params=JudgeParams(
                prompt_template="Rate this answer: {answer}",
                response_format=JudgeResponseFormat.JSON,
                judgment_type=JudgeOutputType.ENUM,
                judgment_scores=self.TEST_ENUM_SCORES,
                include_explanation=False,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )

    @pytest.fixture
    def json_config_with_explanation(self):
        return JudgeConfig(
            judge_params=JudgeParams(
                prompt_template="Rate this answer: {answer}",
                response_format=JudgeResponseFormat.JSON,
                judgment_type=JudgeOutputType.ENUM,
                judgment_scores=self.TEST_ENUM_SCORES,
                include_explanation=True,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )

    @pytest.fixture
    def mock_inference_engine(self):
        return Mock()

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_init_with_engine_no_explanation(
        self, mock_create_engine, xml_config_no_explanation
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(judge_config=xml_config_no_explanation)

        # Verify the config was properly processed
        assert judge._judge_params == xml_config_no_explanation.judge_params
        assert judge._inference_config == xml_config_no_explanation.inference_config
        assert judge.inference_engine == mock_engine
        assert (
            judge.prompt_template
            == xml_config_no_explanation.judge_params.prompt_template
        )
        assert (
            judge.response_format
            == xml_config_no_explanation.judge_params.response_format
        )

        # Should have one output field for judgment
        assert len(judge.output_fields) == 1
        assert judge.output_fields[0].field_key == JUDGMENT_KEY
        assert judge.output_fields[0].field_type == JudgeOutputType.BOOL
        assert judge.output_fields[0].field_scores is None

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_init_with_engine_explanation(
        self, mock_create_engine, json_config_with_explanation
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(judge_config=json_config_with_explanation)

        # Should have two output fields: explanation and judgment
        assert len(judge.output_fields) == 2
        assert judge.output_fields[0].field_key == EXPLANATION_KEY
        assert judge.output_fields[0].field_type == JudgeOutputType.TEXT
        assert judge.output_fields[0].field_scores is None
        assert judge.output_fields[1].field_key == JUDGMENT_KEY
        assert judge.output_fields[1].field_type == JudgeOutputType.ENUM
        assert judge.output_fields[1].field_scores == {
            "excellent": 1.0,
            "good": 0.7,
            "poor": 0.3,
        }

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_init_creates_inference_engine(
        self, mock_create_engine, xml_config_no_explanation
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(judge_config=xml_config_no_explanation)

        assert judge.inference_engine == mock_engine
        mock_create_engine.assert_called_once_with(
            xml_config_no_explanation.inference_config
        )

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_build_prompt_xml_no_explanation(
        self, mock_create_engine, xml_config_no_explanation
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(judge_config=xml_config_no_explanation)

        prompt = judge._build_judgment_prompt(self.XML_JUDGE_INPUT)

        expected = (
            f"Is this helpful? Question: What is 2+2?, Answer: 4"
            f"{self.XML_SUFFIX_FORMATTED_BOOL}"
        )
        assert prompt == expected

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_build_prompt_xml_with_explanation(
        self, mock_create_engine, xml_config_with_explanation
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(judge_config=xml_config_with_explanation)

        prompt = judge._build_judgment_prompt(self.XML_JUDGE_INPUT)

        expected = (
            f"Is this helpful? Question: What is 2+2?, Answer: 4"
            f"{self.XML_SUFFIX_WITH_EXPLANATION_FORMATTED_BOOL}"
        )
        assert prompt == expected

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_build_prompt_json_no_explanation(
        self, mock_create_engine, json_config_no_explanation
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(judge_config=json_config_no_explanation)

        prompt = judge._build_judgment_prompt(self.JSON_JUDGE_INPUT)

        expected = (
            f"Rate this answer: Some good answer{self.JSON_SUFFIX_FORMATTED_ENUM}"
        )
        assert prompt == expected

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_build_prompt_json_with_explanation(
        self, mock_create_engine, json_config_with_explanation
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(judge_config=json_config_with_explanation)

        prompt = judge._build_judgment_prompt(self.JSON_JUDGE_INPUT)

        expected = (
            f"Rate this answer: Some good answer"
            f"{self.JSON_SUFFIX_WITH_EXPLANATION_FORMATTED_ENUM}"
        )
        assert prompt == expected

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_build_prompt_raw_no_explanation(self, mock_create_engine):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        config = JudgeConfig(
            judge_params=JudgeParams(
                prompt_template="Evaluate: {content}",
                response_format=JudgeResponseFormat.RAW,
                judgment_type=JudgeOutputType.TEXT,
                include_explanation=False,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )
        judge = SimpleJudge(judge_config=config)

        prompt = judge._build_judgment_prompt(self.RAW_JUDGE_INPUT)

        expected = "Evaluate: Some content"  # No suffix for RAW without explanation
        assert prompt == expected

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_build_prompt_raw_with_explanation(self, mock_create_engine):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        config = JudgeConfig(
            judge_params=JudgeParams(
                prompt_template="Evaluate: {content}",
                response_format=JudgeResponseFormat.RAW,
                judgment_type=JudgeOutputType.TEXT,
                include_explanation=True,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )
        judge = SimpleJudge(judge_config=config)

        prompt = judge._build_judgment_prompt(self.RAW_JUDGE_INPUT)

        expected = (
            f"Evaluate: Some content{self.RAW_SUFFIX_WITH_EXPLANATION_FORMATTED_TEXT}"
        )
        assert prompt == expected

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_get_format_suffix_xml(self, mock_create_engine):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        config_no_exp = JudgeConfig(
            judge_params=JudgeParams(
                prompt_template=self.TEST_PROMPT_TEMPLATE,
                response_format=JudgeResponseFormat.XML,
                include_explanation=False,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )
        judge_no_exp = SimpleJudge(judge_config=config_no_exp)
        assert judge_no_exp._get_format_suffix() == self.XML_SUFFIX_FORMATTED_BOOL

        config_with_exp = JudgeConfig(
            judge_params=JudgeParams(
                prompt_template=self.TEST_PROMPT_TEMPLATE,
                response_format=JudgeResponseFormat.XML,
                include_explanation=True,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )
        judge_with_exp = SimpleJudge(judge_config=config_with_exp)
        assert (
            judge_with_exp._get_format_suffix()
            == self.XML_SUFFIX_WITH_EXPLANATION_FORMATTED_BOOL
        )

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_get_format_suffix_json(self, mock_create_engine):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        config_no_exp = JudgeConfig(
            judge_params=JudgeParams(
                prompt_template=self.TEST_PROMPT_TEMPLATE,
                response_format=JudgeResponseFormat.JSON,
                judgment_type=JudgeOutputType.ENUM,
                judgment_scores=self.TEST_ENUM_SCORES,
                include_explanation=False,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )
        judge_no_exp = SimpleJudge(judge_config=config_no_exp)
        assert judge_no_exp._get_format_suffix() == self.JSON_SUFFIX_FORMATTED_ENUM

        config_with_exp = JudgeConfig(
            judge_params=JudgeParams(
                prompt_template=self.TEST_PROMPT_TEMPLATE,
                response_format=JudgeResponseFormat.JSON,
                judgment_type=JudgeOutputType.ENUM,
                judgment_scores=self.TEST_ENUM_SCORES,
                include_explanation=True,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )
        judge_with_exp = SimpleJudge(judge_config=config_with_exp)
        assert (
            judge_with_exp._get_format_suffix()
            == self.JSON_SUFFIX_WITH_EXPLANATION_FORMATTED_ENUM
        )

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_get_format_suffix_raw(self, mock_create_engine):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        config_no_exp = JudgeConfig(
            judge_params=JudgeParams(
                prompt_template=self.TEST_PROMPT_TEMPLATE,
                response_format=JudgeResponseFormat.RAW,
                judgment_type=JudgeOutputType.TEXT,
                include_explanation=False,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )
        judge_no_exp = SimpleJudge(judge_config=config_no_exp)
        assert judge_no_exp._get_format_suffix() == ""

        config_with_exp = JudgeConfig(
            judge_params=JudgeParams(
                prompt_template=self.TEST_PROMPT_TEMPLATE,
                response_format=JudgeResponseFormat.RAW,
                judgment_type=JudgeOutputType.TEXT,
                include_explanation=True,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )
        judge_with_exp = SimpleJudge(judge_config=config_with_exp)
        assert (
            judge_with_exp._get_format_suffix()
            == self.RAW_SUFFIX_WITH_EXPLANATION_FORMATTED_TEXT
        )

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_create_judgment_output_field(
        self, mock_create_engine, xml_config_no_explanation
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(judge_config=xml_config_no_explanation)

        field = judge._create_judgment_output_field(
            xml_config_no_explanation.judge_params
        )

        assert field.field_key == JUDGMENT_KEY
        assert field.field_type == JudgeOutputType.BOOL
        assert field.field_scores is None

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_create_judgment_output_field_with_scores(
        self, mock_create_engine, json_config_with_explanation
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(judge_config=json_config_with_explanation)

        field = judge._create_judgment_output_field(
            json_config_with_explanation.judge_params
        )

        assert field.field_key == JUDGMENT_KEY
        assert field.field_type == JudgeOutputType.ENUM
        assert field.field_scores == {"excellent": 1.0, "good": 0.7, "poor": 0.3}

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_create_explanation_output_field(
        self, mock_create_engine, json_config_with_explanation
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(judge_config=json_config_with_explanation)

        field = judge._create_explanation_output_field()

        assert field.field_key == EXPLANATION_KEY
        assert field.field_type == JudgeOutputType.TEXT
        assert field.field_scores is None

    @patch("oumi.builders.inference_engines.build_inference_engine")
    def test_create_inference_engine(
        self, mock_build_engine, xml_config_no_explanation
    ):
        mock_engine = Mock()
        mock_build_engine.return_value = mock_engine

        _ = SimpleJudge(judge_config=xml_config_no_explanation)

        # Should be called during init with inference_config
        mock_build_engine.assert_called_once_with(
            engine_type=xml_config_no_explanation.inference_config.engine,
            model_params=xml_config_no_explanation.inference_config.model,
            remote_params=xml_config_no_explanation.inference_config.remote_params,
            generation_params=xml_config_no_explanation.inference_config.generation,
        )

    def test_enum_judgment_type_requires_scores(self):
        """Test that ENUM judgment type requires judgment_scores to be provided."""
        with pytest.raises(
            ValueError, match="judgment_scores must be provided for ENUM judgment_type"
        ):
            JudgeParams(
                prompt_template="Rate this: {text}",
                response_format=JudgeResponseFormat.JSON,
                judgment_type=JudgeOutputType.ENUM,
                judgment_scores=None,
            )

    def test_enum_judgment_type_with_empty_scores(self):
        """Test that ENUM judgment type with empty scores fails validation."""
        with pytest.raises(
            ValueError, match="judgment_scores must be provided for ENUM judgment_type"
        ):
            JudgeParams(
                prompt_template="Rate this: {text}",
                response_format=JudgeResponseFormat.JSON,
                judgment_type=JudgeOutputType.ENUM,
                judgment_scores={},
            )

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_format_suffix_behavior_with_and_without_system_instruction(
        self, mock_create_engine
    ):
        """Test that format suffix goes to system instruction or judgment prompt."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        # Test case 1: With system instruction - format suffix in system instruction
        config_with_system = JudgeConfig(
            judge_params=JudgeParams(
                prompt_template="Rate: {text}",
                response_format=JudgeResponseFormat.JSON,
                judgment_type=JudgeOutputType.BOOL,
                include_explanation=False,
                system_instruction="You are a judge.",
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )

        judge = SimpleJudge(judge_config=config_with_system)

        # System instruction should have format suffix
        assert judge.system_instruction is not None
        assert "JSON format only" in judge.system_instruction
        assert judge.system_instruction.startswith("You are a judge.")

        # Judgment prompt should NOT have format suffix
        prompt_with_system = judge._build_judgment_prompt({"text": "test"})
        assert prompt_with_system == "Rate: test"  # No format suffix here

        # Test case 2: Without system instruction - format suffix in judgment prompt
        config_without_system = JudgeConfig(
            judge_params=JudgeParams(
                prompt_template="Rate: {text}",
                response_format=JudgeResponseFormat.JSON,
                judgment_type=JudgeOutputType.BOOL,
                include_explanation=False,
                system_instruction=None,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.NATIVE,
                model=ModelParams(model_name="test-model"),
            ),
        )

        judge = SimpleJudge(judge_config=config_without_system)

        # System instruction should be None
        assert judge.system_instruction is None

        # Judgment prompt should have format suffix
        prompt_without_system = judge._build_judgment_prompt({"text": "test"})
        assert "JSON format only" in prompt_without_system
        assert prompt_without_system.startswith("Rate: test")

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_init_with_judge_config_path(self, mock_create_engine):
        """Test initialization with a built-in judge string."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        yaml_judge_config = """
judge_params:
  prompt_template: "Is the following Q&A correct? Question: {question} Answer: {answer}"
  response_format: JSON
  judgment_type: BOOL
  include_explanation: False
inference_config:
  model:
    model_name: "test-model"
  engine: NATIVE
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            judge_config_path = str(Path(temp_dir) / "judge_config.yaml")
            Path(judge_config_path).write_text(yaml_judge_config)

            judge = SimpleJudge(judge_config=judge_config_path)

            # Should resolve to the built-in config
            assert judge._judge_params is not None
            assert judge._judge_params.judgment_type == JudgeOutputType.BOOL
            assert judge._judge_params.include_explanation is False
            assert judge._judge_params.response_format == JudgeResponseFormat.JSON

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_init_with_unknown_judge_config_path(self, mock_create_engine):
        """Test initialization with an unknown built-in judge string raises error."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        with pytest.raises(
            ValueError, match="Could not resolve JudgeConfig from path: unknown_judge"
        ):
            SimpleJudge(judge_config="unknown_judge")
