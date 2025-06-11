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

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.judge_config_v2 import (
    JudgeConfig,
    JudgeOutputType,
    JudgeResponseFormat,
)
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
            prompt_template="Is this helpful? Question: {question}, Answer: {answer}",
            response_format=JudgeResponseFormat.XML,
            judgment_type=JudgeOutputType.BOOL,
            include_explanation=False,
        )

    @pytest.fixture
    def xml_config_with_explanation(self):
        return JudgeConfig(
            prompt_template="Is this helpful? Question: {question}, Answer: {answer}",
            response_format=JudgeResponseFormat.XML,
            judgment_type=JudgeOutputType.BOOL,
            include_explanation=True,
        )

    @pytest.fixture
    def json_config_no_explanation(self):
        return JudgeConfig(
            prompt_template="Rate this answer: {answer}",
            response_format=JudgeResponseFormat.JSON,
            judgment_type=JudgeOutputType.ENUM,
            judgment_scores=self.TEST_ENUM_SCORES,
            include_explanation=False,
        )

    @pytest.fixture
    def json_config_with_explanation(self):
        return JudgeConfig(
            prompt_template="Rate this answer: {answer}",
            response_format=JudgeResponseFormat.JSON,
            judgment_type=JudgeOutputType.ENUM,
            judgment_scores=self.TEST_ENUM_SCORES,
            include_explanation=True,
        )

    @pytest.fixture
    def mock_inference_engine(self):
        return Mock()

    @pytest.fixture
    def mock_inference_config(self):
        from oumi.core.configs.inference_engine_type import InferenceEngineType
        from oumi.core.configs.params.model_params import ModelParams

        return InferenceConfig(
            engine=InferenceEngineType.NATIVE,
            model=ModelParams(model_name="test-model"),
        )

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_init_with_engine_no_explanation(
        self, mock_create_engine, xml_config_no_explanation, mock_inference_config
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(
            judge_config=xml_config_no_explanation,
            inference_config=mock_inference_config,
        )

        assert judge._judge_config == xml_config_no_explanation
        assert judge.inference_engine == mock_engine
        assert judge.prompt_template == xml_config_no_explanation.prompt_template
        assert judge.response_format == xml_config_no_explanation.response_format

        # Should have one output field for judgment
        assert len(judge.output_fields) == 1
        assert judge.output_fields[0].field_key == JUDGMENT_KEY
        assert judge.output_fields[0].field_type == JudgeOutputType.BOOL
        assert judge.output_fields[0].field_scores is None

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_init_with_engine_explanation(
        self, mock_create_engine, json_config_with_explanation, mock_inference_config
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(
            judge_config=json_config_with_explanation,
            inference_config=mock_inference_config,
        )

        # Should have two output fields: judgment and explanation
        assert len(judge.output_fields) == 2
        assert judge.output_fields[0].field_key == JUDGMENT_KEY
        assert judge.output_fields[0].field_type == JudgeOutputType.ENUM
        assert judge.output_fields[0].field_scores == {
            "excellent": 1.0,
            "good": 0.7,
            "poor": 0.3,
        }
        assert judge.output_fields[1].field_key == EXPLANATION_KEY
        assert judge.output_fields[1].field_type == JudgeOutputType.TEXT
        assert judge.output_fields[1].field_scores is None

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_init_with_inference_config(
        self, mock_create_engine, xml_config_no_explanation, mock_inference_config
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(
            judge_config=xml_config_no_explanation,
            inference_config=mock_inference_config,
        )

        assert judge.inference_engine == mock_engine
        mock_create_engine.assert_called_once_with(mock_inference_config)

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_build_prompt_xml_no_explanation(
        self, mock_create_engine, xml_config_no_explanation, mock_inference_config
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(
            judge_config=xml_config_no_explanation,
            inference_config=mock_inference_config,
        )

        prompt = judge._build_judgment_prompt(self.XML_JUDGE_INPUT)

        expected = (
            f"Is this helpful? Question: What is 2+2?, Answer: 4"
            f"{self.XML_SUFFIX_FORMATTED_BOOL}"
        )
        assert prompt == expected

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_build_prompt_xml_with_explanation(
        self, mock_create_engine, xml_config_with_explanation, mock_inference_config
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(
            judge_config=xml_config_with_explanation,
            inference_config=mock_inference_config,
        )

        prompt = judge._build_judgment_prompt(self.XML_JUDGE_INPUT)

        expected = (
            f"Is this helpful? Question: What is 2+2?, Answer: 4"
            f"{self.XML_SUFFIX_WITH_EXPLANATION_FORMATTED_BOOL}"
        )
        assert prompt == expected

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_build_prompt_json_no_explanation(
        self, mock_create_engine, json_config_no_explanation, mock_inference_config
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(
            judge_config=json_config_no_explanation,
            inference_config=mock_inference_config,
        )

        prompt = judge._build_judgment_prompt(self.JSON_JUDGE_INPUT)

        expected = (
            f"Rate this answer: Some good answer{self.JSON_SUFFIX_FORMATTED_ENUM}"
        )
        assert prompt == expected

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_build_prompt_json_with_explanation(
        self, mock_create_engine, json_config_with_explanation, mock_inference_config
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(
            judge_config=json_config_with_explanation,
            inference_config=mock_inference_config,
        )

        prompt = judge._build_judgment_prompt(self.JSON_JUDGE_INPUT)

        expected = (
            f"Rate this answer: Some good answer"
            f"{self.JSON_SUFFIX_WITH_EXPLANATION_FORMATTED_ENUM}"
        )
        assert prompt == expected

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_build_prompt_raw_no_explanation(
        self, mock_create_engine, mock_inference_config
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        config = JudgeConfig(
            prompt_template="Evaluate: {content}",
            response_format=JudgeResponseFormat.RAW,
            judgment_type=JudgeOutputType.TEXT,
            include_explanation=False,
        )
        judge = SimpleJudge(judge_config=config, inference_config=mock_inference_config)

        prompt = judge._build_judgment_prompt(self.RAW_JUDGE_INPUT)

        expected = "Evaluate: Some content"  # No suffix for RAW without explanation
        assert prompt == expected

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_build_prompt_raw_with_explanation(
        self, mock_create_engine, mock_inference_config
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        config = JudgeConfig(
            prompt_template="Evaluate: {content}",
            response_format=JudgeResponseFormat.RAW,
            judgment_type=JudgeOutputType.TEXT,
            include_explanation=True,
        )
        judge = SimpleJudge(judge_config=config, inference_config=mock_inference_config)

        prompt = judge._build_judgment_prompt(self.RAW_JUDGE_INPUT)

        expected = (
            f"Evaluate: Some content{self.RAW_SUFFIX_WITH_EXPLANATION_FORMATTED_TEXT}"
        )
        assert prompt == expected

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_get_format_suffix_xml(self, mock_create_engine, mock_inference_config):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        config_no_exp = JudgeConfig(
            prompt_template=self.TEST_PROMPT_TEMPLATE,
            response_format=JudgeResponseFormat.XML,
            include_explanation=False,
        )
        judge_no_exp = SimpleJudge(
            judge_config=config_no_exp, inference_config=mock_inference_config
        )
        assert judge_no_exp._get_format_suffix() == self.XML_SUFFIX_FORMATTED_BOOL

        config_with_exp = JudgeConfig(
            prompt_template=self.TEST_PROMPT_TEMPLATE,
            response_format=JudgeResponseFormat.XML,
            include_explanation=True,
        )
        judge_with_exp = SimpleJudge(
            judge_config=config_with_exp, inference_config=mock_inference_config
        )
        assert (
            judge_with_exp._get_format_suffix()
            == self.XML_SUFFIX_WITH_EXPLANATION_FORMATTED_BOOL
        )

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_get_format_suffix_json(self, mock_create_engine, mock_inference_config):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        config_no_exp = JudgeConfig(
            prompt_template=self.TEST_PROMPT_TEMPLATE,
            response_format=JudgeResponseFormat.JSON,
            judgment_type=JudgeOutputType.ENUM,
            judgment_scores=self.TEST_ENUM_SCORES,
            include_explanation=False,
        )
        judge_no_exp = SimpleJudge(
            judge_config=config_no_exp, inference_config=mock_inference_config
        )
        assert judge_no_exp._get_format_suffix() == self.JSON_SUFFIX_FORMATTED_ENUM

        config_with_exp = JudgeConfig(
            prompt_template=self.TEST_PROMPT_TEMPLATE,
            response_format=JudgeResponseFormat.JSON,
            judgment_type=JudgeOutputType.ENUM,
            judgment_scores=self.TEST_ENUM_SCORES,
            include_explanation=True,
        )
        judge_with_exp = SimpleJudge(
            judge_config=config_with_exp, inference_config=mock_inference_config
        )
        assert (
            judge_with_exp._get_format_suffix()
            == self.JSON_SUFFIX_WITH_EXPLANATION_FORMATTED_ENUM
        )

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_get_format_suffix_raw(self, mock_create_engine, mock_inference_config):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        config_no_exp = JudgeConfig(
            prompt_template=self.TEST_PROMPT_TEMPLATE,
            response_format=JudgeResponseFormat.RAW,
            judgment_type=JudgeOutputType.TEXT,
            include_explanation=False,
        )
        judge_no_exp = SimpleJudge(
            judge_config=config_no_exp, inference_config=mock_inference_config
        )
        assert judge_no_exp._get_format_suffix() == ""

        config_with_exp = JudgeConfig(
            prompt_template=self.TEST_PROMPT_TEMPLATE,
            response_format=JudgeResponseFormat.RAW,
            judgment_type=JudgeOutputType.TEXT,
            include_explanation=True,
        )
        judge_with_exp = SimpleJudge(
            judge_config=config_with_exp, inference_config=mock_inference_config
        )
        assert (
            judge_with_exp._get_format_suffix()
            == self.RAW_SUFFIX_WITH_EXPLANATION_FORMATTED_TEXT
        )

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_create_judgment_output_field(
        self, mock_create_engine, xml_config_no_explanation, mock_inference_config
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(
            judge_config=xml_config_no_explanation,
            inference_config=mock_inference_config,
        )

        field = judge._create_judgment_output_field(xml_config_no_explanation)

        assert field.field_key == JUDGMENT_KEY
        assert field.field_type == JudgeOutputType.BOOL
        assert field.field_scores is None

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_create_judgment_output_field_with_scores(
        self, mock_create_engine, json_config_with_explanation, mock_inference_config
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(
            judge_config=json_config_with_explanation,
            inference_config=mock_inference_config,
        )

        field = judge._create_judgment_output_field(json_config_with_explanation)

        assert field.field_key == JUDGMENT_KEY
        assert field.field_type == JudgeOutputType.ENUM
        assert field.field_scores == {"excellent": 1.0, "good": 0.7, "poor": 0.3}

    @patch("oumi.judges_v2.simple_judge.SimpleJudge._create_inference_engine")
    def test_create_explanation_output_field(
        self, mock_create_engine, json_config_with_explanation, mock_inference_config
    ):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        judge = SimpleJudge(
            judge_config=json_config_with_explanation,
            inference_config=mock_inference_config,
        )

        field = judge._create_explanation_output_field()

        assert field.field_key == EXPLANATION_KEY
        assert field.field_type == JudgeOutputType.TEXT
        assert field.field_scores is None

    @patch("oumi.builders.inference_engines.build_inference_engine")
    def test_create_inference_engine(
        self, mock_build_engine, xml_config_no_explanation, mock_inference_config
    ):
        mock_engine = Mock()
        mock_build_engine.return_value = mock_engine

        _ = SimpleJudge(
            judge_config=xml_config_no_explanation,
            inference_config=mock_inference_config,
        )

        # Should be called during init with inference_config
        mock_build_engine.assert_called_once_with(
            engine_type=mock_inference_config.engine,
            model_params=mock_inference_config.model,
            remote_params=mock_inference_config.remote_params,
            generation_params=mock_inference_config.generation,
        )
