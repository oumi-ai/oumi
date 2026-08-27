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
from unittest.mock import patch

import pytest

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.judge_config import JudgeConfig
from oumi.core.configs.params.judge_params import (
    JudgeOutputType,
    JudgeParams,
    JudgeResponseFormat,
)
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.rubric_judge_params import (
    JudgeAggregation,
    JudgeCriterion,
    RubricJudgeParams,
)
from oumi.exceptions import OumiConfigError
from oumi.judges.base_judge import JudgeOutput
from oumi.judges.rubric_judge import RubricJudge

TEST_PROMPT_TEMPLATE = "Question: {question}\nAnswer: {answer}"
TEST_INPUT = {"question": "What is 2+2?", "answer": "4"}
# Annotated because dict is invariant: a bare dict[str, float] is not assignable
# to a dict[str, float | None] parameter.
ENUM_SCORES: dict[str, float | None] = {"excellent": 1.0, "good": 0.5, "poor": 0.0}


def _criteria(include_explanation: bool = False) -> list[JudgeCriterion]:
    return [
        JudgeCriterion(
            id="correctness",
            description="The answer is correct.",
            judgment_type=JudgeOutputType.BOOL,
            include_explanation=include_explanation,
        ),
        JudgeCriterion(
            id="clarity",
            description="The answer is clearly written.",
            judgment_type=JudgeOutputType.ENUM,
            judgment_scores=ENUM_SCORES,
            include_explanation=include_explanation,
        ),
    ]


def _build_config(
    criteria: list[JudgeCriterion] | None = None,
    response_format: JudgeResponseFormat = JudgeResponseFormat.JSON,
    aggregation: JudgeAggregation = JudgeAggregation.WEIGHTED_MEAN,
    system_instruction: str | None = None,
    judge_params_overrides: dict | None = None,
    examples: list[dict[str, str]] | None = None,
) -> JudgeConfig:
    return JudgeConfig(
        judge_params=JudgeParams(
            prompt_template=TEST_PROMPT_TEMPLATE,
            system_instruction=system_instruction,
            response_format=response_format,
            examples=examples or [],
            **(judge_params_overrides or {}),
        ),
        rubric_judge_params=RubricJudgeParams(
            criteria=criteria if criteria is not None else _criteria(),
            aggregation=aggregation,
        ),
        inference_config=InferenceConfig(
            model=ModelParams(model_name="gpt-4o"),
            engine=InferenceEngineType.OPENAI,
        ),
    )


def _build_judge(**kwargs) -> RubricJudge:
    with patch("oumi.judges.rubric_judge.RubricJudge._create_inference_engine"):
        return RubricJudge(judge_config=_build_config(**kwargs))


class TestJudgeOutputFields:
    """Output field construction from criteria."""

    def test_one_field_per_criterion(self):
        judge = _build_judge()
        assert [f.field_key for f in judge.output_fields] == ["correctness", "clarity"]
        assert judge.output_fields[0].field_type == JudgeOutputType.BOOL
        assert judge.output_fields[0].field_scores is None
        assert judge.output_fields[1].field_type == JudgeOutputType.ENUM
        assert judge.output_fields[1].field_scores == ENUM_SCORES

    def test_explanation_precedes_its_judgment(self):
        """Each explanation must come first, so the judge reasons before judging."""
        judge = _build_judge(criteria=_criteria(include_explanation=True))
        assert [f.field_key for f in judge.output_fields] == [
            "correctness_explanation",
            "correctness",
            "clarity_explanation",
            "clarity",
        ]

    def test_explanation_is_opt_in_per_criterion(self):
        judge = _build_judge(
            criteria=[
                JudgeCriterion(id="a", description="A.", include_explanation=True),
                JudgeCriterion(id="b", description="B.", include_explanation=False),
            ]
        )
        assert [f.field_key for f in judge.output_fields] == [
            "a_explanation",
            "a",
            "b",
        ]


class TestRubricJudgePrompt:
    """Rubric block and response format instructions."""

    def test_rubric_block_lists_every_criterion(self):
        judge = _build_judge()
        prompt = judge._build_judgment_prompt(TEST_INPUT)

        assert "1. correctness: The answer is correct." in prompt
        assert "2. clarity: The answer is clearly written." in prompt
        assert "Your judgment should be a single word: 'Yes' or 'No'." in prompt
        assert (
            "Your judgment should be one of the following options: "
            "'excellent', 'good', 'poor'." in prompt
        )

    def test_rubric_block_mentions_explanation_fields(self):
        judge = _build_judge(criteria=_criteria(include_explanation=True))
        prompt = judge._build_judgment_prompt(TEST_INPUT)
        assert "Justify this judgment first, in 'correctness_explanation'." in prompt
        assert "Justify this judgment first, in 'clarity_explanation'." in prompt

    def test_json_suffix_lists_keys_in_order(self):
        judge = _build_judge(criteria=_criteria(include_explanation=True))
        prompt = judge._build_judgment_prompt(TEST_INPUT)
        assert (
            'the following keys, in this order: "correctness_explanation", '
            '"correctness", "clarity_explanation", "clarity".' in prompt
        )
        assert "Do not nest objects" in prompt

    def test_xml_suffix_lists_tags_in_order(self):
        judge = _build_judge(response_format=JudgeResponseFormat.XML)
        prompt = judge._build_judgment_prompt(TEST_INPUT)
        assert "<correctness></correctness>\n<clarity></clarity>" in prompt
        assert "XML format only" in prompt

    def test_suffix_goes_to_system_instruction_when_present(self):
        judge = _build_judge(system_instruction="You are an evaluator.")

        assert judge.system_instruction is not None
        assert judge.system_instruction.startswith("You are an evaluator.")
        assert "1. correctness" in judge.system_instruction

        # ...and not duplicated onto the judgment prompt.
        prompt = judge._build_judgment_prompt(TEST_INPUT)
        assert "1. correctness" not in prompt
        assert prompt == "Question: What is 2+2?\nAnswer: 4"

    def test_placeholders_are_resolved(self):
        judge = _build_judge()
        assert judge.prompt_template_placeholders == {"question", "answer"}
        assert "Question: What is 2+2?" in judge._build_judgment_prompt(TEST_INPUT)


class TestRubricJudgeResponseSchema:
    """JSON schema used for guided decoding."""

    def test_schema_is_flat_and_ordered(self):
        judge = _build_judge(criteria=_criteria(include_explanation=True))
        schema = judge._build_response_schema()

        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert list(schema["properties"].keys()) == [
            "correctness_explanation",
            "correctness",
            "clarity_explanation",
            "clarity",
        ]
        assert schema["required"] == list(schema["properties"].keys())
        # Flat: no property is itself an object.
        assert all(p["type"] != "object" for p in schema["properties"].values())

    def test_schema_field_types(self):
        judge = _build_judge(
            criteria=[
                JudgeCriterion(id="b", description="B.", include_explanation=True),
                JudgeCriterion(
                    id="i", description="I.", judgment_type=JudgeOutputType.INT
                ),
                JudgeCriterion(
                    id="f", description="F.", judgment_type=JudgeOutputType.FLOAT
                ),
                JudgeCriterion(
                    id="t", description="T.", judgment_type=JudgeOutputType.TEXT
                ),
                JudgeCriterion(
                    id="e",
                    description="E.",
                    judgment_type=JudgeOutputType.ENUM,
                    judgment_scores=ENUM_SCORES,
                ),
            ]
        )
        props = judge._build_response_schema()["properties"]

        assert props["b_explanation"] == {"type": "string"}
        assert props["b"] == {"type": "string", "enum": ["Yes", "No"]}
        assert props["i"] == {"type": "integer"}
        assert props["f"] == {"type": "number"}
        assert props["t"] == {"type": "string"}
        assert props["e"] == {"type": "string", "enum": list(ENUM_SCORES.keys())}

    def test_guided_decoding_enabled_for_json_only(self):
        for response_format, expects_schema in (
            (JudgeResponseFormat.JSON, True),
            (JudgeResponseFormat.XML, False),
        ):
            with patch(
                "oumi.judges.rubric_judge.RubricJudge._create_inference_engine"
            ) as mock_create:
                RubricJudge(judge_config=_build_config(response_format=response_format))
            schema = mock_create.call_args.kwargs["response_schema"]
            assert (schema is not None) == expects_schema


class TestRubricJudgeParsing:
    """Parsing model responses into per-criterion judgments."""

    def test_parse_json(self):
        judge = _build_judge(criteria=_criteria(include_explanation=True))
        raw = json.dumps(
            {
                "correctness_explanation": "2+2 is 4.",
                "correctness": "Yes",
                "clarity_explanation": "Terse but clear.",
                "clarity": "good",
            }
        )
        output = judge._transform_judge_output(raw)

        assert output.field_values == {
            "correctness_explanation": "2+2 is 4.",
            "correctness": True,
            "clarity_explanation": "Terse but clear.",
            "clarity": "good",
        }
        assert output.field_scores["correctness"] == 1.0
        assert output.field_scores["clarity"] == 0.5

    def test_parse_xml(self):
        judge = _build_judge(response_format=JudgeResponseFormat.XML)
        raw = "<correctness>No</correctness>\n<clarity>poor</clarity>"
        output = judge._transform_judge_output(raw)

        assert output.field_values == {"correctness": False, "clarity": "poor"}
        assert output.field_scores == {"correctness": 0.0, "clarity": 0.0}

    def test_parse_mixed_types(self):
        judge = _build_judge(
            criteria=[
                JudgeCriterion(
                    id="rating", description="R.", judgment_type=JudgeOutputType.INT
                ),
                JudgeCriterion(
                    id="confidence",
                    description="C.",
                    judgment_type=JudgeOutputType.FLOAT,
                ),
                JudgeCriterion(
                    id="notes", description="N.", judgment_type=JudgeOutputType.TEXT
                ),
            ],
            aggregation=JudgeAggregation.NONE,
        )
        raw = json.dumps({"rating": 4, "confidence": 0.75, "notes": "looks fine"})
        output = judge._transform_judge_output(raw)

        assert output.field_values == {
            "rating": 4,
            "confidence": 0.75,
            "notes": "looks fine",
        }
        # Types without a score mapping produce no numeric score.
        assert output.field_scores == {
            "rating": None,
            "confidence": None,
            "notes": None,
        }

    def test_partial_response_keeps_parsed_criteria(self):
        """Some criteria present is a success, with the rest left as None."""
        judge = _build_judge()
        output = judge._transform_judge_output(json.dumps({"correctness": "Yes"}))

        assert output.field_values == {"correctness": True, "clarity": None}
        assert output.field_scores["clarity"] is None
        # Aggregate over the one criterion that did parse.
        assert output.aggregate_score == 1.0

    @pytest.mark.parametrize(
        "raw",
        [
            "I think the answer is pretty good actually.",
            '{"correctness": "Yes"',  # truncated JSON
            "",
            '{"unrelated_key": "Yes"}',
        ],
    )
    def test_unparseable_response_reports_none_and_warns(self, raw, caplog):
        """Matches how a single-judgment judge reports the same situation.

        The row still comes back, with every criterion None, so callers detect it
        from the values rather than from an exception. A warning names the cause.
        """
        judge = _build_judge()
        with caplog.at_level("WARNING"):
            output = judge._transform_judge_output(raw)

        assert output.field_values == {"correctness": None, "clarity": None}
        assert output.field_scores == {"correctness": None, "clarity": None}
        assert output.aggregate_score is None
        assert output.raw_output == raw
        assert "No criteria could be parsed" in caplog.text

    def test_thinking_tags_are_stripped(self):
        judge = _build_judge()
        raw = "<think>Let me work through this.</think>" + json.dumps(
            {"correctness": "Yes", "clarity": "excellent"}
        )
        output = judge._transform_judge_output(raw)
        assert output.field_values["correctness"] is True
        assert output.raw_output == raw


class TestRubricJudgeBatchResilience:
    """One unparseable response must not disturb the rest of the batch."""

    def _judge_replaying(self, raw_outputs: list[str]) -> RubricJudge:
        from unittest.mock import Mock

        from oumi.core.types.conversation import Conversation, Message, Role

        engine = Mock()
        engine.infer.return_value = [
            Conversation(
                messages=[
                    Message(content="p", role=Role.USER),
                    Message(content=raw, role=Role.ASSISTANT),
                ]
            )
            for raw in raw_outputs
        ]
        with patch(
            "oumi.judges.rubric_judge.RubricJudge._create_inference_engine",
            return_value=engine,
        ):
            return RubricJudge(judge_config=_build_config())

    def test_judge_returns_plain_judge_outputs_carrying_the_aggregate(self):
        """aggregate_score lives on JudgeOutput, so no subclass or cast is needed."""
        judge = self._judge_replaying([json.dumps({"correctness": "Yes"})])
        outputs = judge.judge(inputs=[TEST_INPUT])
        assert all(type(o) is JudgeOutput for o in outputs)
        assert outputs[0].aggregate_score == 1.0

    def test_one_bad_row_does_not_disturb_the_good_ones(self, caplog):
        good = json.dumps({"correctness": "Yes", "clarity": "good"})
        truncated = '{"correctness": "Ye'
        judge = self._judge_replaying([good, truncated, good])

        with caplog.at_level("WARNING"):
            outputs = judge.judge(inputs=[TEST_INPUT] * 3)

        assert len(outputs) == 3
        assert outputs[0].aggregate_score == pytest.approx(0.75)
        assert outputs[2].aggregate_score == pytest.approx(0.75)

        assert outputs[1].field_values == {"correctness": None, "clarity": None}
        assert outputs[1].aggregate_score is None
        assert outputs[1].raw_output == truncated
        assert "No criteria could be parsed" in caplog.text

    def test_judge_partial_reports_an_unparseable_row_as_successful(self):
        """Consistent with SimpleJudge: the None values are the failure signal."""
        from unittest.mock import Mock

        from oumi.core.inference.base_inference_engine import InferenceResult
        from oumi.core.types.conversation import Conversation, Message, Role

        good = json.dumps({"correctness": "Yes", "clarity": "good"})
        raws = [good, "total garbage"]
        mock_engine = Mock()
        mock_engine.infer_partial.return_value = InferenceResult(
            successful=[
                (
                    idx,
                    Conversation(
                        messages=[
                            Message(content="p", role=Role.USER),
                            Message(content=raw, role=Role.ASSISTANT),
                        ]
                    ),
                )
                for idx, raw in enumerate(raws)
            ],
            failures={},
        )
        with patch(
            "oumi.judges.rubric_judge.RubricJudge._create_inference_engine",
            return_value=mock_engine,
        ):
            judge = RubricJudge(judge_config=_build_config())

        result = judge.judge_partial([TEST_INPUT, TEST_INPUT])

        assert not result.has_failures
        assert [idx for idx, _ in result.successful] == [0, 1]
        assert result.successful[0][1].aggregate_score == pytest.approx(0.75)
        # The unparseable row is "successful" but carries no judgments.
        assert result.successful[1][1].field_values == {
            "correctness": None,
            "clarity": None,
        }
        assert result.successful[1][1].aggregate_score is None


class TestRubricJudgeAggregation:
    """Combining per-criterion scores into an overall score."""

    def _judge_with(self, aggregation, weights=(1.0, 1.0)):
        criteria = _criteria()
        criteria[0].weight, criteria[1].weight = weights
        return _build_judge(criteria=criteria, aggregation=aggregation)

    def test_weighted_mean(self):
        judge = self._judge_with(JudgeAggregation.WEIGHTED_MEAN, weights=(3.0, 1.0))
        output = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "clarity": "poor"})
        )
        # (3 * 1.0 + 1 * 0.0) / 4
        assert output.aggregate_score == pytest.approx(0.75)

    def test_weighted_mean_uniform_weights(self):
        judge = self._judge_with(JudgeAggregation.WEIGHTED_MEAN)
        output = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "clarity": "good"})
        )
        assert output.aggregate_score == pytest.approx(0.75)

    def test_min(self):
        judge = self._judge_with(JudgeAggregation.MIN)
        output = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "clarity": "good"})
        )
        assert output.aggregate_score == 0.5

    def test_all(self):
        judge = self._judge_with(JudgeAggregation.ALL)

        all_pass = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "clarity": "excellent"})
        )
        assert all_pass.aggregate_score == 1.0

        one_fails = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "clarity": "good"})
        )
        assert one_fails.aggregate_score == 0.0

    def test_none(self):
        judge = self._judge_with(JudgeAggregation.NONE)
        output = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "clarity": "good"})
        )
        assert output.aggregate_score is None

    def test_zero_weight_criterion_is_excluded(self):
        judge = self._judge_with(JudgeAggregation.WEIGHTED_MEAN, weights=(1.0, 0.0))
        output = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "clarity": "poor"})
        )
        assert output.aggregate_score == 1.0

    def test_unparseable_bool_is_excluded_not_scored_zero(self):
        """The base parser scores an unparseable bool 0.0; that must not aggregate.

        Otherwise a garbled value is indistinguishable from a genuine failing
        judgment, and silently drags the aggregate down.
        """
        judge = _build_judge(
            criteria=[
                JudgeCriterion(id="a", description="A."),
                JudgeCriterion(id="b", description="B."),
            ]
        )
        output = judge._transform_judge_output(json.dumps({"a": "Yes", "b": "maybe"}))

        assert output.field_values["b"] is None
        # The base parser still reports 0.0 for the unparseable boolean...
        assert output.field_scores["b"] == 0.0
        # ...but aggregation ignores it, rather than reading it as a real failure.
        assert output.aggregate_score == 1.0

    def test_no_scoreable_criteria_yields_none(self):
        judge = _build_judge(
            criteria=[
                JudgeCriterion(
                    id="notes", description="N.", judgment_type=JudgeOutputType.TEXT
                )
            ]
        )
        output = judge._transform_judge_output(json.dumps({"notes": "some notes"}))
        assert output.aggregate_score is None


class TestRubricJudgeUnscoredLabels:
    """An ENUM label may map to None, meaning "carries no score" (e.g. N/A)."""

    NA_SCORES: dict[str, float | None] = {
        "good": 1.0,
        "poor": 0.0,
        "not_applicable": None,
    }

    def _judge(self, aggregation=JudgeAggregation.WEIGHTED_MEAN, na_weight=3.0):
        return _build_judge(
            criteria=[
                JudgeCriterion(id="correctness", description="C.", weight=1.0),
                JudgeCriterion(
                    id="citations",
                    description="How well the answer cites sources.",
                    judgment_type=JudgeOutputType.ENUM,
                    judgment_scores=self.NA_SCORES,
                    weight=na_weight,
                ),
            ],
            aggregation=aggregation,
        )

    def test_config_accepts_a_none_score(self):
        criterion = JudgeCriterion(
            id="c",
            description="D.",
            judgment_type=JudgeOutputType.ENUM,
            judgment_scores=self.NA_SCORES,
        )
        assert criterion.judgment_scores is not None
        assert criterion.judgment_scores["not_applicable"] is None

    def test_non_numeric_non_none_scores_still_rejected(self):
        with pytest.raises(OumiConfigError, match="must be numeric, or None"):
            JudgeCriterion(
                id="c",
                description="D.",
                judgment_type=JudgeOutputType.ENUM,
                judgment_scores={"good": "high"},  # type: ignore[dict-item]
            )

    @pytest.mark.parametrize(
        "judgment_type",
        [
            JudgeOutputType.BOOL,
            JudgeOutputType.INT,
            JudgeOutputType.FLOAT,
            JudgeOutputType.TEXT,
        ],
    )
    def test_none_scores_rejected_for_non_enum_types(self, judgment_type):
        """Only ENUM keeps the chosen label as its value.

        For the other types the value is parsed out of the label, so an unscored
        label would come back as None -- indistinguishable from a failed parse.
        Rejecting at config time beats silently losing the label at runtime.
        """
        with pytest.raises(OumiConfigError, match="only supported for ENUM"):
            JudgeCriterion(
                id="c",
                description="D.",
                judgment_type=judgment_type,
                judgment_scores={"a": 1.0, "na": None},
            )

    @pytest.mark.parametrize(
        "judgment_type",
        [
            JudgeOutputType.ENUM,
            JudgeOutputType.BOOL,
            JudgeOutputType.INT,
            JudgeOutputType.FLOAT,
            JudgeOutputType.TEXT,
        ],
    )
    def test_all_float_scores_still_allowed_on_every_type(self, judgment_type):
        """The restriction is on None values only, not on judgment_scores itself."""
        criterion = JudgeCriterion(
            id="c",
            description="D.",
            judgment_type=judgment_type,
            judgment_scores={"a": 1.0},
        )
        assert criterion.judgment_scores == {"a": 1.0}

    def test_unscored_label_is_distinguishable_from_an_invalid_one(self):
        """N/A keeps its label; a label the judge invented parses to None."""
        judge = self._judge()

        na = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "citations": "not_applicable"})
        )
        invalid = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "citations": "hallucinated"})
        )

        assert na.field_values["citations"] == "not_applicable"
        assert invalid.field_values["citations"] is None
        # Both carry no score, so both are excluded from the aggregate.
        assert na.field_scores["citations"] is None
        assert invalid.field_scores["citations"] is None

    def test_unscored_label_is_offered_to_the_model(self):
        """The judge can only pick N/A if the prompt and schema allow it."""
        judge = self._judge()
        prompt = judge._build_judgment_prompt(TEST_INPUT)
        assert "'good', 'poor', 'not_applicable'" in prompt
        assert judge._build_response_schema()["properties"]["citations"]["enum"] == [
            "good",
            "poor",
            "not_applicable",
        ]

    def test_unscored_label_parses_to_the_label_with_no_score(self):
        judge = self._judge()
        output = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "citations": "not_applicable"})
        )
        # The label is preserved for the reader...
        assert output.field_values["citations"] == "not_applicable"
        # ...but it carries no numeric score.
        assert output.field_scores["citations"] is None

    def test_unscored_label_drops_out_of_the_weighted_mean_entirely(self):
        """Its weight must leave the denominator too, not just the numerator."""
        judge = self._judge(na_weight=3.0)

        scored = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "citations": "poor"})
        )
        # (1*1.0 + 3*0.0) / 4
        assert scored.aggregate_score == pytest.approx(0.25)

        na = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "citations": "not_applicable"})
        )
        # Only correctness counts: 1.0 / 1. If the weight had stayed in the
        # denominator this would be 0.25 instead.
        assert na.aggregate_score == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "aggregation,expected",
        [(JudgeAggregation.MIN, 1.0), (JudgeAggregation.ALL, 1.0)],
    )
    def test_unscored_label_excluded_from_min_and_all(self, aggregation, expected):
        judge = self._judge(aggregation=aggregation)
        output = judge._transform_judge_output(
            json.dumps({"correctness": "Yes", "citations": "not_applicable"})
        )
        assert output.aggregate_score == expected

    def test_every_criterion_unscored_yields_no_aggregate(self):
        judge = _build_judge(
            criteria=[
                JudgeCriterion(
                    id="citations",
                    description="D.",
                    judgment_type=JudgeOutputType.ENUM,
                    judgment_scores=self.NA_SCORES,
                )
            ]
        )
        output = judge._transform_judge_output(
            json.dumps({"citations": "not_applicable"})
        )
        assert output.field_values["citations"] == "not_applicable"
        assert output.aggregate_score is None

    def test_all_labels_unscored_is_warned_about(self, caplog):
        """Such a criterion constrains the options but can never be scored."""
        with caplog.at_level("WARNING"):
            _build_judge(
                criteria=[
                    JudgeCriterion(id="ok", description="A."),
                    JudgeCriterion(
                        id="category",
                        description="Categorical only.",
                        judgment_type=JudgeOutputType.ENUM,
                        judgment_scores={"alpha": None, "beta": None},
                    ),
                ]
            )
        assert "['category']" in caplog.text
        assert "produce no numeric score" in caplog.text

    def test_partially_scored_enum_is_not_warned_about(self, caplog):
        with caplog.at_level("WARNING"):
            self._judge()
        assert "produce no numeric score" not in caplog.text


class TestRubricJudgeExamples:
    """Few-shot examples must round-trip through the multi-field output format."""

    def test_examples_build_assistant_responses(self):
        examples = [
            {
                "question": "What is 2+2?",
                "answer": "4",
                "correctness": "Yes",
                "clarity": "excellent",
            }
        ]
        judge = _build_judge(examples=examples)
        conversations = judge.build_conversations([TEST_INPUT])

        # system + (example user, example assistant) + judgment user
        messages = conversations[0].messages
        assert len(messages) == 3
        assert json.loads(str(messages[1].content)) == {
            "correctness": "Yes",
            "clarity": "excellent",
        }

    def test_example_missing_a_criterion_raises(self):
        examples = [{"question": "q", "answer": "a", "correctness": "Yes"}]
        judge = _build_judge(examples=examples)
        with pytest.raises(ValueError, match="Missing values for required output"):
            judge.build_conversations([TEST_INPUT])


class TestRubricJudgeValidation:
    """Config-level rejections at judge construction time."""

    def test_missing_rubric_params_raises(self):
        config = _build_config()
        config.rubric_judge_params = None
        with pytest.raises(ValueError, match="rubric_judge_params must be provided"):
            RubricJudge(judge_config=config)

    def test_missing_inference_config_raises(self):
        config = _build_config()
        config.inference_config = None
        with pytest.raises(ValueError, match="inference_config must be provided"):
            RubricJudge(judge_config=config)

    def test_raw_response_format_raises(self):
        with pytest.raises(ValueError, match="does not support the RAW response"):
            _build_judge(response_format=JudgeResponseFormat.RAW)

    @pytest.mark.parametrize(
        "overrides,expected",
        [
            ({"judgment_type": JudgeOutputType.TEXT}, "judge_params.judgment_type"),
            ({"include_explanation": True}, "judge_params.include_explanation"),
            (
                {
                    "judgment_type": JudgeOutputType.ENUM,
                    "judgment_scores": ENUM_SCORES,
                },
                "judge_params.judgment_type, judge_params.judgment_scores",
            ),
        ],
    )
    def test_single_judgment_params_are_rejected(self, overrides, expected):
        """These judge_params would be silently ignored, so reject them loudly."""
        with pytest.raises(ValueError, match=expected.replace(".", r"\.")):
            _build_judge(judge_params_overrides=overrides)

    def test_unscored_criteria_are_warned_about(self, caplog):
        with caplog.at_level("WARNING"):
            _build_judge(
                criteria=[
                    JudgeCriterion(id="ok", description="O."),
                    JudgeCriterion(
                        id="rating",
                        description="R.",
                        judgment_type=JudgeOutputType.INT,
                    ),
                ]
            )
        assert "['rating']" in caplog.text
        assert "excluded from the 'weighted_mean' aggregate score" in caplog.text

    @pytest.mark.parametrize(
        "aggregation",
        [JudgeAggregation.MIN, JudgeAggregation.ALL, JudgeAggregation.NONE],
    )
    def test_weights_ignored_by_aggregation_are_warned_about(self, aggregation, caplog):
        """Only WEIGHTED_MEAN reads `weight`; elsewhere it is a silent no-op."""
        with caplog.at_level("WARNING"):
            _build_judge(
                criteria=[
                    JudgeCriterion(id="a", description="A.", weight=3.0),
                    JudgeCriterion(id="b", description="B."),
                ],
                aggregation=aggregation,
            )
        assert "['a']" in caplog.text
        assert f"'{aggregation.value}' aggregation ignores weights" in caplog.text

    def test_no_weight_warning_under_weighted_mean(self, caplog):
        with caplog.at_level("WARNING"):
            _build_judge(
                criteria=[JudgeCriterion(id="a", description="A.", weight=3.0)],
                aggregation=JudgeAggregation.WEIGHTED_MEAN,
            )
        assert "ignores weights" not in caplog.text

    def test_default_weights_are_a_plain_mean(self):
        """WEIGHTED_MEAN with default weights is exactly the arithmetic mean."""
        from statistics import mean

        judge = _build_judge(
            criteria=[
                JudgeCriterion(id="a", description="A."),
                JudgeCriterion(id="b", description="B."),
                JudgeCriterion(id="c", description="C."),
            ]
        )
        output = judge._transform_judge_output(
            json.dumps({"a": "Yes", "b": "Yes", "c": "No"})
        )
        assert output.aggregate_score == mean([1.0, 1.0, 0.0])

    def test_no_warning_when_aggregation_disabled(self, caplog):
        with caplog.at_level("WARNING"):
            _build_judge(
                criteria=[
                    JudgeCriterion(
                        id="rating",
                        description="R.",
                        judgment_type=JudgeOutputType.INT,
                    )
                ],
                aggregation=JudgeAggregation.NONE,
            )
        assert "excluded from" not in caplog.text


class TestRubricJudgeParamsValidation:
    """Validation inside RubricJudgeParams / JudgeCriterion."""

    def test_empty_criteria_raises(self):
        with pytest.raises(OumiConfigError, match="`criteria` cannot be empty"):
            RubricJudgeParams(criteria=[])

    def test_duplicate_ids_raise(self):
        with pytest.raises(OumiConfigError, match="Duplicate criterion id: 'a'"):
            RubricJudgeParams(
                criteria=[
                    JudgeCriterion(id="a", description="A."),
                    JudgeCriterion(id="a", description="Also A."),
                ]
            )

    @pytest.mark.parametrize("key", ["tone_explanation", "explanation"])
    def test_reserved_explanation_ids_raise(self, key):
        """The explanation namespace belongs to include_explanation, not to users.

        It would otherwise be ambiguous whether `tone_explanation` is a criterion of
        its own or the explanation generated for a `tone` criterion.
        """
        with pytest.raises(OumiConfigError, match="is reserved"):
            JudgeCriterion(id=key, description="D.")

    @pytest.mark.parametrize(
        "key",
        [
            "with-dash",
            "with space",
            "with.dot",
            "with/slash",
            "123",  # a tag name cannot start with a digit
            "1st_criterion",
            "",
        ],
    )
    def test_invalid_id_charset_raises(self, key):
        """Ids become XML tag names, so they must be identifier-like."""
        with pytest.raises(OumiConfigError, match="is invalid|cannot be empty"):
            JudgeCriterion(id=key, description="D.")

    @pytest.mark.parametrize("key", ["_leading", "a1", "snake_case", "CamelCase"])
    def test_valid_ids_accepted(self, key):
        assert JudgeCriterion(id=key, description="D.").id == key

    def test_empty_id_raises(self):
        with pytest.raises(OumiConfigError, match="`id` cannot be empty"):
            JudgeCriterion(id="  ", description="D.")

    def test_empty_description_raises(self):
        with pytest.raises(OumiConfigError, match="non-empty `description`"):
            JudgeCriterion(id="a", description="  ")

    def test_enum_without_scores_raises(self):
        with pytest.raises(OumiConfigError, match="must be provided for ENUM"):
            JudgeCriterion(id="a", description="A.", judgment_type=JudgeOutputType.ENUM)

    def test_empty_scores_raise(self):
        with pytest.raises(OumiConfigError, match="cannot be empty when provided"):
            JudgeCriterion(
                id="a",
                description="A.",
                judgment_type=JudgeOutputType.BOOL,
                judgment_scores={},
            )

    def test_non_numeric_scores_raise(self):
        with pytest.raises(OumiConfigError, match="must be numeric"):
            JudgeCriterion(
                id="a",
                description="A.",
                judgment_type=JudgeOutputType.ENUM,
                judgment_scores={"good": "high"},  # type: ignore[dict-item]
            )

    def test_negative_weight_raises(self):
        with pytest.raises(OumiConfigError, match="weight must be non-negative"):
            JudgeCriterion(id="a", description="A.", weight=-1.0)

    def test_all_zero_weights_raise_under_weighted_mean(self):
        with pytest.raises(OumiConfigError, match="all weights are zero"):
            RubricJudgeParams(
                criteria=[JudgeCriterion(id="a", description="A.", weight=0.0)],
                aggregation=JudgeAggregation.WEIGHTED_MEAN,
            )

    def test_all_zero_weights_allowed_without_weighted_mean(self):
        params = RubricJudgeParams(
            criteria=[JudgeCriterion(id="a", description="A.", weight=0.0)],
            aggregation=JudgeAggregation.MIN,
        )
        assert params.aggregation == JudgeAggregation.MIN


class TestRubricJudgeDispatch:
    """_create_judge() routing based on which judge params the config carries."""

    def test_rubric_params_select_rubric_judge(self):
        from oumi.judge import _create_judge

        with patch("oumi.judges.rubric_judge.RubricJudge._create_inference_engine"):
            judge = _create_judge(_build_config())
        assert isinstance(judge, RubricJudge)

    def test_conflicting_params_raise(self):
        from oumi.core.configs.params.rule_judge_params import RuleJudgeParams
        from oumi.judge import _create_judge

        config = _build_config()
        config.rule_judge_params = RuleJudgeParams(
            rule_type="regex", input_fields=["answer"]
        )
        with pytest.raises(ValueError, match="which select different judges"):
            _create_judge(config)
