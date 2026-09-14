import os

import pytest

from oumi.core.configs.judge_config import JudgeConfig
from oumi.judges.rubric_judge import RubricJudge

skip_if_no_openai_key = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY not set"
)

YAML_RUBRIC_CONFIG_JSON = """
judge_params:
    prompt_template: "Statement: {statement}"
    response_format: JSON
rubric_judge_params:
    aggregation: WEIGHTED_MEAN
    criteria:
        - id: correctness
          description: The statement is factually correct.
          judgment_type: BOOL
          include_explanation: true
          weight: 2.0
        - id: specificity
          description: How specific and concrete the statement is.
          judgment_type: ENUM
          judgment_scores:
              specific: 1.0
              vague: 0.0
          weight: 1.0
inference_config:
    model:
        model_name: "gpt-4.1"
    engine: OPENAI
    generation:
        max_new_tokens: 4096
        temperature: 0.0
"""

YAML_RUBRIC_CONFIG_XML = YAML_RUBRIC_CONFIG_JSON.replace(
    "response_format: JSON", "response_format: XML"
)

JUDGE_DATASET = [
    {"statement": "The capital of France is Paris."},
    {"statement": "The Earth is flat."},
]

EXPECTED_KEYS = {"correctness_explanation", "correctness", "specificity"}


def _assert_rubric_outputs(judge_outputs):
    assert len(judge_outputs) == len(JUDGE_DATASET)

    for output in judge_outputs:
        # Every criterion (and its explanation) must come back.
        assert EXPECTED_KEYS <= set(output.parsed_output.keys()), output.raw_output
        assert output.field_values["correctness_explanation"]
        assert isinstance(output.field_values["correctness"], bool)
        assert output.field_values["specificity"] in ("specific", "vague")
        assert output.aggregate_score is not None
        assert 0.0 <= output.aggregate_score <= 1.0

    assert judge_outputs[0].field_values["correctness"] is True
    assert judge_outputs[1].field_values["correctness"] is False
    # Correctness is weighted 2:1, so a true+specific statement must outscore a
    # false one.
    assert judge_outputs[0].aggregate_score > judge_outputs[1].aggregate_score


@skip_if_no_openai_key
def test_rubric_judge_json_guided_decoding():
    """JSON responses are schema-constrained, so every criterion must be present."""
    judge = RubricJudge(judge_config=JudgeConfig.from_str(YAML_RUBRIC_CONFIG_JSON))
    _assert_rubric_outputs(judge.judge(inputs=JUDGE_DATASET))


@skip_if_no_openai_key
def test_rubric_judge_xml():
    judge = RubricJudge(judge_config=JudgeConfig.from_str(YAML_RUBRIC_CONFIG_XML))
    _assert_rubric_outputs(judge.judge(inputs=JUDGE_DATASET))
