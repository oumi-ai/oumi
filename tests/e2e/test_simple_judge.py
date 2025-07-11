import os
import unittest

import pytest

from oumi.core.configs.judge_config_v2 import JudgeConfig
from oumi.judges_v2.simple_judge import SimpleJudge

skip_if_no_openai_key = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY not set"
)

YAML_JUDGE_CONFIG_XML_ENUM = """
judge_params:
    prompt_template: Is the following statement correct? {statement}
    response_format: XML
    judgment_type: ENUM
    judgment_scores:
        "Correct": 0.99
        "Unsure": 0.5
        "Incorrect": 0.01
    include_explanation: True
inference_config:
    model:
        model_name: "gpt-4.1"
    engine: OPENAI
    generation:
        max_new_tokens: 8192
        temperature: 0.0
"""

YAML_JUDGE_CONFIG_JSON_BOOL = """
judge_params:
    prompt_template: "{prompt_question} {statement}"
    template_variables:
        prompt_question: "Is the following statement correct?"
    response_format: JSON
    judgment_type: BOOL
    include_explanation: False
inference_config:
    model:
        model_name: "gpt-4.1"
    engine: OPENAI
    generation:
        max_new_tokens: 8192
        temperature: 0.0
"""

JUDGE_DATASET = [
    {"statement": "The capital of France is Paris.", "useless_field": "Not used"},
    {"statement": "The Earth is flat.", "useless_field": "Not used"},
]


@skip_if_no_openai_key
def test_simple_judge_xml_enum():
    # Instantiate the judge using a YAML configuration.
    judge_config = JudgeConfig.from_str(YAML_JUDGE_CONFIG_XML_ENUM)
    simple_judge = SimpleJudge(judge_config=judge_config)

    # Call the judge with the dataset.
    judge_output = simple_judge.judge(inputs=JUDGE_DATASET)

    # Ensure the output is correct.
    print(judge_output)
    assert len(judge_output) == 2

    assert set(judge_output[0].parsed_output.keys()) == {"judgment", "explanation"}
    assert judge_output[0].parsed_output["judgment"] == "Correct"
    assert judge_output[0].parsed_output["explanation"] is not None
    assert judge_output[0].field_values["judgment"] == "Correct"
    assert judge_output[0].field_values["explanation"] is not None
    assert judge_output[0].field_scores["judgment"] == 0.99
    assert judge_output[0].field_scores["explanation"] is None

    assert set(judge_output[1].parsed_output.keys()) == {"judgment", "explanation"}
    assert judge_output[1].parsed_output["judgment"] == "Incorrect"
    assert judge_output[1].parsed_output["explanation"] is not None
    assert judge_output[1].field_values["judgment"] == "Incorrect"
    assert judge_output[1].field_values["explanation"] is not None
    assert judge_output[1].field_scores["judgment"] == 0.01
    assert judge_output[1].field_scores["explanation"] is None


@skip_if_no_openai_key
def test_simple_judge_json_bool():
    # Instantiate the judge using a YAML configuration.
    judge_config = JudgeConfig.from_str(YAML_JUDGE_CONFIG_JSON_BOOL)
    simple_judge = SimpleJudge(judge_config=judge_config)

    # Call the judge with the dataset.
    judge_output = simple_judge.judge(inputs=JUDGE_DATASET)

    # Ensure the output is correct.
    print(judge_output)
    assert len(judge_output) == 2

    assert set(judge_output[0].parsed_output.keys()) == {"judgment"}
    assert judge_output[0].parsed_output["judgment"] == "Yes"
    assert isinstance(judge_output[0].field_values["judgment"], bool)
    assert judge_output[0].field_values["judgment"]
    assert judge_output[0].field_scores["judgment"] == 1.0

    assert set(judge_output[1].parsed_output.keys()) == {"judgment"}
    assert judge_output[1].parsed_output["judgment"] == "No"
    assert isinstance(judge_output[1].field_values["judgment"], bool)
    assert not judge_output[1].field_values["judgment"]
    assert judge_output[1].field_scores["judgment"] == 0.0


if __name__ == "__main__":
    unittest.main()
