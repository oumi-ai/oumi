import unittest

import pytest

from oumi.core.configs import JudgeConfigV2 as JudgeConfig
from oumi.judges_v2.oumi_judge import OumiJudge

YAML_CONFIG_XML_ENUM = """
    prompt_template: Is the following statement correct? {statement}
    response_format: XML
    judgment_type: ENUM
    judgment_scores:
        "Correct": 0.99
        "Unsure": 0.5
        "Incorrect": 0.01
    include_explanation: True

    model:
        model_name: "gpt-4.1-mini-2025-04-14"

    engine: OPENAI

    generation:
        max_new_tokens: 8192
        temperature: 0.0
"""

YAML_CONFIG_JSON_BOOL = """
    prompt_template: Is the following statement correct? {statement}
    response_format: JSON
    judgment_type: BOOL
    include_explanation: False

    model:
        model_name: "gpt-4.1-mini-2025-04-14"

    engine: OPENAI

    generation:
        max_new_tokens: 8192
        temperature: 0.0
"""

JUDGE_DATASET = [
    {"statement": "The capital of France is Paris.", "useless_field": "Not used"},
    {"statement": "The Earth is flat.", "useless_field": "Not used"},
]


@pytest.mark.skip(reason="No API key. Need to switch to a decent local model.")
def test_oumi_judge_xml_enum():
    # Instantiate the judge using a YAML configuration.
    config = JudgeConfig.from_str(YAML_CONFIG_XML_ENUM)
    oumi_judge = OumiJudge(config=config)

    # Call the judge with the dataset.
    judge_output = oumi_judge.judge(inputs=JUDGE_DATASET)

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


@pytest.mark.skip(reason="No API key. Need to switch to a decent local model.")
def test_oumi_judge_json_bool():
    # Instantiate the judge using a YAML configuration.
    config = JudgeConfig.from_str(YAML_CONFIG_JSON_BOOL)
    oumi_judge = OumiJudge(config=config)

    # Call the judge with the dataset.
    judge_output = oumi_judge.judge(inputs=JUDGE_DATASET)

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
