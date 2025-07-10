import pytest

from oumi.core.configs.params.judge_params import (
    JudgeOutputType,
    JudgeParams,
    JudgeResponseFormat,
)


def test_template_variables_applied_during_init():
    template_variables = {"role": "classifier", "domain": "medical"}

    prompt_template = "You are a {role} in the {domain} domain. Rate: {question}"
    system_instruction = "You are an expert specializing in the {domain} domain."
    expected_prompt = "You are a classifier in the medical domain. Rate: {question}"
    expected_system = "You are an expert specializing in the medical domain."

    params = JudgeParams(
        prompt_template=prompt_template,
        system_instruction=system_instruction,
        template_variables=template_variables,
        response_format=JudgeResponseFormat.XML,
        judgment_type=JudgeOutputType.BOOL,
    )

    assert params.prompt_template == expected_prompt
    assert params.system_instruction == expected_system


def test_no_template_variables_prompts_unchanged():
    prompt_template = "Rate: {question}"
    system_instruction = "You are a helpful assistant."

    params = JudgeParams(
        prompt_template=prompt_template,
        system_instruction=system_instruction,
        response_format=JudgeResponseFormat.XML,
        judgment_type=JudgeOutputType.BOOL,
    )

    assert params.prompt_template == prompt_template
    assert params.system_instruction == system_instruction


def test_template_variables_unused_variables():
    template_variables = {
        "role": "classifier",
        "domain": "medical",
        "unused_variable": "unused_value",
    }

    prompt_template = "You are a {role} in the {domain} domain. Rate: {question}"

    with pytest.raises(
        ValueError,
        match=(
            r"The following template variables are not used in the prompt_template "
            r"or system_instruction: \['unused_variable'\]"
        ),
    ):
        JudgeParams(
            prompt_template=prompt_template,
            template_variables=template_variables,
            response_format=JudgeResponseFormat.XML,
            judgment_type=JudgeOutputType.BOOL,
        )
