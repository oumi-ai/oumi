from oumi.core.configs.params.judge_params import (
    JudgeOutputType,
    JudgeParams,
    JudgeResponseFormat,
)


def test_template_variables_applied_during_init():
    template_variables = {"role": "classifier", "domain": "medical"}

    prompt_template = "You are a {role} in the {domain} domain. Rate: {question}"
    system_instruction = "You are a {domain} expert specializing in {domain} domain."
    expected_prompt = "You are a classifier in the medical domain. Rate: {question}"
    expected_system = "You are a medical expert specializing in medical domain."

    params = JudgeParams(
        prompt_template=prompt_template,
        system_instruction=system_instruction,
        template_variables=template_variables,
        response_format=JudgeResponseFormat.XML,
        judgment_type=JudgeOutputType.BOOL,
    )
    params.replace_template_variables()

    assert params.prompt_template == expected_prompt
    assert params.system_instruction == expected_system


def test_no_template_variables_prompts_unchanged():
    template_variables = {"unused": "unused"}

    prompt_template = "Rate: {question}"
    system_instruction = "You are a helpful assistant."

    params = JudgeParams(
        prompt_template=prompt_template,
        system_instruction=system_instruction,
        template_variables=template_variables,
        response_format=JudgeResponseFormat.XML,
        judgment_type=JudgeOutputType.BOOL,
    )
    params.replace_template_variables()

    assert params.prompt_template == prompt_template
    assert params.system_instruction == system_instruction
