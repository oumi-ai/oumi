from oumi.core.configs import (
    GenerationConfig,
    JudgeConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.configs.judge_config import JudgeAttribute
from oumi.core.registry import register_judge
from oumi.utils.io_utils import get_oumi_root_directory


@register_judge("oumi_v1_xml_claude_sonnet_judge")
def oumi_v1_xml_claude_sonnet_judge() -> JudgeConfig:
    """Returns a JudgeConfig for the Oumi v1 XML Anthropic judge.

    This function creates and returns a JudgeConfig object for the Oumi V1 Judge, which
    uses Claude Sonnet as a judge, with inputs and outpunts in XML format.

    Returns:
        JudgeConfig: A configuration object for the Oumi v1 XML Anthropic judge.

    Note:
        This judge uses the Anthropic API, so the ANTHROPIC_API_KEY environment
        variable must be set with a valid API key.
    """
    judges_directory = get_oumi_root_directory() / "judges" / "oumi_v1"

    attribute_names = ["helpful", "honest", "safe"]

    attributes = {
        attribute: JudgeAttribute.load(str(judges_directory / f"{attribute}.json"))
        for attribute in attribute_names
    }

    config = JudgeConfig(
        attributes=attributes,
        model=ModelParams(
            model_name="claude-3-5-sonnet-20240620",
        ),
        generation=GenerationConfig(
            max_new_tokens=1024,
            remote_params=RemoteParams(
                api_url="https://api.anthropic.com/v1/messages",
                api_key_env_varname="ANTHROPIC_API_KEY",
                max_retries=3,
            ),
        ),
    )
    return config


@register_judge("oumi_v1_xml_local_judge")
def oumi_v1_xml_local_judge() -> JudgeConfig:
    """Returns a JudgeConfig for the Oumi v1 XML local judge.

    Returns:
        JudgeConfig: A configuration object for the Oumi v1 XML local judge.

    Note:
        This judge uses a local GGUF model file for inference.
    """
    judges_directory = get_oumi_root_directory() / "judges" / "oumi_v1"

    attribute_names = ["helpful", "honest", "safe"]

    attributes = {
        attribute: JudgeAttribute.load(str(judges_directory / f"{attribute}.json"))
        for attribute in attribute_names
    }
    config = JudgeConfig(
        attributes=attributes,
        model=ModelParams(model_name="Qwen/Qwen2-0.5B-Instruct-GGUF"),
        generation=GenerationConfig(max_new_tokens=1024),
    )
    return config


@register_judge("oumi_v1_xml_gpt4o_judge")
def oumi_v1_xml_gpt4o_judge() -> JudgeConfig:
    """Returns a JudgeConfig for the Oumi v1 XML GPT-4 judge.

    This function creates and returns a JudgeConfig object for the Oumi V1 Judge, which
    uses GPT-4 as a judge, with inputs and outputs in XML format.

    Returns:
        JudgeConfig: A configuration object for the Oumi v1 XML GPT-4 judge.

    Note:
        This judge uses the OpenAI API, so the OPENAI_API_KEY environment
        variable must be set with a valid API key.
    """
    judges_directory = get_oumi_root_directory() / "judges" / "oumi_v1"

    attribute_names = ["helpful", "honest", "safe"]

    attributes = {
        attribute: JudgeAttribute.load(str(judges_directory / f"{attribute}.json"))
        for attribute in attribute_names
    }

    config = JudgeConfig(
        attributes=attributes,
        model=ModelParams(
            model_name="gpt-4o-2024-08-06",
        ),
        generation=GenerationConfig(
            max_new_tokens=1024,
            remote_params=RemoteParams(
                api_url="https://api.openai.com/v1/chat/completions",
                api_key_env_varname="OPENAI_API_KEY",
                max_retries=3,
            ),
        ),
    )
    return config
