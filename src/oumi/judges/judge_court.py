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

from typing import Union

from oumi.core.configs import (
    GenerationParams,
    JudgeConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.judge_config import JudgeAttribute
from oumi.core.registry import register_judge
from oumi.judges.oumi_judge import OumiJudgeInput, OumiJudgeOutput
from oumi.utils.io_utils import get_oumi_root_directory


@register_judge("oumi/v1_xml_claude_sonnet")
def oumi_v1_xml_claude_sonnet_judge() -> JudgeConfig:
    """Returns a JudgeConfig for the Oumi v1 XML Anthropic judge.

    This function creates and returns a JudgeConfig object for the Oumi V1 Judge, which
    uses Claude Sonnet as a judge, with inputs and outputs in XML format.

    Returns:
        JudgeConfig: A configuration object for the Oumi v1 XML Anthropic judge.

    Note:
        This judge uses the Anthropic API, so the ANTHROPIC_API_KEY environment
        variable must be set with a valid API key.
    """
    judges_directory = get_oumi_root_directory() / "judges" / "oumi_v1"

    attribute_names = ["helpful", "honest", "safe"]

    attributes = {
        attribute: JudgeAttribute[Union[OumiJudgeInput, OumiJudgeOutput]].load(
            str(judges_directory / f"{attribute}.json")
        )
        for attribute in attribute_names
    }

    config = JudgeConfig(
        attributes=attributes,
        model=ModelParams(
            model_name="claude-3-5-sonnet-20240620",
        ),
        generation=GenerationParams(
            max_new_tokens=1024,
        ),
        remote_params=RemoteParams(
            api_url="https://api.anthropic.com/v1/messages",
            api_key_env_varname="ANTHROPIC_API_KEY",
            max_retries=3,
        ),
        engine=InferenceEngineType.ANTHROPIC,
    )
    return config


@register_judge("oumi/v1_xml_local")
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
        attribute: JudgeAttribute[Union[OumiJudgeInput, OumiJudgeOutput]].load(
            str(judges_directory / f"{attribute}.json")
        )
        for attribute in attribute_names
    }
    config = JudgeConfig(
        attributes=attributes,
        model=ModelParams(model_name="Qwen/Qwen2-0.5B-Instruct-GGUF"),
        engine=InferenceEngineType.LLAMACPP,
        generation=GenerationParams(max_new_tokens=1024, temperature=0.0),
    )
    return config


@register_judge("oumi/v1_xml_gpt4o")
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
        attribute: JudgeAttribute[Union[OumiJudgeInput, OumiJudgeOutput]].load(
            str(judges_directory / f"{attribute}.json")
        )
        for attribute in attribute_names
    }

    config = JudgeConfig(
        attributes=attributes,
        model=ModelParams(model_name="gpt-4o-2024-08-06"),
        engine=InferenceEngineType.REMOTE,
        generation=GenerationParams(
            max_new_tokens=1024,
            temperature=0.0,
        ),
        remote_params=RemoteParams(
            api_url="https://api.openai.com/v1/chat/completions",
            api_key_env_varname="OPENAI_API_KEY",
            max_retries=3,
        ),
    )
    return config


@register_judge("oumi/v1_xml_deepseek_r1_hosted_by_deepseek")
def oumi_v1_xml_deepseek_r1_judge_hosted_by_deepseek() -> JudgeConfig:
    """Returns a JudgeConfig for the Oumi v1 XML DeepSeek R1 judge.

    This function creates and returns a JudgeConfig object for the Oumi V1 Judge, which
    uses DeepSeek R1 as a judge, with inputs and outputs in XML format.

    Returns:
        JudgeConfig: A configuration object for the Oumi v1 XML DeepSeek R1 judge.

    Note:
        This judge uses the DeepSeek API, so the DEEPSEEK_API_KEY environment
        variable must be set with a valid API key.
    """
    judges_directory = get_oumi_root_directory() / "judges" / "oumi_v1"

    attribute_names = ["helpful", "honest", "safe"]

    attributes = {
        attribute: JudgeAttribute[Union[OumiJudgeInput, OumiJudgeOutput]].load(
            str(judges_directory / f"{attribute}.json")
        )
        for attribute in attribute_names
    }

    config = JudgeConfig(
        attributes=attributes,
        # refer to model name https://api-docs.deepseek.com/quick_start/pricing
        model=ModelParams(model_name="deepseek-reasoner"),
        engine=InferenceEngineType.DEEPSEEK,
        generation=GenerationParams(
            max_new_tokens=1024,
            temperature=0.0,
        ),
        remote_params=RemoteParams(
            max_retries=3,
        ),
    )
    return config


@register_judge("oumi/v1_xml_deepseek_r1_hosted_by_sambanova")
def oumi_v1_xml_deepseek_r1_judge_hosted_by_sambanova() -> JudgeConfig:
    """Returns a JudgeConfig for the Oumi v1 XML DeepSeek R1 judge.

    This function creates and returns a JudgeConfig object for the Oumi V1 Judge, which
    uses DeepSeek R1 as a judge, with inputs and outputs in XML format.

    Returns:
        JudgeConfig: A configuration object for the Oumi v1 XML DeepSeek R1 judge.

    Note:
        This judge uses the SambaNova Cloud API, so the SAMBANOVA_API_KEY environment
        variable must be set with a valid API key.
    """
    judges_directory = get_oumi_root_directory() / "judges" / "oumi_v1"

    attribute_names = ["helpful", "honest", "safe"]

    attributes = {
        attribute: JudgeAttribute[Union[OumiJudgeInput, OumiJudgeOutput]].load(
            str(judges_directory / f"{attribute}.json")
        )
        for attribute in attribute_names
    }

    config = JudgeConfig(
        attributes=attributes,
        # refer to model name https://api-docs.deepseek.com/quick_start/pricing
        model=ModelParams(model_name="DeepSeek-R1"),
        engine=InferenceEngineType.SAMBANOVA,
        generation=GenerationParams(
            max_new_tokens=1024,
            temperature=0.0,
        ),
        remote_params=RemoteParams(
            max_retries=3, connection_timeout=10, politeness_policy=2.0
        ),
    )
    return config


@register_judge("oumi/v1_xml_deepseek_r1_hosted_by_together")
def oumi_v1_xml_deepseek_r1_judge_hosted_by_together() -> JudgeConfig:
    """Returns a JudgeConfig for the Oumi v1 XML DeepSeek R1 judge.

    This function creates and returns a JudgeConfig object for the Oumi V1 Judge, which
    uses DeepSeek R1 as a judge, with inputs and outputs in XML format.

    Returns:
        JudgeConfig: A configuration object for the Oumi v1 XML DeepSeek R1 judge.

    Note:
        This judge uses the Together Cloud API, so the TOGETHER_API_KEY environment
        variable must be set with a valid API key.
    """
    judges_directory = get_oumi_root_directory() / "judges" / "oumi_v1"

    attribute_names = ["helpful", "honest", "safe"]

    attributes = {
        attribute: JudgeAttribute[Union[OumiJudgeInput, OumiJudgeOutput]].load(
            str(judges_directory / f"{attribute}.json")
        )
        for attribute in attribute_names
    }

    config = JudgeConfig(
        attributes=attributes,
        model=ModelParams(model_name="deepseek-ai/DeepSeek-R1"),
        engine=InferenceEngineType.TOGETHER,
        generation=GenerationParams(
            max_new_tokens=1024,
            temperature=0.0,
        ),
        remote_params=RemoteParams(
            max_retries=3,
            # Experience number. Based on curl test, it takes about 48 seconds to
            # complete a LLM as judge task
            connection_timeout=60,
            # Note: Due to high demand, DeepSeek R1 has a model specific rate limit of \
            # 6 RPM for Free Users, and 12 RPM for Build Tier 1 users.
            # https://docs.together.ai/docs/rate-limits#rate-limit-tiers
            politeness_policy=11.0,
        ),
    )
    return config


@register_judge("oumi/v1_xml_unit_test")
def unit_test_judge():
    """Tiny judge for unit testing.

    Do not use this judge for anything serious as it returns random results.
    """
    attribute_path = (
        get_oumi_root_directory() / "judges" / "test_judge" / "helpful.json"
    )

    attribute = JudgeAttribute[Union[OumiJudgeInput, OumiJudgeOutput]].load(
        str(attribute_path)
    )

    config = JudgeConfig(
        attributes={"helpful": attribute},
        engine=InferenceEngineType.NATIVE,
        model=ModelParams(
            model_name="gpt2",
            tokenizer_pad_token="</s>",
            chat_template="gpt2",
        ),
        generation=GenerationParams(
            max_new_tokens=128,
            temperature=0.0,
        ),
    )

    return config
