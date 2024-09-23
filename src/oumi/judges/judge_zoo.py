from pathlib import Path

from oumi import OUMI_ROOT_DIRECTORY
from oumi.core.configs import (
    GenerationConfig,
    JudgeConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.configs.judge_config import JudgeAttribute


def _get_default_judge_config() -> JudgeConfig:
    judges_directory = Path(OUMI_ROOT_DIRECTORY) / "judges" / "oumi_v1"

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
                max_retries=0,
            ),
        ),
    )
    return config


def _get_ollama_judge_config() -> JudgeConfig:
    judges_directory = Path(OUMI_ROOT_DIRECTORY) / "judges" / "oumi_v1"

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
                api_url="http://localhost:1234/v1/chat/completions",
                max_retries=2,
            ),
        ),
    )
    return config


def _get_default_local_judge_config() -> JudgeConfig:
    judges_directory = Path(OUMI_ROOT_DIRECTORY) / "judges" / "oumi_v1"

    attribute_names = ["helpful", "honest", "safe", "valid"]
    attributes = {
        attribute: JudgeAttribute.load(str(judges_directory / f"{attribute}.json"))
        for attribute in attribute_names
    }
    config = JudgeConfig(
        attributes=attributes,
        model=ModelParams(
            model_name=str(judges_directory / "Q4_K_M-00001-of-00001.gguf"),
        ),
        generation=GenerationConfig(
            max_new_tokens=1024,
        ),
    )
    return config
