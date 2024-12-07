import pytest

from oumi.core.configs.internal.supported_models import (
    find_internal_model_config,
    find_internal_model_config_using_model_name,
    find_model_hf_config,
)
from oumi.core.configs.params.model_params import ModelParams


@pytest.mark.parametrize(
    "model_name, trust_remote_code",
    [
        ("llava-hf/llava-1.5-7b-hf", False),
        ("microsoft/Phi-3-vision-128k-instruct", True),
        ("Qwen/Qwen2-VL-2B-Instruct", True),
        ("Salesforce/blip2-opt-2.7b", False),
        # Access is restricted (gated repo):
        # ("meta-llama/Llama-3.2-11B-Vision-Instruct", False),
    ],
)
def test_common_vlm_models(model_name: str, trust_remote_code):
    debug_tag = f"model_name: {model_name} trust_remote_code:{trust_remote_code}"
    assert (
        find_model_hf_config(model_name, trust_remote_code=trust_remote_code)
        is not None
    ), debug_tag

    assert (
        find_internal_model_config_using_model_name(
            model_name, trust_remote_code=trust_remote_code
        )
        is not None
    ), debug_tag

    assert (
        find_internal_model_config(
            ModelParams(model_name=model_name, trust_remote_code=trust_remote_code)
        )
        is not None
    ), debug_tag
