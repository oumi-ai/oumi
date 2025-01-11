import copy
import functools
import types
from collections.abc import Mapping
from typing import NamedTuple, Optional

import transformers

from oumi.core.configs import ModelParams
from oumi.core.configs.internal.internal_model_config import (
    InternalFeatureFirstDimAction,
    InternalFeatureSpec,
    InternalModelConfig,
    InternalVisualModelConfig,
)
from oumi.core.registry import REGISTRY, RegistryType
from oumi.utils.logging import logger


@functools.cache
def find_model_hf_config(model_name: str, *, trust_remote_code: bool):
    """Finds HF model config by model name."""
    hf_config, unused_kwargs = transformers.AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        return_unused_kwargs=True,
    )
    if unused_kwargs:
        logger.warning(
            f"Unused kwargs found in '{model_name}' config: {unused_kwargs}."
        )
    return hf_config


class _ModelTypeInfo(NamedTuple):
    model_type: str
    model_class: type
    config: InternalModelConfig
    tested: bool = False


def _create_default_vlm_config(
    pixel_values_variable_shape: bool = False,
) -> InternalModelConfig:
    config = InternalModelConfig()
    config.chat_template = "llava"
    config.model_input_features.update(
        {
            "pixel_values": InternalFeatureSpec(
                name="pixel_values",
                required=True,
                variable_shape=pixel_values_variable_shape,
                first_dim_action=InternalFeatureFirstDimAction.DROP_IF_DUMMY,
            )
        }
    )
    visual_config = InternalVisualModelConfig()
    visual_config.variable_shape_image_features = pixel_values_variable_shape
    config.visual_config = visual_config
    return config


def _create_gpt2_config() -> InternalModelConfig:
    return InternalModelConfig(
        chat_template="gpt2", tokenizer_pad_token="<|endoftext|>"
    )


@functools.cache
def get_default_vlm_model_config() -> InternalModelConfig:
    """Returns default VLM model config."""
    return _create_default_vlm_config()


def _create_llava_vlm_config() -> InternalModelConfig:
    config = _create_default_vlm_config()
    config.chat_template = "llava"
    assert config.visual_config is not None
    config.processor_kwargs.update(
        {"patch_size": 14, "vision_feature_select_strategy": "default"}
    )
    return config


def _create_blip2_vlm_config() -> InternalModelConfig:
    config = _create_default_vlm_config()
    config.chat_template = "default"
    assert config.visual_config is not None
    config.processor_kwargs.update({"num_query_tokens": 32})
    return config


def _create_mllama_vlm_config() -> InternalModelConfig:
    config = _create_default_vlm_config()
    config.chat_template = "llama3-instruct"
    config.model_input_features.update(
        {
            feature_name: InternalFeatureSpec(
                name=feature_name,
                required=True,
                variable_shape=False,
            )
            for feature_name in (
                "aspect_ratio_ids",
                "aspect_ratio_mask",
                "cross_attention_mask",
            )
        }
    )
    return config


def _create_qwen2_vl_vlm_config() -> InternalModelConfig:
    config = _create_default_vlm_config(pixel_values_variable_shape=True)
    config.chat_template = "qwen2-vl-instruct"
    config.model_input_features.update(
        {
            feature_name: InternalFeatureSpec(
                name=feature_name,
                required=True,
                variable_shape=False,
            )
            for feature_name in ("image_grid_thw",)
        }
    )
    config.processor_kwargs.update(
        {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }
    )
    return config


def _create_phi3_vlm_config() -> InternalModelConfig:
    config = _create_default_vlm_config(pixel_values_variable_shape=True)
    config.chat_template = "phi3-instruct"
    config.label_ignore_index = None
    config.sanitize_negative_labels = True
    config.model_input_features.update(
        {
            feature_name: InternalFeatureSpec(
                name=feature_name,
                required=True,
                variable_shape=False,
            )
            for feature_name in ("image_sizes",)
        }
    )
    assert config.visual_config is not None
    visual_config = config.visual_config
    visual_config.supports_multiple_images = True
    return config


def _create_idefics3_vlm_config() -> InternalModelConfig:
    config = _create_default_vlm_config(pixel_values_variable_shape=False)
    # FIXME OPE-697 Create model-specific chat template
    config.chat_template = "llava"
    config.model_input_features.update(
        {
            feature_name: InternalFeatureSpec(
                name=feature_name,
                required=True,
                variable_shape=False,
            )
            for feature_name in ("pixel_attention_mask",)
        }
    )
    assert config.visual_config is not None
    visual_config = config.visual_config
    visual_config.supports_multiple_images = True
    visual_config.variable_shape_image_features = True
    return config


@functools.cache
def get_all_models_map() -> (
    Mapping[
        str,  # model type
        _ModelTypeInfo,
    ]
):
    """Creates a map of all supported VLMs with related configs."""
    default_vlm_config: InternalModelConfig = _create_default_vlm_config()

    default_llm_class = transformers.AutoModelForCausalLM
    default_vlm_class = transformers.AutoModelForVision2Seq

    all_models_list: list[_ModelTypeInfo] = [
        _ModelTypeInfo(
            model_type="gpt2",
            model_class=default_llm_class,
            tested=True,
            config=_create_gpt2_config(),
        ),
        _ModelTypeInfo(
            model_type="blip-2",
            model_class=default_vlm_class,
            tested=True,
            config=_create_blip2_vlm_config(),
        ),
        _ModelTypeInfo(
            model_type="blip",
            model_class=default_vlm_class,
            config=copy.deepcopy(default_vlm_config),
        ),
        _ModelTypeInfo(
            model_type="chameleon",
            model_class=default_vlm_class,
            config=copy.deepcopy(default_vlm_config),
        ),
        _ModelTypeInfo(
            model_type="idefics",
            model_class=default_vlm_class,
            config=copy.deepcopy(default_vlm_config),
        ),
        _ModelTypeInfo(
            model_type="idefics2",
            model_class=default_vlm_class,
            config=copy.deepcopy(default_vlm_config),
        ),
        _ModelTypeInfo(
            model_type="idefics3",
            model_class=default_vlm_class,
            config=_create_idefics3_vlm_config(),
        ),
        _ModelTypeInfo(
            model_type="instructblip",
            model_class=default_vlm_class,
            config=copy.deepcopy(default_vlm_config),
        ),
        _ModelTypeInfo(
            model_type="llava",
            model_class=default_vlm_class,
            tested=True,
            config=_create_llava_vlm_config(),
        ),
        _ModelTypeInfo(
            model_type="mllama",
            model_class=default_vlm_class,
            tested=True,
            config=_create_mllama_vlm_config(),
        ),
        _ModelTypeInfo(
            model_type="paligemma",
            model_class=default_vlm_class,
            config=copy.deepcopy(default_vlm_config),
        ),
        _ModelTypeInfo(
            model_type="qwen2_vl",
            model_class=default_vlm_class,
            tested=True,
            config=_create_qwen2_vl_vlm_config(),
        ),
        _ModelTypeInfo(
            model_type="vipllava",
            model_class=default_vlm_class,
            config=copy.deepcopy(default_vlm_config),
        ),
        _ModelTypeInfo(
            model_type="molmo",
            model_class=transformers.AutoModelForCausalLM,
            config=copy.deepcopy(default_vlm_config),
        ),
        _ModelTypeInfo(
            model_type="phi3_v",
            model_class=transformers.AutoModelForCausalLM,
            tested=True,
            config=_create_phi3_vlm_config(),
        ),
    ]

    # Make it immutable.
    return types.MappingProxyType({x.model_type: x for x in all_models_list})


def is_custom_model(model_name: str):
    """Determines whether the model is a custom model defined in oumi registry."""
    if len(model_name) > 0 and REGISTRY.contains(
        name=model_name, type=RegistryType.MODEL
    ):
        return True

    return False


def find_internal_model_config_using_model_name(
    model_name: str, trust_remote_code: bool
) -> Optional[InternalModelConfig]:
    """Finds an internal model config for supported models using model name.

    Args:
        model_name: The model name.
        trust_remote_code: Whether to trust external code associated with the model.

    Returns:
        Model config, or `None` if model is not recognized.
    """
    if is_custom_model(model_name):
        return None

    hf_config = find_model_hf_config(model_name, trust_remote_code=trust_remote_code)
    llm_info = get_all_models_map().get(hf_config.model_type, None)
    return llm_info.config if llm_info is not None else None


def find_internal_model_config(
    model_params: ModelParams,
) -> Optional[InternalModelConfig]:
    """Finds an internal model config for supported models using `ModelParams`.

    Args:
        model_params: The model parameters.

    Returns:
        Model config, or `None` if model is not recognized.
    """
    return find_internal_model_config_using_model_name(
        model_params.model_name, model_params.trust_remote_code
    )
