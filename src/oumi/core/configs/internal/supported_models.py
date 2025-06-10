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

"""Supported models configuration for Oumi framework.

This module defines the configuration for all non-standard models in the Oumi framework,
including both language models (LLMs) and vision-language models (VLMs). It provides
a centralized registry of model configurations that specify how different models
should be handled during training, inference, and evaluation.

Note that most models should work without any special configuration, and therefore do
not need to be added to this module.

Key Components:
    - InternalModelConfig: Configuration parameters for model behavior
    - Model-specific config creators: Functions that create configs for specific models
    - Registry functions: For looking up and accessing model configurations
    - _ModelTypeInfo: Metadata for each supported model type

How to Add a New Model:
    1. Create a configuration function:
       ```python
       def _create_my_model_config() -> InternalModelConfig:
           config = InternalModelConfig()
           # Configure the model's specific settings
           config.chat_template = "my_template"
           # Add any special features or parameters
           return config
       ```

    2. Add the model to get_all_models_map():
       ```python
       _ModelTypeInfo(
           model_type="my_model",  # Must match HF config.model_type
           model_class=transformers.AutoModelForCausalLM,  # Or appropriate class
           tested=False,  # Set to True once tests are added
           config=_create_my_model_config(),
       )
       ```

    3. For VLMs, configure visual features:
       ```python
       vlm_config = _create_default_vlm_config(
           supports_multiple_images=True,  # If model supports multiple images
           pixel_values_variable_shape=True,  # If images can have different sizes
       )
       # Add any model-specific image features
       vlm_config.model_input_features.update({...})
       ```
"""

import copy
import functools
import types
from collections.abc import Mapping
from typing import Any, NamedTuple, Optional

import transformers

from oumi.core.configs import ModelParams
from oumi.core.configs.internal.internal_model_config import (
    InternalFeatureFirstDimAction,
    InternalFeatureSpec,
    InternalModelConfig,
    InternalPaddingSide,
    InternalVisualModelConfig,
)
from oumi.core.registry import REGISTRY, RegistryType
from oumi.utils.logging import logger


@functools.cache
def find_model_hf_config(
    model_name: str,
    *,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    **kwargs: dict[str, Any],
):
    """Finds HF model config by model name."""
    hf_config, unused_kwargs = transformers.AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        return_unused_kwargs=True,
        revision=revision,
        **kwargs,
    )
    if unused_kwargs:
        logger.warning(
            f"Unused kwargs found in '{model_name}' config: {unused_kwargs}."
        )
    return hf_config


class _ModelTypeInfo(NamedTuple):
    """Metadata for a supported model type.

    This class encapsulates all the information needed to support a specific model
    type in the Oumi framework. Each supported model must have an entry with this
    metadata in the model registry.

    Attributes:
        model_type: The model type identifier that matches the HuggingFace model's
            config.model_type field. This is used to automatically detect and configure
            models based on their type. Examples: "llama", "gpt2", "qwen2_vl", "llava".

        model_class: The HuggingFace transformers class used to load this model type.
            Common classes include:
            - transformers.AutoModelForCausalLM: For standard language models
            - transformers.AutoModelForVision2Seq: For vision-to-text models
            - transformers.AutoModelForImageTextToText: For image+text to text models

        config: The InternalModelConfig instance that defines how this model should
            be configured. This includes settings like:
            - Chat template to use for formatting conversations
            - Input features the model expects (input_ids, pixel_values, etc.)
            - Special tokenizer settings
            - Visual model configuration for VLMs

        tested: Whether this model configuration has been tested and verified to work
            correctly with the framework. Set to True only after adding comprehensive
            tests in test_supported_models.py.
    """

    model_type: str
    model_class: type
    config: InternalModelConfig
    tested: bool = False


def _create_default_vlm_config(
    *,
    supports_multiple_images: bool = False,
    pixel_values_variable_shape: bool = False,
    pixel_values_first_dim_action: InternalFeatureFirstDimAction = (
        InternalFeatureFirstDimAction.DROP_IF_DUMMY
    ),
) -> InternalModelConfig:
    """Creates a default configuration for vision-language models.

    This function provides a base configuration that can be used for most VLMs.
    It sets up the basic visual features and configurations that VLMs typically need.

    Args:
        supports_multiple_images: Whether the model can process multiple images in a
            single prompt. Models like MLLaMA support this, while others like early
            LLAVA versions only support single images.

        pixel_values_variable_shape: Whether the model can handle images of different
            sizes within the same batch. When True, special handling is needed during
            collation to group same-sized images or use batch_size=1.

        pixel_values_first_dim_action: How to handle the first dimension of pixel_values
            tensor. Options include:
            - DROP_IF_DUMMY: Drop first dim if it's size 1 (common for single images)
            - DROP_ALWAYS: Always drop the first dimension
            - KEEP: Keep all dimensions as-is

    Returns:
        InternalModelConfig with basic VLM setup including:
        - "llava" chat template as default
        - pixel_values feature configuration
        - Visual model configuration

    Example:
        >>> config = _create_default_vlm_config(supports_multiple_images=True)
        >>> config.visual_config.supports_multiple_images
        True
    """
    config = InternalModelConfig()
    config.chat_template = "llava"
    config.model_input_features.update(
        {
            "pixel_values": InternalFeatureSpec(
                name="pixel_values",
                required=True,
                variable_shape=pixel_values_variable_shape,
                first_dim_action=pixel_values_first_dim_action,
                image_dependent=True,
            )
        }
    )
    visual_config = InternalVisualModelConfig()
    visual_config.supports_multiple_images = supports_multiple_images
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
    config = _create_default_vlm_config(supports_multiple_images=True)
    config.chat_template = "llama3-instruct"
    config.model_input_features.update(
        {
            feature_name: InternalFeatureSpec(
                name=feature_name,
                required=True,
                variable_shape=False,
                image_dependent=True,
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
    config = _create_default_vlm_config(
        pixel_values_variable_shape=True,
        # FIXME OPE-355 Set to True once multi-image issues are resolved for the model.
        supports_multiple_images=False,
    )
    config.chat_template = "qwen2-vl-instruct"
    # FIXME OPE-946 Consider updating to "right":
    # config.padding_side = InternalPaddingSide.PAD_RIGHT
    config.model_input_features.update(
        {
            feature_name: InternalFeatureSpec(
                name=feature_name,
                required=True,
                variable_shape=False,
                image_dependent=True,
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


def _create_qwen2_5_vl_vlm_config() -> InternalModelConfig:
    config = _create_qwen2_vl_vlm_config()
    # Update default parameters that differ from Qwen2:
    config.padding_side = InternalPaddingSide.PAD_RIGHT
    config.processor_kwargs.update(
        # Defaults per Qwen2.5-VL:
        # https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py # noqa: E501
        {
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 16384 * 28 * 28,
        }
    )
    return config


def _create_phi3_vlm_config() -> InternalModelConfig:
    config = _create_default_vlm_config(
        pixel_values_variable_shape=True,
        # FIXME OPE-355 Set to True once multi-image issues are resolved for the model.
        supports_multiple_images=False,
    )
    config.chat_template = "phi3-instruct"
    config.label_ignore_index = None
    config.sanitize_negative_labels = True
    config.model_input_features.update(
        {
            feature_name: InternalFeatureSpec(
                name=feature_name,
                required=True,
                variable_shape=False,
                image_dependent=True,
            )
            for feature_name in ("image_sizes",)
        }
    )
    return config


def _create_phi4_vlm_config() -> InternalModelConfig:
    config = InternalModelConfig()
    config.chat_template = "phi3-instruct"
    config.ignore_features = [
        "audio_attention_mask",  # We won't use audio features.
        "audio_embed_sizes",
        "input_audio_embeds",
    ]

    config.model_input_features.update(
        {
            feature_name: InternalFeatureSpec(
                name=feature_name,
                required=True,
                variable_shape=True,
                image_dependent=True,
                first_dim_action=InternalFeatureFirstDimAction.DROP_IF_DUMMY,
            )
            for feature_name in (
                "input_image_embeds",
                "image_attention_mask",
            )
        }
    )
    config.model_input_features.update(
        {
            feature_name: InternalFeatureSpec(
                name=feature_name,
                required=True,
                variable_shape=False,
                image_dependent=True,
            )
            for feature_name in ("image_sizes",)
        }
    )
    visual_config = InternalVisualModelConfig()
    # FIXME OPE-355 Set to True once multi-image issues are resolved for the model.
    visual_config.supports_multiple_images = False
    visual_config.variable_shape_image_features = True
    visual_config.main_image_feature = "input_image_embeds"

    config.visual_config = visual_config
    return config


def _create_internvl_config() -> InternalModelConfig:
    config = _create_default_vlm_config(
        pixel_values_variable_shape=True,
        # FIXME OPE-355 Set to True once multi-image issues are resolved for the model.
        supports_multiple_images=False,
    )
    config.chat_template = "internvl3"

    # Add to processor to return key-values pairs (e.g., "pixel_values": torch.Tensor):
    config.processor_kwargs.update({"return_dict": True})
    assert (
        config.model_input_features["pixel_values"].first_dim_action
        == InternalFeatureFirstDimAction.DROP_IF_DUMMY
    )
    return config


def _create_idefics3_vlm_config() -> InternalModelConfig:
    config = _create_default_vlm_config(
        supports_multiple_images=True, pixel_values_variable_shape=True
    )
    # FIXME OPE-697 Create model-specific chat template
    config.chat_template = "llava"
    config.model_input_features.update(
        {
            feature_name: InternalFeatureSpec(
                name=feature_name,
                required=True,
                variable_shape=False,
                image_dependent=True,
            )
            for feature_name in ("pixel_attention_mask",)
        }
    )
    return config


def _create_molmo_vlm_config() -> InternalModelConfig:
    """Creates a config for Molmo VLM model.

    Molmo uses a specific set of features including image masks and input indices
    for handling images in the model. The config is set up to handle these
    features appropriately.
    """
    config = InternalModelConfig()

    config.model_input_features.update(
        {
            feature_name: InternalFeatureSpec(
                name=feature_name,
                required=True,
                variable_shape=True,
                first_dim_action=InternalFeatureFirstDimAction.KEEP,
                image_dependent=True
                if feature_name in ("images", "image_masks", "image_input_idx")
                else False,
            )
            for feature_name in (
                "attention_mask",
                "input_ids",
                "labels",
                "images",
                "image_masks",
                "image_input_idx",
            )
        }
    )
    config.chat_template = "molmo"

    visual_config = InternalVisualModelConfig()
    visual_config.supports_multiple_images = False
    visual_config.variable_shape_image_features = True
    visual_config.main_image_feature = "images"

    config.visual_config = visual_config

    return config


@functools.cache
def get_all_models_map() -> Mapping[
    str,  # model type
    _ModelTypeInfo,
]:
    """Creates a map of all supported models with their configurations.

    This is the central registry of the non-standard models supported by the Oumi
    framework. Each entry maps a model type (as defined in the HuggingFace model config)
    to its corresponding configuration and metadata.

    Returns:
        An immutable mapping from model_type strings to _ModelTypeInfo objects.
        The mapping includes both LLMs and VLMs with their specific configurations.
    """
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
            model_type="qwen2_5_vl",
            model_class=default_vlm_class,
            tested=True,
            config=_create_qwen2_5_vl_vlm_config(),
        ),
        _ModelTypeInfo(
            model_type="vipllava",
            model_class=default_vlm_class,
            config=copy.deepcopy(default_vlm_config),
        ),
        _ModelTypeInfo(
            model_type="molmo",
            model_class=transformers.AutoModelForCausalLM,
            config=_create_molmo_vlm_config(),
        ),
        _ModelTypeInfo(
            model_type="phi3_v",
            model_class=transformers.AutoModelForCausalLM,
            tested=True,
            config=_create_phi3_vlm_config(),
        ),
        _ModelTypeInfo(
            model_type="phi4mm",
            model_class=transformers.AutoModelForCausalLM,
            config=_create_phi4_vlm_config(),
        ),
        _ModelTypeInfo(
            model_type="internvl",
            model_class=transformers.AutoModelForImageTextToText,
            config=_create_internvl_config(),
        ),
    ]

    # Make it immutable.
    return types.MappingProxyType({x.model_type: x for x in all_models_list})


def is_custom_model(model_name: str) -> bool:
    """Determines whether the model is a custom model defined in oumi registry."""
    result: bool = len(model_name) > 0 and REGISTRY.contains(
        name=model_name, type=RegistryType.MODEL
    )
    return result


def find_internal_model_config_using_model_name(
    model_name: str, trust_remote_code: bool
) -> Optional[InternalModelConfig]:
    """Finds an internal model config for supported models using model name.

    Args:
        model_name: The model name, either:
            - A HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf")
            - A local path to a model directory
            - A custom model name registered in Oumi
        trust_remote_code: Whether to trust external code associated with the model.
            Required for some models like Qwen2-VL that use custom code.

    Returns:
        InternalModelConfig for the model if it's supported, or None if:
        - The model is a custom Oumi model (handled separately)
        - The model type is not in the supported models registry
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
