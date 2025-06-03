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

from typing import Callable, Optional

import oumi.core.constants as constants
from oumi.core.collators.text_collator_with_padding import TextCollatorWithPadding
from oumi.core.collators.text_completions_collator_with_padding import (
    TextCompletionsCollatorWithPadding,
)
from oumi.core.collators.vision_language_collator_with_padding import (
    VisionLanguageCollatorWithPadding,
)
from oumi.core.collators.vision_language_sft_collator import VisionLanguageSftCollator
from oumi.core.configs import DatasetSplit, TrainingConfig
from oumi.core.configs.internal.supported_models import (
    find_internal_model_config,
)
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.utils.logging import logger

# This is used to set the max input length for a model with infinite size input
_VERY_LARGE_INTEGER = int(1e30)


def build_data_collator(
    collator_name: str,
    tokenizer: BaseTokenizer,
    *,
    max_length: Optional[int],
    label_ignore_index: Optional[int] = constants.LABEL_IGNORE_INDEX,
    **kwargs,
) -> Callable:
    """Builds a data collator based on the given collator name.

    This function creates the appropriate collator for different training scenarios.
    Choose the collator based on your data format and model type:

    Args:
        collator_name: The name of the collator to build.
            Supported values are:

            - "text_with_padding": For standard text-only language models.
                Use when training on text data without images.
                Input format: {"input_ids": [...], "attention_mask": [...], "labels": [...]}

            - "text_completions_only_with_padding": For instruction-following models
                where only completions (assistant responses) contribute to loss.
                Automatically masks instruction tokens in labels.

            - "vision_language_with_padding": For vision-language models with
                pre-processed features. Use when your dataset provides extracted
                image features (pixel_values, image_features, etc.).
                Input format: {...text features..., "pixel_values": tensor, ...}

            - "vision_language_sft": For vision-language models with conversation
                data. Use when your dataset provides Conversation objects with
                image references that need processing.
                Input format: {"conversation_json": serialized_conversation}

        tokenizer: A tokenizer for encoding text data.

        max_length: Maximum sequence length for padding/truncation. If None,
            sequences are padded to the longest in the batch.

        label_ignore_index: Value to replace padding tokens in labels for loss
            masking. Common values:
            - -100: PyTorch's default for CrossEntropyLoss
            - None: No masking (use original pad token IDs)

        **kwargs: Additional collator-specific arguments:
            For vision_language_with_padding:
                - allow_multi_image_inputs: bool
                - main_image_feature: str (default "images")
            For vision_language_sft:
                - processor_name: str (required)
                - processor_kwargs: dict
                - trust_remote_code: bool
                - process_individually: bool (default False)

    Returns:
        Callable: The instantiated data collator.

    Raises:
        ValueError: If an unsupported collator name is provided or required
            parameters are missing.

    Example:
        >>> # For text-only training
        >>> text_collator = build_data_collator(
        ...     "text_with_padding", tokenizer, max_length=512
        ... )
        >>>
        >>> # For vision-language with pre-processed features
        >>> vl_collator = build_data_collator(
        ...     "vision_language_with_padding", tokenizer, max_length=512,
        ...     main_image_feature="pixel_values"
        ... )
        >>>
        >>> # For vision-language SFT with conversations
        >>> sft_collator = build_data_collator(
        ...     "vision_language_sft", tokenizer, max_length=512,
        ...     processor_name="llava-hf/llava-1.5-7b-hf"
        ... )
    """
    if not collator_name:
        raise ValueError("Empty data collator name.")

    enable_truncation: bool = False
    if max_length is not None and max_length > 0:
        enable_truncation = True
        if (
            tokenizer.model_max_length is not None
            and tokenizer.model_max_length < _VERY_LARGE_INTEGER
            and max_length != tokenizer.model_max_length
        ):
            logger.warning(
                f"Data collator's maximum length: ({max_length}) is "
                + (
                    "greater than"
                    if max_length > tokenizer.model_max_length
                    else "less than"
                )
                + f" tokenizer's model maximum length ({tokenizer.model_max_length})"
            )

    if collator_name == "text_with_padding":
        return TextCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=max_length,
            truncation=enable_truncation,
            label_ignore_index=label_ignore_index,
            **kwargs,
        )
    elif collator_name == "vision_language_with_padding":
        return VisionLanguageCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=max_length,
            truncation=enable_truncation,
            label_ignore_index=label_ignore_index,
            **kwargs,
        )
    elif collator_name == "vision_language_sft":
        processor_name = kwargs.pop("processor_name", None)
        if not processor_name:
            raise ValueError(f"Empty processor_name for '{collator_name}'")
        processor_kwargs = kwargs.pop("processor_kwargs", None)
        return VisionLanguageSftCollator(
            tokenizer=tokenizer,
            processor_name=processor_name,
            processor_kwargs=processor_kwargs,
            max_length=max_length,
            truncation=enable_truncation,
            label_ignore_index=label_ignore_index,
            **kwargs,
        )
    elif collator_name == "text_completions_only_with_padding":
        return TextCompletionsCollatorWithPadding(
            tokenizer=tokenizer,
            instruction_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
            response_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    raise ValueError(f"Unknown data collator name: '{collator_name}'")


def build_collator_from_config(
    config: TrainingConfig, tokenizer: Optional[BaseTokenizer]
) -> Optional[Callable]:
    """Creates data collator if specified in config."""
    train_split = config.data.get_split(DatasetSplit.TRAIN)
    if not train_split.collator_name:
        return None
    collator_name: str = train_split.collator_name

    if tokenizer is None:
        raise ValueError(
            "Tokenizer must be provided if collator is specified! "
            f"collator: '{collator_name}'"
        )

    model_config = find_internal_model_config(config.model)

    label_ignore_index: Optional[int] = (
        config.training.label_ignore_index
        if config.training.label_ignore_index is not None
        else (
            model_config.label_ignore_index
            if model_config is not None
            else constants.LABEL_IGNORE_INDEX
        )
    )

    collator_kwargs = {}
    if (
        collator_name in ("vision_language_with_padding", "vision_language_sft")
        and model_config is not None
        and model_config.visual_config is not None
    ):
        collator_kwargs["allow_multi_image_inputs"] = (
            model_config.visual_config.supports_multiple_images
        )
        if collator_name == "vision_language_with_padding":
            collator_kwargs["main_image_feature"] = (
                model_config.visual_config.main_image_feature
            )

    if collator_name == "vision_language_sft":
        processor_name = collator_kwargs.get(
            "processor_name", config.model.tokenizer_name or config.model.model_name
        )
        if not processor_name:
            raise ValueError(f"Processor name must be provided for '{collator_name}'!")
        collator_kwargs["processor_name"] = processor_name
        collator_kwargs["processor_kwargs"] = config.model.processor_kwargs

        collator_kwargs["trust_remote_code"] = collator_kwargs.get(
            "trust_remote_code", config.model.trust_remote_code
        )

    # Merge collator_kwargs from config with the existing kwargs
    # Config kwargs take precedence over automatically determined kwargs
    config_collator_kwargs = train_split.collator_kwargs or {}
    collator_kwargs.update(config_collator_kwargs)

    return build_data_collator(
        collator_name=collator_name,
        tokenizer=tokenizer,
        max_length=config.model.model_max_length,
        label_ignore_index=label_ignore_index,
        **collator_kwargs,
    )
