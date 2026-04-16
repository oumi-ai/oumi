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

from collections.abc import Callable
from dataclasses import dataclass

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
from oumi.core.configs.params.data_params import TrainTarget
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.utils.logging import logger

# This is used to set the max input length for a model with infinite size input
_VERY_LARGE_INTEGER = int(1e30)


# ---------------------------------------------------------------------------
# Template auto-detection for TrainTarget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CollatorTemplates:
    """Chat-format template strings used by the completions collator."""

    response_template: str
    end_of_turn_template: str


_CHATML_TEMPLATES = _CollatorTemplates(
    response_template="<|im_start|>assistant\n",
    end_of_turn_template="<|im_end|>",
)

_LLAMA3_TEMPLATES = _CollatorTemplates(
    response_template="<|start_header_id|>assistant<|end_header_id|>\n\n",
    end_of_turn_template="<|eot_id|>",
)


# Each entry is (detector_fn, templates).  The detector receives the
# tokenizer vocabulary (dict[str, int]) and returns True when the format
# matches.
_COLLATOR_TEMPLATE_DETECTORS: list[
    tuple[Callable[[dict[str, int]], bool], _CollatorTemplates]
] = [
    (lambda vocab: "<|im_start|>" in vocab, _CHATML_TEMPLATES),
    (lambda vocab: "<|start_header_id|>" in vocab, _LLAMA3_TEMPLATES),
]


def _resolve_collator_templates(
    tokenizer: "BaseTokenizer",
) -> _CollatorTemplates:
    """Auto-detect collator templates from the tokenizer vocabulary.

    Raises:
        ValueError: If no known chat format is detected.
    """
    vocab: dict[str, int] = tokenizer.get_vocab()
    for detector, templates in _COLLATOR_TEMPLATE_DETECTORS:
        if detector(vocab):
            return templates
    raise ValueError(
        "Cannot auto-detect chat template format from the tokenizer "
        "vocabulary. Please use `collator_kwargs` to specify templates "
        "manually instead of `train_target`."
    )


def _build_train_target_kwargs(
    train_target: TrainTarget,
    templates: _CollatorTemplates,
) -> dict:
    """Build collator keyword arguments for the given train target."""
    kwargs: dict = {
        "response_template": templates.response_template,
        "train_target": train_target.value,
    }
    if train_target == TrainTarget.ALL_ASSISTANT_TURNS:
        kwargs["end_of_turn_template"] = templates.end_of_turn_template
    # FINAL_ASSISTANT_TURN needs no extra kwargs beyond response_template
    return kwargs


def build_data_collator(
    collator_name: str,
    tokenizer: BaseTokenizer,
    *,
    max_length: int | None,
    label_ignore_index: int | None = constants.LABEL_IGNORE_INDEX,
    debug: bool = False,
    **kwargs,
) -> Callable:
    """Builds a data collator based on the given collator name.

    Args:
        collator_name: The name of the collator to build.
            Supported values are:

            - "text_with_padding": Uses `TextCollatorWithPadding`.
            - "text_completions_only_with_padding": Uses
                `TextCompletionsCollatorWithPadding`. Supports optional
                ``end_of_turn_template`` for tool-aware span-based masking.
            - "vision_language_with_padding": Uses `VisionLanguageCollatorWithPadding`.
            - "vision_language_sft": Uses `VisionLanguageSftCollator`.

        tokenizer: A tokenizer.
        max_length: An optional maximum sequence length.
        label_ignore_index: If set, then label values of tokens that shouldn't
            contribute to the loss computation will be replaced by this special value.
            For example, this can be `PAD`, or image tokens.
            PyTorch convention is to use -100 as the `ignore_index` label. Refer to
            the `ignore_index` parameter of `torch.nn.CrossEntropyLoss()`
            for more details.
        debug: If True, logs a single example for debugging purposes.
        **kwargs: Additional keyword arguments to pass to the collator constructor.

    Returns:
        Callable: The data collator function or class.

    Raises:
        ValueError: If an unsupported collator name is provided.
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
            debug=debug,
            **kwargs,
        )
    elif collator_name == "vision_language_with_padding":
        return VisionLanguageCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=max_length,
            truncation=enable_truncation,
            label_ignore_index=label_ignore_index,
            debug=debug,
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
        if not kwargs.get("response_template"):
            raise ValueError(
                "'text_completions_only_with_padding' requires a "
                "response_template. Either set train_target in your config "
                "(which auto-resolves templates from the tokenizer) or "
                "provide response_template via collator_kwargs."
            )

        return TextCompletionsCollatorWithPadding(
            tokenizer=tokenizer,
            debug=debug,
            ignore_index=(
                label_ignore_index if label_ignore_index is not None else -100
            ),
            **kwargs,
        )
    raise ValueError(f"Unknown data collator name: '{collator_name}'")


def build_collator_from_config(
    config: TrainingConfig, tokenizer: BaseTokenizer | None, debug: bool = False
) -> Callable | None:
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

    label_ignore_index: int | None = (
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

    # --- TrainTarget auto-resolution ---
    if train_split.train_target is not None:
        if collator_name != "text_completions_only_with_padding":
            raise ValueError(
                f"`train_target` is only supported with the "
                f"'text_completions_only_with_padding' collator, "
                f"got '{collator_name}'."
            )
        if tokenizer is None:
            raise ValueError("Tokenizer is required for `train_target` auto-detection.")
        templates = _resolve_collator_templates(tokenizer)
        train_target_kwargs = _build_train_target_kwargs(
            train_split.train_target, templates
        )
        collator_kwargs.update(train_target_kwargs)

    # Merge collator_kwargs from config with the existing kwargs
    # Config kwargs take precedence over automatically determined kwargs
    config_collator_kwargs = train_split.collator_kwargs or {}
    collator_kwargs.update(config_collator_kwargs)

    return build_data_collator(
        collator_name=collator_name,
        tokenizer=tokenizer,
        max_length=config.model.model_max_length,
        label_ignore_index=label_ignore_index,
        debug=debug,
        **collator_kwargs,
    )
