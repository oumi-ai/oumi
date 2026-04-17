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

import warnings
from collections.abc import Callable

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

_VERY_LARGE_INTEGER = int(1e30)
_SENTINEL_USER = "<<__U__>>"
_SENTINEL_ASST = "<<__A__>>"
_FALLBACK_MSG = (
    "Cannot auto-detect collator templates from the chat template. "
    "Provide response_template (and end_of_turn_template for "
    "all_assistant_turns) via collator_kwargs."
)


def _resolve_collator_templates(
    tokenizer: "BaseTokenizer",
) -> tuple[str, str]:
    """Auto-detect response_template and end_of_turn_template.

    Applies the chat template to a known test conversation, then finds
    the assistant boundary strings in the rendered output.

    Returns:
        (response_template, end_of_turn_template)

    Raises:
        ValueError: If templates cannot be extracted.
    """
    msgs = [
        {"role": "user", "content": _SENTINEL_USER},
        {"role": "assistant", "content": _SENTINEL_ASST},
        {"role": "user", "content": _SENTINEL_USER},
        {"role": "assistant", "content": _SENTINEL_ASST},
    ]

    try:
        rendered = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    except Exception as exc:
        raise ValueError(_FALLBACK_MSG) from exc

    if not isinstance(rendered, str):
        raise ValueError(_FALLBACK_MSG)

    # Locate boundaries around the second turn pair
    # to avoid system-prompt effects on the first turn.
    try:
        a1 = rendered.index(_SENTINEL_ASST)
        first_asst_end = a1 + len(_SENTINEL_ASST)
        second_user = rendered.index(_SENTINEL_USER, first_asst_end)
        second_user_end = second_user + len(_SENTINEL_USER)
        second_asst = rendered.index(_SENTINEL_ASST, second_user_end)
        second_asst_end = second_asst + len(_SENTINEL_ASST)
    except ValueError:
        raise ValueError(_FALLBACK_MSG)

    # End-of-turn: common token-ID prefix of the two strings that
    # follow assistant content (mid-conversation vs. end-of-sequence).
    after_ids = tokenizer.encode(rendered[second_asst_end:], add_special_tokens=False)
    between_ids = tokenizer.encode(
        rendered[first_asst_end:second_user], add_special_tokens=False
    )
    eot_len = 0
    for a, b in zip(after_ids, between_ids):
        if a != b:
            break
        eot_len += 1
    eot_ids = after_ids[:eot_len]
    end_of_turn_template: str = tokenizer.decode(eot_ids, skip_special_tokens=False)

    # Response template: strip the EOT prefix to get just the assistant header.
    resp_ids = tokenizer.encode(
        rendered[second_user_end:second_asst], add_special_tokens=False
    )
    if eot_len > 0 and resp_ids[:eot_len] == eot_ids:
        resp_ids = resp_ids[eot_len:]
    response_template: str = tokenizer.decode(resp_ids, skip_special_tokens=False)

    if not response_template.strip():
        raise ValueError(_FALLBACK_MSG)

    # Qwen3 and similar reasoning models inject <think>...</think> into
    # every assistant turn via their chat template.  If training data was
    # formatted without thinking tokens the response_template won't match
    # and every example will be silently masked.
    if "<think>" in response_template:
        logger.warning(
            "The extracted response_template contains <think> tokens "
            "(from the model's chat template). If you're training without "
            "thinking tokens, use collator_kwargs to specify "
            "response_template manually."
        )

    return response_template, end_of_turn_template


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

    # --- Resolve train_target and templates ---
    config_collator_kwargs = train_split.collator_kwargs or {}

    if collator_name == "text_completions_only_with_padding":
        if train_split.train_target is not None:
            # Path 1: train_target is set, auto-detect templates from
            # the tokenizer's chat template. Falls back to user-provided
            # response_template in collator_kwargs if auto-detection fails.
            collator_kwargs["train_target"] = train_split.train_target.value

            try:
                response_template, end_of_turn_template = _resolve_collator_templates(
                    tokenizer
                )
                collator_kwargs["response_template"] = response_template
                if train_split.train_target == TrainTarget.ALL_ASSISTANT_TURNS:
                    collator_kwargs["end_of_turn_template"] = end_of_turn_template
            except ValueError:
                if config_collator_kwargs.get("response_template") is None:
                    raise

        elif config_collator_kwargs.get("response_template") is not None:
            # Path 2: train_target not set, templates provided manually
            # via collator_kwargs. Infer train_target from which templates
            # are present.
            has_eot = config_collator_kwargs.get("end_of_turn_template") is not None
            has_inst = config_collator_kwargs.get("instruction_template") is not None
            if has_eot:
                collator_kwargs["train_target"] = "all_assistant_turns"
            elif has_inst:
                warnings.warn(
                    "Instruction-based masking is deprecated. "
                    "Use train_target='all_assistant_turns' with "
                    "end_of_turn_template for multi-turn conversations, "
                    "or train_target='final_assistant_turn' "
                    "for single-turn completions.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                collator_kwargs["train_target"] = "_legacy_instruction_response"
            else:
                collator_kwargs["train_target"] = "final_assistant_turn"
        else:
            raise ValueError(
                "'text_completions_only_with_padding' requires either "
                "train_target or response_template in collator_kwargs."
            )

    # User-provided collator_kwargs override auto-resolved values
    collator_kwargs.update(config_collator_kwargs)

    return build_data_collator(
        collator_name=collator_name,
        tokenizer=tokenizer,
        max_length=config.model.model_max_length,
        label_ignore_index=label_ignore_index,
        debug=debug,
        **collator_kwargs,
    )
