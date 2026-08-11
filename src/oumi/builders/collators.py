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
    find_model_type_using_model_name,
)
from oumi.core.configs.params.data_params import TrainTarget
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.utils.logging import logger

_VERY_LARGE_INTEGER = int(1e30)

# Placeholder message contents for the probe conversation that template detection
# renders. Deliberately unlike anything a chat template emits, so they can be located
# by string search in the rendered output to find the turn boundaries around them.
_SENTINEL_SYS = "<<__S__>>"
_SENTINEL_USER = "<<__U__>>"
_SENTINEL_ASST = "<<__A__>>"
_FIX_HINT = (
    "Fix: provide response_template (and end_of_turn_template for "
    "all_assistant_turns) in collator_kwargs."
)

# The `train_split.train_target` options that mask by span. Only these unmask a stretch
# of the sequence wholesale and so can expose a tool result nested in an assistant turn
_SPAN_TRAIN_TARGETS = ("all_assistant_turns", "final_assistant_turn")

# The collator_kwargs keys holding the tool-result bracket, as (opener, closer).
_TOOL_TEMPLATE_KEYS = ("tool_response_template", "end_of_tool_response_template")

# Family labels shared by the two tables below, which together map an HF model_type
# to a bracket.
_GEMMA_4 = "gemma-4"
_GLM_4_MOE = "glm-4.5/4.6"

# The bracket each family's chat template wraps a tool result in.
_TOOL_RESPONSE_BRACKETS: dict[str, tuple[str, str]] = {
    _GEMMA_4: ("<|tool_response>", "<tool_response|>"),
    _GLM_4_MOE: ("<tool_response>", "</tool_response>"),
}

# HF ``config.model_type`` -> family, for architectures whose chat template renders
# tool results inside the assistant turn. Keying on model_type rather than repo name
# to also cover finetunes, mirrors, quantizations and local checkpoints
#
# WARNING: "glm4v" is excluded even though GLM-4.6V-Flash nests, because GLM-4.1V
# reports the same model_type without nesting. GLM-4.6V-Flash therefore needs the
# bracket set in collator_kwargs.
_NESTING_MODEL_TYPES: dict[str, str] = {
    "gemma4": _GEMMA_4,  # E2B/E4B/26B-A4B/31B, plus their -it and QAT variants
    "gemma4_unified": _GEMMA_4,  # 12B
    "glm4_moe": _GLM_4_MOE,  # GLM-4.5, -Air, 4.6, and FP8 variants
    "glm4v_moe": _GLM_4_MOE,  # GLM-4.5V, GLM-4.6V
}


def _detect_eot_template(
    tokenizer: "BaseTokenizer",
    after_text: str,
    between_text: str,
) -> tuple[list[int], str]:
    r"""Detect end-of-turn token IDs and template string.

    Compares token-ID prefixes of the text after the last assistant turn
    (end-of-sequence) with the text between assistant turns (mid-conversation).
    Both start with the turn closer and diverge after it, so what they agree on
    is the marker. Comparing IDs rather than characters keeps the result on token
    boundaries, so it can still be matched against a tokenized example.

    Primary: longest common token-ID prefix.
    Fallback: first token of between_text (for models like GPT OSS
    that use different mid-conversation vs end-of-sequence tokens).

    Args:
        tokenizer: Tokenizer to tokenize texts.
        after_text: Rendered text following the final assistant turn, i.e. the
            end-of-sequence side of the comparison.
        between_text: Rendered text between an assistant turn and the next user
            turn, i.e. the mid-conversation side.

    Returns:
        (eot_ids, end_of_turn_template): the end-of-turn token IDs and the same
        markers decoded back to a string.

    Examples:
        Qwen2.5 closes every assistant turn the same way, so the two sides agree
        for the whole marker and the primary path returns it::

            after_text    '<|im_end|>\n'
            between_text  '<|im_end|>\n<|im_start|>user\n'
            after_ids     [151645, 198]
            between_ids   [151645, 198, 151644, 872, 198]
                           ^^^^^^^^^^^ agree, then diverge
            -> eot_ids [151645, 198], '<|im_end|>\n'

        GPT OSS ends the final turn with a different token than a mid-conversation
        turn, so the two disagree at position 0 and the fallback takes the first
        token of between_text -- the mid-conversation closer is the one span
        masking needs, and the end-of-sequence token is irrelevant to it::

            after_text    '<|return|>'
            between_text  '<|end|><|start|>user<|message|>'
            after_ids     [200002]
            between_ids   [200007, 200006, 1428, 200008]
                           ^ differs at position 0, so no common prefix
            -> eot_ids [200007], '<|end|>'

        Olmo-3 lands on the same fallback, ending the conversation with
        '<|endoftext|>' but its turns with '<|im_end|>'. Note the fallback claims
        exactly one token: with no agreement to measure against there is nothing
        to say how far the marker extends, so it does not guess. That is why
        Olmo-3 detects '<|im_end|>' while Qwen2.5 keeps its trailing newline.
    """
    after_ids = tokenizer.encode(after_text, add_special_tokens=False)
    between_ids = tokenizer.encode(between_text, add_special_tokens=False)

    prefix_len = 0
    for a, b in zip(after_ids, between_ids):
        if a != b:
            break
        prefix_len += 1
    eot_ids = after_ids[:prefix_len]

    if not eot_ids and between_ids:
        eot_ids = between_ids[:1]

    eot_decoded = tokenizer.decode(eot_ids, skip_special_tokens=False)
    assert isinstance(eot_decoded, str)
    return eot_ids, eot_decoded


def _detect_response_template(
    tokenizer: "BaseTokenizer",
    header_text: str,
    eot_ids: list[int],
) -> str:
    r"""Detect the assistant response header from the user-to-assistant boundary.

    Strips the leading end-of-turn prefix (which belongs to the previous
    turn, not the response header) and any ``<think>`` blocks injected
    by reasoning-model chat templates (e.g. Qwen3).

    Args:
        tokenizer: Tokenizer whose chat template produced `header_text`.
        eot_ids: End-of-turn token IDs from ``_detect_eot_template``.
        header_text: Rendered text between the end of a user turn and the start of
            the assistant's content.

    Returns:
        The response_template string.

    Raises:
        ValueError: The header is nothing but a ``<think>`` block, leaving no
            usable response_template.

    Examples:
        Qwen2.5. The slice starts with the user turn's closer, which is dropped
        because it belongs to that turn, and the trailing newline goes so the
        marker survives BPE merging with whatever content follows::

            header_text  '<|im_end|>\n<|im_start|>assistant\n'
            header_ids   [151645, 198, 151644, 77091, 198]
            eot_ids      [151645, 198]        -> drop this prefix
            remaining    '<|im_start|>assistant\n'
            -> '<|im_start|>assistant'

        Qwen3 renders an empty reasoning block into the header. Everything from
        ``<think>`` on is model output, not a boundary marker, so it is cut::

            header_text  '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
            remaining    '<|im_start|>assistant\n<think>\n\n</think>\n\n'
            -> '<|im_start|>assistant'

        A header that is *only* a reasoning block leaves nothing to match on, so
        that case raises rather than returning a marker that would silently fail.
    """
    resp_ids = tokenizer.encode(header_text, add_special_tokens=False)
    eot_len = len(eot_ids)
    if eot_len > 0 and resp_ids[:eot_len] == eot_ids:
        resp_ids = resp_ids[eot_len:]

    resp_decoded = tokenizer.decode(resp_ids, skip_special_tokens=False)
    assert isinstance(resp_decoded, str)
    response_template = resp_decoded

    if "<think>" in response_template:
        idx = response_template.index("<think>")
        stripped = response_template[:idx].rstrip()
        if stripped:
            logger.info(
                "Stripped <think> block from auto-detected response_template: %r -> %r",
                response_template,
                stripped,
            )
            response_template = stripped
        else:
            raise ValueError(
                f"Extracted response_template is only a <think> block.\n{_FIX_HINT}"
            )

    response_template = response_template.rstrip("\n")
    return response_template


def resolve_collator_templates(
    tokenizer: "BaseTokenizer",
) -> tuple[str, str]:
    """Auto-detect response_template and end_of_turn_template.

    Applies the chat template to a known test conversation, then finds
    the assistant boundary strings in the rendered output.

    Args:
        tokenizer: Tokenizer to probe. Its chat template is rendered against a
            sentinel conversation; the tokenizer is not modified.

    Returns:
        (response_template, end_of_turn_template)

    Raises:
        ValueError: If templates cannot be extracted — the tokenizer has no chat
            template, the render did not return a string, the assistant turn
            boundaries were not locatable in it, or either extracted template came
            out empty.
    """
    msgs_with_sys = [
        {"role": "system", "content": _SENTINEL_SYS},
        {"role": "user", "content": _SENTINEL_USER},
        {"role": "assistant", "content": _SENTINEL_ASST},
        {"role": "user", "content": _SENTINEL_USER},
        {"role": "assistant", "content": _SENTINEL_ASST},
    ]
    msgs_no_sys = msgs_with_sys[1:]

    rendered = None
    for msgs in (msgs_with_sys, msgs_no_sys):
        try:
            rendered = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            break
        except Exception:
            continue
    if rendered is None:
        raise ValueError(
            f"Tokenizer has no chat template or it failed to render.\n{_FIX_HINT}"
        )

    if not isinstance(rendered, str):
        raise ValueError(
            f"Chat template returned a non-string type ({type(rendered).__name__}).\n"
            f"{_FIX_HINT}"
        )

    # Locate boundaries around the second turn pair
    # to avoid system-prompt effects on the first turn.
    try:
        first_asst = rendered.index(_SENTINEL_ASST)
        first_asst_end = first_asst + len(_SENTINEL_ASST)
        second_user = rendered.index(_SENTINEL_USER, first_asst_end)
        second_user_end = second_user + len(_SENTINEL_USER)
        second_asst = rendered.index(_SENTINEL_ASST, second_user_end)
        second_asst_end = second_asst + len(_SENTINEL_ASST)
    except ValueError:
        raise ValueError(
            "Could not locate assistant turn boundaries in the rendered "
            f"chat template.\n{_FIX_HINT}"
        )

    eot_ids, end_of_turn_template = _detect_eot_template(
        tokenizer,
        after_text=rendered[second_asst_end:],
        between_text=rendered[first_asst_end:second_user],
    )
    response_template = _detect_response_template(
        tokenizer,
        header_text=rendered[second_user_end:second_asst],
        eot_ids=eot_ids,
    )

    if not response_template.strip():
        raise ValueError(f"Extracted response_template is empty.\n{_FIX_HINT}")
    if not end_of_turn_template.strip():
        raise ValueError(f"Extracted end_of_turn_template is empty.\n{_FIX_HINT}")

    return response_template, end_of_turn_template


def _known_tool_response_markers(model_type: str | None) -> tuple[str, str] | None:
    """The tool-result bracket for an architecture known to nest tool results.

    Args:
        model_type: HF ``config.model_type``, or None when it could not be read.

    Returns:
        (opener, closer) when the architecture is listed in ``_NESTING_MODEL_TYPES``,
        else None.
    """
    family = _NESTING_MODEL_TYPES.get(model_type or "")
    return _TOOL_RESPONSE_BRACKETS[family] if family else None


def _resolve_tool_bracket(
    collator_kwargs: dict,
    config_collator_kwargs: dict,
    model_type: str | None,
) -> tuple[str, str] | None:
    """The bracket around tool results this model nests inside the assistant turn.

    Span masking unmasks everything between ``response_template`` and the next
    ``end_of_turn_template``. Some chat templates render tool results inside that
    span, which trains the model on environment output. The returned bracket lets the
    collator mask it again.

    Args:
        collator_kwargs: Auto-resolved kwargs built so far; read for ``train_target``.
        config_collator_kwargs: User-supplied ``collator_kwargs`` from the data config.
            A bracket set here is an override and wins over the known-model lookup.
        model_type: HF ``config.model_type`` for the model being trained, or None when
            it could not be determined.

    Returns:
        (tool_response_opener, tool_response_closer) strings, or None when the bracket
        is already configured, the train_target isn't span-based, or the architecture
        isn't a known nester.

    Raises:
        ValueError: Only one of the two markers was supplied.
    """
    # 1. Check if the user has supplied a tool response prefix. User specification wins
    user_defined_prefix = [
        key for key in _TOOL_TEMPLATE_KEYS if config_collator_kwargs.get(key)
    ]
    if user_defined_prefix:
        if len(user_defined_prefix) == 1:
            # One marker alone masks nothing, so a half-configured pair is always a
            # mistake rather than a partial opt-in.
            missing = next(
                key for key in _TOOL_TEMPLATE_KEYS if key not in user_defined_prefix
            )
            raise ValueError(
                f"collator_kwargs sets '{user_defined_prefix[0]}' without "
                f"'{missing}'. Tool results are only excluded from the training "
                "loss when both markers are set.\n"
                f"Fix: also set '{missing}' in collator_kwargs."
            )
        else:
            return None  # Hand-supplied bracket wins; nothing to look up.

    # 1b. Skip if on the legacy path
    if collator_kwargs.get("train_target") not in _SPAN_TRAIN_TARGETS:
        return None  # Only span-based masking can unmask a tool result.

    # 2. Check if the model being trained is in the known list of models that nest tool
    # response in assistant span
    tool_response_markers = _known_tool_response_markers(model_type)
    if tool_response_markers:
        logger.info(
            "Architecture %r renders tool results inside the assistant turn; masking "
            "them via bracket %r.",
            model_type,
            tool_response_markers,
        )
        return tool_response_markers

    # 3. Autodetection of tool response prefix from chat template fallback
    # Not implemented yet, current active mechanisms are active override or
    # using a known nested model
    # TODO(OPE-2185): fall back to probing the chat template.
    logger.warning(
        "No tool-response bracket specified for architecture %r; if your model chat "
        "template nests tool results in assistant span, it will not be remasked.",
        model_type,
    )
    return None


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
                "'text_completions_only_with_padding' requires a response_template.\n"
                "Fix: set train_target in your data config (auto-resolves templates "
                "from the tokenizer), or provide response_template in collator_kwargs."
            )
        if not kwargs.get("train_target"):
            raise ValueError(
                "'text_completions_only_with_padding' requires a train_target.\n"
                "Fix: set train_target in your data config, or provide "
                "train_target in collator_kwargs."
            )

        ignore_index = kwargs.pop(
            "ignore_index",
            label_ignore_index if label_ignore_index is not None else -100,
        )

        return TextCompletionsCollatorWithPadding(
            tokenizer=tokenizer,
            debug=debug,
            ignore_index=ignore_index,
            **kwargs,
        )
    raise ValueError(f"Unknown data collator name: '{collator_name}'")


def build_collator_from_config(
    config: TrainingConfig, tokenizer: BaseTokenizer | None, debug: bool = False
) -> Callable | None:
    """Creates data collator if specified in config.

    Resolves everything the collator needs from the config: the label ignore index,
    the visual-model kwargs, the train_target and its templates, and — for models
    whose chat template nests tool results inside the assistant turn — the tool-result
    bracket. Anything set in ``collator_kwargs`` overrides what is resolved here.

    Args:
        config: Training config. Its train split supplies ``collator_name``,
            ``train_target`` and ``collator_kwargs``; ``config.model`` supplies the
            model name used to look up the architecture and the max sequence length.
        tokenizer: Tokenizer the collator will use. Required whenever the split names
            a collator; templates are auto-detected from its chat template.
        debug: If True, the collator logs a single collated example.

    Returns:
        The collator, or None when the train split names no collator.

    Raises:
        ValueError: No tokenizer was given for a split that names a collator, a
            required processor name is missing, the completions collator was left
            unconfigured, or the tool-result bracket was only half supplied.
    """
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
        collator_kwargs["model_revision"] = config.model.model_revision

    # --- Resolve train_target and templates ---
    config_collator_kwargs = train_split.collator_kwargs or {}

    if collator_name == "text_completions_only_with_padding":
        if train_split.train_target is not None:
            # Path 1: train_target is set, auto-detect templates from
            # the tokenizer's chat template. Falls back to user-provided
            # response_template in collator_kwargs if auto-detection fails.
            collator_kwargs["train_target"] = train_split.train_target.value

            try:
                response_template, end_of_turn_template = resolve_collator_templates(
                    tokenizer
                )
                collator_kwargs["response_template"] = response_template
                if train_split.train_target == TrainTarget.ALL_ASSISTANT_TURNS:
                    collator_kwargs["end_of_turn_template"] = end_of_turn_template
            except ValueError:
                if config_collator_kwargs.get("response_template") is None:
                    raise

            if (
                train_split.train_target == TrainTarget.ALL_ASSISTANT_TURNS
                and "end_of_turn_template" not in collator_kwargs
                and config_collator_kwargs.get("end_of_turn_template") is None
            ):
                raise ValueError(
                    "train_target='all_assistant_turns' requires end_of_turn_template, "
                    "but auto-detection failed.\n"
                    "Fix: provide end_of_turn_template in collator_kwargs."
                )

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
                    "Instruction-based masking is deprecated.\n"
                    "Use train_target='all_assistant_turns'"
                    "or train_target='final_assistant_turn' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                collator_kwargs["train_target"] = "_legacy_instruction_response"
            else:
                collator_kwargs["train_target"] = "final_assistant_turn"
        else:
            raise ValueError(
                "'text_completions_only_with_padding' collator requires"
                " configuration.\n"
                "Fix: set train_target in your data config, "
                "or provide response_template in collator_kwargs."
            )

        tool_bracket = _resolve_tool_bracket(
            collator_kwargs,
            config_collator_kwargs,
            find_model_type_using_model_name(
                config.model.model_name,
                trust_remote_code=config.model.trust_remote_code,
                revision=config.model.model_revision,
            ),
        )
        if tool_bracket is not None:
            opener_key, closer_key = _TOOL_TEMPLATE_KEYS
            collator_kwargs[opener_key], collator_kwargs[closer_key] = tool_bracket

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
