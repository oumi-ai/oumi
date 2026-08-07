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
from typing import Any, cast

from transformers import PreTrainedTokenizerFast

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
_SENTINEL_SYS = "<<__S__>>"
_SENTINEL_USER = "<<__U__>>"
_SENTINEL_ASST = "<<__A__>>"
_FIX_HINT = (
    "Fix: provide response_template (and end_of_turn_template for "
    "all_assistant_turns) in collator_kwargs."
)
_TOOL_FIX_HINT = (
    "Fix: provide tool_response_template and end_of_tool_response_template in "
    "collator_kwargs, or use a chat template that renders tool results in their "
    "own turn."
)
_PROBE_RESULT = "PROBE_TOOL_RESULT_"
# Mistral's template rejects any tool call id whose length isn't exactly 9 (its error
# message claims alphanumeric too, but only the length is checked).
_PROBE_CALL_ID = "probecall"
_PROBE_TOOLS: list[Any] = [
    {
        "type": "function",
        "function": {
            "name": "probe_fn",
            "description": "probe",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
        },
    }
]


def _detect_eot_template(
    tokenizer: "BaseTokenizer",
    after_text: str,
    between_text: str,
) -> tuple[list[int], str]:
    """Detect end-of-turn token IDs and template string.

    Compares token-ID prefixes of the text after the last assistant turn
    (end-of-sequence) with the text between assistant turns (mid-conversation).

    Primary: longest common token-ID prefix.
    Fallback: first token of between_text (for models like GPT OSS
    that use different mid-conversation vs end-of-sequence tokens).

    Returns:
        (eot_ids, end_of_turn_template)
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
    """Detect the assistant response header from the user-to-assistant boundary.

    Strips the leading end-of-turn prefix (which belongs to the previous
    turn, not the response header) and any ``<think>`` blocks injected
    by reasoning-model chat templates (e.g. Qwen3).

    Returns:
        response_template string
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

    Returns:
        (response_template, end_of_turn_template)

    Raises:
        ValueError: If templates cannot be extracted.
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


def _tool_probe_messages(num_results: int) -> list[dict]:
    """Conversation with one tool call answered by `num_results` tool messages.

    Only the result count varies, so diffing two renders isolates one rendered tool
    result.
    """
    messages: list[dict] = [
        {"role": "user", "content": "PROBE_USER"},
        {
            "role": "assistant",
            # Explicit empty content: Phi, DeepSeek and Granite templates read
            # message.content unconditionally and raise on a tool_calls-only message.
            "content": "",
            "tool_calls": [
                {
                    # Mistral requires 9-character alphanumeric tool call ids.
                    "id": _PROBE_CALL_ID,
                    "type": "function",
                    # A mapping, not a JSON string: gemma-4 and Qwen3.5 reject strings.
                    "function": {"name": "probe_fn", "arguments": {"a": "1"}},
                }
            ],
        },
    ]
    messages += [
        {
            "role": "tool",
            "tool_call_id": _PROBE_CALL_ID,
            "name": "probe_fn",
            "content": f"{_PROBE_RESULT}{i}",
        }
        for i in range(num_results)
    ]
    messages.append({"role": "assistant", "content": "PROBE_ANSWER"})
    return messages


def _diff_ids(shorter: list[int], longer: list[int]) -> list[int]:
    """Token ids `longer` gained over `shorter`, stripping common prefix and suffix."""
    prefix = 0
    while (
        prefix < len(shorter)
        and prefix < len(longer)
        and shorter[prefix] == longer[prefix]
    ):
        prefix += 1
    suffix = 0
    while (
        suffix < len(shorter) - prefix
        and suffix < len(longer) - prefix
        and shorter[len(shorter) - suffix - 1] == longer[len(longer) - suffix - 1]
    ):
        suffix += 1
    return longer[prefix : len(longer) - suffix]


def resolve_tool_response_template(
    tokenizer: "BaseTokenizer",
    response_template: str,
    end_of_turn_template: str,
) -> tuple[list[int], list[int]] | None:
    """Detects the bracket around tool results nested inside assistant spans.

    Span masking unmasks everything between ``response_template`` and the next
    ``end_of_turn_template``. Some chat templates (gemma-4) render tool results
    *inside* the assistant turn, which puts environment output inside that span.
    This finds the marker pair bracketing those results so masking can exclude them.

    Returns token ids rather than strings: ``_tokenize_template`` accepts either, and
    ids avoid a decode/re-encode round trip that isn't guaranteed to be lossless.

    Returns:
        (opener_ids, closer_ids) when tool results are nested and a specific bracket
        was recovered. None when they aren't nested (nothing to mask), or when the
        probe couldn't be rendered (nesting undetermined).

    Raises:
        ValueError: Tool results are nested inside assistant spans but no specific
            bracket was recovered, so they can't be excluded from the loss.
    """
    try:
        one = tokenizer.apply_chat_template(
            _tool_probe_messages(1),
            tools=_PROBE_TOOLS,
            tokenize=False,
            add_generation_prompt=False,
        )
        two = tokenizer.apply_chat_template(
            _tool_probe_messages(2),
            tools=_PROBE_TOOLS,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        # A template that rejects the probe tells us nothing about nesting.
        logger.debug(
            "Tool-response probe failed to render: %s: %s", type(e).__name__, e
        )
        return None

    if not isinstance(one, str) or not isinstance(two, str):
        return None

    first_result = f"{_PROBE_RESULT}0"
    # Template drops tool messages entirely; nothing to mask.
    if first_result not in one:
        return None

    # Nested when the closest preceding turn boundary is an assistant header.
    head = one[: one.index(first_result)]
    not_nested = head.rfind(response_template) <= head.rfind(end_of_turn_template)
    if not_nested:
        return None

    one_ids = tokenizer.encode(one, add_special_tokens=False)
    two_ids = tokenizer.encode(two, add_special_tokens=False)
    tool_response_ids = _diff_ids(one_ids, two_ids)
    if not tool_response_ids:
        raise ValueError(
            "Chat template renders tool results inside assistant turns, but adding a "
            "second tool result did not change the rendered output, so the tool-result "
            "block could not be isolated and cannot be excluded from the training "
            f"loss.\n{_TOOL_FIX_HINT}"
        )

    # Filter for marker tokens
    # get_added_vocab/convert_ids_to_tokens are declared on the concrete tokenizer
    # classes, not on the PreTrainedTokenizerBase alias; the cast is types-only.
    vocab_tokenizer = cast(PreTrainedTokenizerFast, tokenizer)
    added_vocab = set(vocab_tokenizer.get_added_vocab())
    tool_response_tokens = vocab_tokenizer.convert_ids_to_tokens(tool_response_ids)
    markers = [
        (token_id, token)
        for token_id, token in zip(tool_response_ids, tool_response_tokens)
        if token in added_vocab
    ]
    if len(markers) < 2:
        raise ValueError(
            "Chat template renders tool results inside assistant turns, but no marker "
            "tokens bracket them, so they cannot be excluded from the training loss.\n"
            f"{_TOOL_FIX_HINT}"
        )

    (open_id, opener), (close_id, closer) = markers[0], markers[-1]

    # A candidate is only a delimiter if it appears once per tool result and nowhere
    # else e.g. Falcon3 has a newline in its added vocab and plain-text tool tags, so
    # its only candidate delimiters are newlines, which occur throughout the render.
    if one_ids.count(open_id) != 1 or one_ids.count(close_id) != 1:
        raise ValueError(
            "Chat template renders tool results inside assistant turns, but the "
            f"candidate bracket ({opener!r}, {closer!r}) also appears elsewhere, so it "
            "cannot be used to exclude them from the training loss.\n"
            f"{_TOOL_FIX_HINT}"
        )

    logger.info(
        "Chat template nests tool results inside assistant turns; masking them via "
        "bracket (%r, %r).",
        opener,
        closer,
    )
    return [open_id], [close_id]


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

    # Templates that nest tool results inside the assistant turn put environment output
    # inside the unmasked span. Resolved here, after both paths above have settled
    # train_target, and before the user override below so a hand-supplied bracket
    # short-circuits the resolver (which raises when it can't recover one).
    _tool_keys = ("tool_response_template", "end_of_tool_response_template")
    supplied_tool_keys = [key for key in _tool_keys if config_collator_kwargs.get(key)]
    if len(supplied_tool_keys) == 1:
        # Masking needs both markers, so one alone would silently do nothing — and it
        # would also suppress the auto-detection that reports a nesting template.
        missing = next(key for key in _tool_keys if key not in supplied_tool_keys)
        raise ValueError(
            f"collator_kwargs sets '{supplied_tool_keys[0]}' without '{missing}'. "
            "Tool results are only excluded from the training loss when both markers "
            f"are set.\nFix: also set '{missing}' in collator_kwargs."
        )

    effective_response = config_collator_kwargs.get(
        "response_template", collator_kwargs.get("response_template")
    )
    effective_eot = config_collator_kwargs.get(
        "end_of_turn_template", collator_kwargs.get("end_of_turn_template")
    )
    if (
        collator_kwargs.get("train_target")
        in ("all_assistant_turns", "final_assistant_turn")
        and effective_response
        and effective_eot
        and not supplied_tool_keys
    ):
        tool_bracket = resolve_tool_response_template(
            tokenizer, effective_response, effective_eot
        )
        if tool_bracket is not None:
            collator_kwargs["tool_response_template"] = tool_bracket[0]
            collator_kwargs["end_of_tool_response_template"] = tool_bracket[1]

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
