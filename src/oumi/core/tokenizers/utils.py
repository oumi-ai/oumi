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


import functools
from typing import Literal, NamedTuple

import numpy as np
import transformers

from oumi.core.constants import LABEL_IGNORE_INDEX
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types import Conversation, Role
from oumi.utils.canonical_tool_conversations import (
    ARGUMENT_SENTINEL,
    canonical_tool_conversation,
)
from oumi.utils.conversation_utils import create_chat_template_inputs
from oumi.utils.logging import logger


class ChatTemplateToolFormat(NamedTuple):
    """The form a chat template expects tool-call data in."""

    arguments: Literal["mapping", "string"]
    """Shape of ``function.arguments`` on rendered tool calls.

    ``"mapping"`` decodes the stored JSON string into a dict before rendering,
    which most templates require; ``"string"`` passes the JSON string through
    unchanged, for templates that double-encode dicts (e.g. Qwen2.5).
    """

    content: Literal["null", "empty"]
    """Value of ``content`` on assistant messages that only carry tool calls.

    ``"null"`` keeps ``None``; ``"empty"`` coerces it to ``""``, for templates
    that stringify ``None`` into a literal ``"None"`` (e.g. GLM-4.5).
    """


# Candidate payload shapes, in preference order. The first that renders cleanly wins.
#
# Note: `content: ""` is preferred over `None` because a template that interpolates
# `content` unconditionally renders `""` as nothing, but stringifies `None` into
# the literal text "None". GLM-4.5 does exactly this, injecting a stray `None`
# line before every tool call. Where `""` is instead the worse choice --
# DeepSeek-V3 drops the tool call entirely -- it is *detectably* worse, so the
# scan moves on and still resolves ("string", "null").
_CANDIDATE_FORMATS: tuple[ChatTemplateToolFormat, ...] = (
    ChatTemplateToolFormat(arguments="mapping", content="empty"),
    ChatTemplateToolFormat(arguments="mapping", content="null"),
    ChatTemplateToolFormat(arguments="string", content="empty"),
    ChatTemplateToolFormat(arguments="string", content="null"),
)

# Used when no candidate renders; see `detect_chat_template_tool_format`.
_FALLBACK_FORMAT = _CANDIDATE_FORMATS[0]


def _probe_renders_sentinel(
    tokenizer: BaseTokenizer,
    *,
    arguments_form: Literal["mapping", "string"],
    content_form: Literal["null", "empty"],
) -> bool | None:
    """Render the canonical tool call and check whether the sentinel survives.

    Detection is limited to the sentinel's presence and its escaping. Structural
    validation is not possible because templates share no common format:

        gemma-4   <|tool_call>call:get_weather{city:<|"|><sentinel><|"|>}<tool_call|>
        GLM-4.5   <tool_call>get_weather<arg_key>city</arg_key>
                  <arg_value><sentinel></arg_value></tool_call>
        Qwen3.5   <tool_call><function=get_weather><parameter=city>
                  <sentinel></parameter></function>

    So a template that mangles the output around the sentinel, while preserving
    the sentinel itself, is not detected here.

    Args:
        tokenizer: The tokenizer whose chat template is rendered. Not modified.
        arguments_form: The shape to give ``function.arguments``. ``"mapping"``
            decodes the stored JSON string to a dict; ``"string"`` leaves it.
        content_form: The value to give ``content`` on the tool-call message.
            ``"null"`` keeps ``None``; ``"empty"`` uses ``""``.

    Returns:
        True if the sentinel rendered verbatim, meaning this form is usable.
        False if the template silently dropped the tool call, or double-encoded
        the arguments so the value arrived escaped.
        None if the template raised, meaning it rejects this form outright.
    """
    messages, tools = create_chat_template_inputs(
        canonical_tool_conversation(),
        tool_arguments_format=arguments_form,
        content_format=content_form,
    )

    try:
        rendered: str = tokenizer.apply_chat_template(
            messages,  # type: ignore
            tokenize=False,
            tools=tools,  # type: ignore[arg-type]
        )
    except Exception:
        return None

    # The template dropped the tool call without complaining.
    if ARGUMENT_SENTINEL not in rendered:
        return False

    # An escaped quote beside the value means the template JSON-encoded
    # arguments that were already a JSON string, so the model would train on
    # `"{\"city\": \"<sentinel>\"}"` where an object belongs.
    if f'\\"{ARGUMENT_SENTINEL}' in rendered or f'{ARGUMENT_SENTINEL}\\"' in rendered:
        return False

    return True


@functools.cache
def detect_chat_template_tool_format(
    tokenizer: BaseTokenizer,
) -> ChatTemplateToolFormat:
    """Probe the tokenizer's chat template to find the accepted tool-call form.

    Renders a canonical tool call with a sentinel value across the four
    (arguments x content) combinations and returns the first that renders
    cleanly. Deterministic tie-break prefers ``("mapping", "empty")``.

    Cached per tokenizer — cost is at most 4 string renders, once. The cache
    also means the fallback warning fires once, at the moment the first
    tool-carrying record is rendered.

    Args:
        tokenizer: The tokenizer whose chat template is probed.

    Returns:
        The first form that rendered cleanly, or ``("mapping", "empty")`` with a
        warning logged when no form did.
    """
    for candidate in _CANDIDATE_FORMATS:
        if _probe_renders_sentinel(
            tokenizer,
            arguments_form=candidate.arguments,
            content_form=candidate.content,
        ):
            return candidate

    logger.warning(
        "No tool-call form rendered correctly for %r: its chat template "
        "rejected or corrupted every candidate in %r. Tool calls in this data "
        "will be rendered with the fallback form %r and may not match how the "
        "model is expected to emit them (dropped, double-encoded, or "
        "reformatted). Under the fallback form, %s\nChat template:\n%s",
        getattr(tokenizer, "name_or_path", None) or "<unknown>",
        [tuple(c) for c in _CANDIDATE_FORMATS],
        tuple(_FALLBACK_FORMAT),
        _describe_fallback_rendering(tokenizer),
        getattr(tokenizer, "chat_template", None) or "<none>",
    )
    return _FALLBACK_FORMAT


def _describe_fallback_rendering(tokenizer: BaseTokenizer) -> str:
    """Shows what the canonical record becomes under the fallback form.

    The rendered text — or the exception the template raises — is the evidence
    a reader needs to judge whether the fallback output is usable for training.
    """
    messages, tools = create_chat_template_inputs(
        canonical_tool_conversation(),
        tool_arguments_format=_FALLBACK_FORMAT.arguments,
        content_format=_FALLBACK_FORMAT.content,
    )
    try:
        rendered = tokenizer.apply_chat_template(
            messages,  # type: ignore
            tokenize=False,
            tools=tools,  # type: ignore[arg-type]
        )
    except Exception as e:
        return f"rendering the canonical example record raises {type(e).__name__}: {e}"
    return f"the canonical example record renders as:\n{rendered}"


def apply_chat_template_inputs(
    tokenizer: BaseTokenizer, conversation: Conversation
) -> tuple[Conversation | list[dict], list[dict] | None]:
    """Returns the ``(conversation, tools)`` pair to hand ``apply_chat_template``.

    A conversation with neither tool definitions nor tool calls is returned
    untouched, so that path is unchanged and never pays for a probe.
    """
    has_tool_calls = any(
        m.tool_calls for m in conversation.messages if m.role == Role.ASSISTANT
    )
    if not conversation.tools and not has_tool_calls:
        return conversation, None

    fmt = detect_chat_template_tool_format(tokenizer)
    return create_chat_template_inputs(
        conversation,
        tool_arguments_format=fmt.arguments,
        content_format=fmt.content,
    )


#
# Base class functions
#
def tokenize_for_completions_only_training_with_template(
    tokenizer: BaseTokenizer, conversation: Conversation
) -> dict:
    """Tokenize a conversation for completions-only training with a template."""
    chat_input, tools = apply_chat_template_inputs(tokenizer, conversation)

    batch: transformers.BatchEncoding = tokenizer.apply_chat_template(
        conversation=chat_input,  # type: ignore
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        tools=tools,  # type: ignore[arg-type]
    )

    data = batch.data

    assistant_tokens_mask = data.pop("assistant_masks")

    data["labels"] = [
        token_id if mask else LABEL_IGNORE_INDEX
        for mask, token_id in zip(assistant_tokens_mask, data["input_ids"])
    ]

    return data


def tokenize_for_completions_only_training_with_prefix(
    tokenizer: BaseTokenizer,
    conversation: Conversation,
    response_template: str,
    instruction_template: str,
    response_token_ids: list[int],
    instruction_token_ids: list[int],
) -> dict:
    """Tokenize a conversation for completions-only training with a prefix."""
    chat_input, tools = apply_chat_template_inputs(tokenizer, conversation)

    prompt: str = tokenizer.apply_chat_template(
        conversation=chat_input,  # type: ignore
        tokenize=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        tools=tools,  # type: ignore[arg-type]
    )
    tokenizer_batch: transformers.BatchEncoding = tokenizer(
        prompt, truncation=True, padding=False, return_tensors="pt"
    )

    batch = {k: v[0] for k, v in tokenizer_batch.data.items()}
    batch["labels"] = batch["input_ids"].clone()

    response_token_ids_idxs = []
    human_token_ids_idxs = []

    cond = np.atleast_1d(batch["labels"] == response_token_ids[0])

    for assistant_idx in np.where(cond)[0]:
        # find the indexes of the start of a response.
        if (
            response_token_ids
            == batch["labels"][
                assistant_idx : assistant_idx + len(response_token_ids)
            ].tolist()
        ):
            response_token_ids_idxs.append(assistant_idx + len(response_token_ids))

    if len(response_token_ids_idxs) == 0:
        logger.warning(
            f"Could not find response key `{response_template}` in the "
            f"following instance: {tokenizer.decode(batch['input_ids'])} "
            f"This instance will be ignored in loss calculation. "
            f"Note, if this happens often, consider increasing the `max_seq_length`."
        )
        batch["labels"][:] = LABEL_IGNORE_INDEX

    human_token_ids = instruction_token_ids
    for human_idx in np.where(batch["labels"] == human_token_ids[0])[0]:
        # find the indexes of the start of a human answer.
        if (
            human_token_ids
            == batch["labels"][human_idx : human_idx + len(human_token_ids)].tolist()
        ):
            human_token_ids_idxs.append(human_idx)

    if len(human_token_ids_idxs) == 0:
        logger.warning(
            f"Could not find instruction key `{instruction_template}` in the "
            f"following instance: {tokenizer.decode(batch['input_ids'])} "
            f"This instance will be ignored in loss calculation. "
            f"Note, if this happens often, consider increasing the `max_seq_length`."
        )
        batch["labels"][:] = LABEL_IGNORE_INDEX

    if (
        len(human_token_ids_idxs) > 0
        and len(response_token_ids_idxs) > 0
        and human_token_ids_idxs[0] > response_token_ids_idxs[0]
    ):
        human_token_ids_idxs = [0] + human_token_ids_idxs

    for idx, (start, end) in enumerate(
        zip(human_token_ids_idxs, response_token_ids_idxs)
    ):
        # Make pytorch loss function ignore all non response tokens
        if idx != 0:
            batch["labels"][start:end] = LABEL_IGNORE_INDEX
        else:
            batch["labels"][:end] = LABEL_IGNORE_INDEX

    if len(response_token_ids_idxs) < len(human_token_ids_idxs):
        batch["labels"][human_token_ids_idxs[-1] :] = LABEL_IGNORE_INDEX

    return batch


#
# Multi-turn collator functions
#
def mask_labels_without_user_template(
    labels: np.ndarray,
    response_token_ids: list[int],
    ignore_index: int = LABEL_IGNORE_INDEX,
) -> None:
    """Apply completion-only masking when no user template is provided.

    This strategy masks everything except the last assistant response, allowing
    the model to learn only from the final assistant turn in the conversation.

    Args:
        labels: Label array to mask
        response_token_ids: Token IDs of the response template.
        ignore_index: Value to use for masked positions
    """
    # Find all response positions
    response_starts = find_all_sequences(labels, response_token_ids)

    if not response_starts:
        # No assistant responses found, mask everything
        labels[:] = ignore_index
        return

    # Save original labels before masking
    original_labels = labels.copy()

    # Mask everything initially
    labels[:] = ignore_index

    # Only unmask the last assistant response
    last_response_start = response_starts[-1]

    # Unmask from the last response start to the end of the sequence
    labels[last_response_start:] = original_labels[last_response_start:]


def mask_labels_for_completions_only(
    labels: np.ndarray,
    response_token_ids: list[int],
    instruction_token_ids: list[int],
    ignore_index: int = LABEL_IGNORE_INDEX,
) -> None:
    """Apply completion-only masking to labels with user and assistant templates.

    This strategy masks everything except assistant response content, using user
    templates to determine the boundaries of each assistant response.

    Args:
        labels: Label array to mask
        response_token_ids: Token IDs of the response template.
        instruction_token_ids: Token IDs of the instruction template.
        ignore_index: Value to use for masked positions
    """
    # Find all response and user positions
    response_starts = find_all_sequences(labels, response_token_ids)
    user_starts = find_all_sequences(labels, instruction_token_ids)

    # If no response templates found, mask everything
    if not response_starts:
        labels[:] = ignore_index
        return

    # Save original labels before masking
    original_labels = labels.copy()

    # Mask everything except assistant responses
    labels[:] = ignore_index  # Start by masking everything

    # Unmask each assistant response (content after the template)
    for resp_start in response_starts:
        # Find the next user template start after this response
        resp_end = len(labels)  # Default to end of sequence
        for user_start in user_starts:
            # user_start is position after user template, so we need to go back
            user_template_start = user_start - len(instruction_token_ids)
            if user_template_start > resp_start:
                resp_end = user_template_start
                break

        # Restore the original labels for the response content only
        # (starting after the response template)
        labels[resp_start:resp_end] = original_labels[resp_start:resp_end]


def find_all_sequences(arr: np.ndarray, target: list[int]) -> list[int]:
    """Find all occurrences of target sequence in array.

    Returns the positions of the target sequence AFTER the found sequence.
    """
    arr_list = arr.tolist()
    positions = []
    for i in range(len(arr_list) - len(target) + 1):
        if arr_list[i : i + len(target)] == target:
            positions.append(i + len(target))  # Return position after the sequence
    return positions


#
# Utils
#
def tokenizer_for_inference(
    tokenizer: BaseTokenizer, conversation: Conversation
) -> dict:
    """Tokenize a conversation for inference."""
    chat_input, tools = apply_chat_template_inputs(tokenizer, conversation)

    return tokenizer.apply_chat_template(
        conversation=chat_input,  # type: ignore
        tokenize=True,
        return_dict=True,
        tools=tools,  # type: ignore[arg-type]
    )
