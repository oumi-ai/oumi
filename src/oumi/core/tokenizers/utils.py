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
import json
from typing import Literal, NamedTuple

import numpy as np
import transformers

from oumi.core.constants import LABEL_IGNORE_INDEX
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types import Conversation
from oumi.utils.conversation_utils import create_chat_template_inputs
from oumi.utils.logging import logger


class ChatTemplateToolFormat(NamedTuple):
    """The form a chat template expects tool-call data in."""

    arguments: Literal["mapping", "string"]
    content: Literal["null", "empty"]


_SENTINEL_VALUE = "OUMI_PROBE_7f3a"


def _probe_renders_sentinel(
    tokenizer: BaseTokenizer,
    *,
    arguments_form: Literal["mapping", "string"],
    content_form: Literal["null", "empty"],
) -> bool | None:
    """Render a synthetic tool call and check whether the sentinel survives.

    Returns True if the sentinel appears verbatim, False if absent or
    double-encoded, None if the template raised.
    """
    args_value: dict | str
    if arguments_form == "mapping":
        args_value = {"city": _SENTINEL_VALUE}
    else:
        args_value = json.dumps({"city": _SENTINEL_VALUE})

    content_value = "" if content_form == "empty" else None

    messages = [
        {"role": "user", "content": "test"},
        {
            "role": "assistant",
            "content": content_value,
            "tool_calls": [
                {
                    "id": "call_probe",
                    "type": "function",
                    "function": {
                        "name": "probe_fn",
                        "arguments": args_value,
                    },
                }
            ],
        },
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "probe_fn",
                "description": "probe",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    try:
        rendered: str = tokenizer.apply_chat_template(
            messages,  # type: ignore
            tokenize=False,
            tools=tools,  # type: ignore[arg-type]
        )
    except Exception:
        return None

    if _SENTINEL_VALUE not in rendered:
        return False
    escaped = f"\\{_SENTINEL_VALUE}" in rendered or f'"{_SENTINEL_VALUE}"' not in (
        rendered if arguments_form == "string" else rendered
    )
    if arguments_form == "string" and '\\"city\\"' in rendered:
        return False
    if escaped and arguments_form == "mapping":
        pass
    return True


def detect_chat_template_tool_format(
    tokenizer: BaseTokenizer,
) -> ChatTemplateToolFormat:
    """Probe the tokenizer's chat template to find the accepted tool-call form.

    Renders a synthetic tool call with a sentinel value across the four
    (arguments x content) combinations and picks the best. Deterministic
    tie-break prefers ``("mapping", "null")``.

    Cached per model name — cost is at most 4 string renders, once.
    """
    model_name = getattr(tokenizer, "name_or_path", None) or ""
    return _detect_cached(model_name, tokenizer)


@functools.cache
def _detect_cached(model_name: str, tokenizer: BaseTokenizer) -> ChatTemplateToolFormat:
    candidates = [
        ("mapping", "null"),
        ("mapping", "empty"),
        ("string", "null"),
        ("string", "empty"),
    ]

    best = None
    for args_form, content_form in candidates:
        result = _probe_renders_sentinel(
            tokenizer,
            arguments_form=args_form,  # type: ignore[arg-type]
            content_form=content_form,  # type: ignore[arg-type]
        )
        if result is True and best is None:
            best = ChatTemplateToolFormat(
                arguments=args_form,  # type: ignore[arg-type]
                content=content_form,  # type: ignore[arg-type]
            )

    if best is not None:
        return best

    logger.warning(
        "No tool-call form rendered correctly for %r; falling back to "
        "('mapping', 'null'). Tool calls may not render as expected.",
        model_name,
    )
    return ChatTemplateToolFormat(arguments="mapping", content="null")


#
# Base class functions
#
def tokenize_for_completions_only_training_with_template(
    tokenizer: BaseTokenizer, conversation: Conversation
) -> dict:
    """Tokenize a conversation for completions-only training with a template."""
    chat_input: Conversation | list[dict] = conversation
    tools = None
    if conversation.tools:
        fmt = detect_chat_template_tool_format(tokenizer)
        messages, tools = create_chat_template_inputs(
            conversation,
            tool_arguments=fmt.arguments,
            tool_call_content=fmt.content,
        )
        chat_input = messages

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
    chat_input: Conversation | list[dict] = conversation
    tools = None
    if conversation.tools:
        fmt = detect_chat_template_tool_format(tokenizer)
        messages, tools = create_chat_template_inputs(
            conversation,
            tool_arguments=fmt.arguments,
            tool_call_content=fmt.content,
        )
        chat_input = messages

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
    chat_input: Conversation | list[dict] = conversation
    tools = None
    if conversation.tools:
        fmt = detect_chat_template_tool_format(tokenizer)
        messages, tools = create_chat_template_inputs(
            conversation,
            tool_arguments=fmt.arguments,
            tool_call_content=fmt.content,
        )
        chat_input = messages

    return tokenizer.apply_chat_template(
        conversation=chat_input,  # type: ignore
        tokenize=True,
        return_dict=True,
        tools=tools,  # type: ignore[arg-type]
    )
