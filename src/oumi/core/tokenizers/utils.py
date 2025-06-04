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

from typing import Optional

import numpy as np
import torch
import transformers

from oumi.core.constants import LABEL_IGNORE_INDEX
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types import Conversation
from oumi.utils.logging import logger


#
# Base class functions
#
def tokenize_for_completions_only_training_with_template(
    tokenizer: BaseTokenizer, conversation: Conversation
) -> dict:
    """Tokenize a conversation for completions-only training with a template."""
    batch: transformers.BatchEncoding = tokenizer.apply_chat_template(
        conversation=conversation,  # type: ignore
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
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
    prompt: str = tokenizer.apply_chat_template(
        conversation=conversation,  # type: ignore
        tokenize=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
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
# VL Collator functions
#
def find_token_sequence(sequence, target_tokens: list[int]) -> Optional[int]:
    """Find the starting index of a token sequence in labels.

    Args:
        sequence: Sequence of token IDs (can be torch.Tensor, np.ndarray, or list).
        target_tokens: List of token IDs to search for.

    Returns:
        Start index of the target sequence, or None if not found.
    """
    # Convert to list for consistent handling
    if isinstance(sequence, torch.Tensor):
        sequence_list = sequence.tolist()
    elif isinstance(sequence, np.ndarray):
        sequence_list = sequence.tolist()
    else:
        sequence_list = list(sequence)

    # Search for the target token sequence
    for i in range(len(sequence_list) - len(target_tokens) + 1):
        if sequence_list[i : i + len(target_tokens)] == target_tokens:
            return i

    return None


def mask_labels_for_completions_only(
    labels,
    response_token_ids: list[int],
    instruction_token_ids: Optional[list[int]] = None,
    ignore_index: int = LABEL_IGNORE_INDEX,
    response_template: Optional[str] = None,
) -> None:
    """Apply completion-only masking to labels.

    This function masks all tokens before the response template, so that loss
    is only computed on the model's response tokens.

    Args:
        labels: Labels to mask (can be torch.Tensor, np.ndarray, or list).
        response_token_ids: Token IDs of the response template.
        instruction_token_ids: Token IDs of the instruction template (optional).
        ignore_index: Index to use for masking tokens.
        response_template: String representation of response template for logging.
    """
    if not response_token_ids:
        raise ValueError("response_token_ids is required for completions-only training")

    # Find response template
    response_start_idx = find_token_sequence(labels, response_token_ids)

    if response_start_idx is None:
        # If response template not found, mask the entire sequence
        if response_template:
            logger.warning(
                f"Could not find response template '{response_template}' "
                "in sequence. Masking entire sequence."
            )
        labels[:] = ignore_index
    else:
        # Mask everything before the end of the response template
        response_end_idx = response_start_idx + len(response_token_ids)
        labels[:response_end_idx] = ignore_index


#
# Multi-turn collator functions
#
def mask_labels_for_arbitrary_conversations(
    labels: np.ndarray,
    input_ids: np.ndarray,
    tokenizer,
    response_template: str,
    ignore_index: int = -100,
) -> None:
    """Mask labels for arbitrary multi-turn conversations.

    Only assistant responses (after response_template) are kept for loss computation.

    Args:
        labels: Label array to mask
        input_ids: Corresponding input token IDs
        tokenizer: Tokenizer for encoding templates
        response_template: String marking start of assistant responses
        ignore_index: Value to use for masked positions
    """
    # Tokenize the response template
    response_tokens = tokenizer.encode(response_template, add_special_tokens=False)
    if not response_tokens:
        raise ValueError(f"Response template '{response_template}' produced no tokens")

    # Convert to lists for easier searching
    labels_list = labels.tolist()
    input_ids_list = input_ids.tolist()

    # Find all occurrences of the response template
    response_positions = []
    for i in range(len(input_ids_list) - len(response_tokens) + 1):
        if input_ids_list[i : i + len(response_tokens)] == response_tokens:
            response_positions.append(
                i + len(response_tokens)
            )  # Position after template

    if not response_positions:
        # No response template found, mask everything
        labels[:] = ignore_index
        return

    # Create masking ranges
    # We want to mask from start to first response, and between responses
    mask_ranges = []

    # Mask from start to first response
    mask_ranges.append((0, response_positions[0]))

    # For each response, we need to find where it ends
    # Strategy: mask from one response end to the next response start
    for i in range(len(response_positions) - 1):
        # Find the end of current response (start of next instruction)
        # This is the position of the next response template
        mask_ranges.append((response_positions[i], response_positions[i + 1]))

    # Apply masking
    for start, end in mask_ranges:
        labels[start:end] = ignore_index


def mask_labels_for_conversations_advanced(
    labels: np.ndarray,
    input_ids: np.ndarray,
    tokenizer,
    response_template: str,
    user_template: Optional[str] = None,
    ignore_index: int = -100,
) -> None:
    """Advanced masking that handles arbitrary conversations by detecting role transitions.

    Args:
        labels: Label array to mask
        input_ids: Corresponding input token IDs
        tokenizer: Tokenizer for encoding templates
        response_template: String marking start of assistant responses
        user_template: String marking start of user messages (optional)
        ignore_index: Value to use for masked positions
    """
    response_tokens = tokenizer.encode(response_template, add_special_tokens=False)
    user_tokens = (
        tokenizer.encode(user_template, add_special_tokens=False)
        if user_template
        else None
    )

    # Find all response and user positions
    response_starts = find_all_sequences(input_ids, response_tokens)
    user_starts = find_all_sequences(input_ids, user_tokens) if user_tokens else []

    # If we have user templates, use them to determine response endpoints
    if user_starts:
        # Mask everything except assistant responses
        labels[:] = ignore_index  # Start by masking everything

        # Unmask each assistant response
        for resp_start in response_starts:
            # Find the next user start after this response
            resp_end = len(labels)  # Default to end of sequence
            for user_start in user_starts:
                if user_start > resp_start:
                    resp_end = user_start
                    break

            # Unmask the response (keeping it for loss computation)
            labels[resp_start:resp_end] = input_ids[resp_start:resp_end]
    else:
        # Without user templates, use a simpler strategy
        # Mask everything before each response template
        current_pos = 0
        for resp_start in response_starts:
            labels[current_pos:resp_start] = ignore_index
            current_pos = resp_start


def find_all_sequences(arr: np.ndarray, target: list[int]) -> list[int]:
    """Find all occurrences of target sequence in array."""
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
    return tokenizer.apply_chat_template(
        conversation=conversation,  # type: ignore
        tokenize=True,
        return_dict=True,
    )
