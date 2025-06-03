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


def _find_pattern_start(
    labels: torch.Tensor, pattern_tokens: list[int]
) -> Optional[int]:
    """Find the starting index of the pattern in the labels."""
    # Get all positions where the first token matches
    potential_starts = np.where(np.atleast_1d(labels == pattern_tokens[0]))[0]

    # Check each position for full template match
    for start_idx in potential_starts:
        sequence = labels[start_idx : start_idx + len(pattern_tokens)].tolist()
        if sequence == pattern_tokens:
            return start_idx

    return None


def tokenizer_for_inference(
    tokenizer: BaseTokenizer, conversation: Conversation
) -> dict:
    """Tokenize a conversation for inference."""
    return tokenizer.apply_chat_template(
        conversation=conversation,  # type: ignore
        tokenize=True,
        return_dict=True,
    )
