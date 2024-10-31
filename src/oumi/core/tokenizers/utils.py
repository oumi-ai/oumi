import warnings
from typing import Optional, cast

import numpy as np
import torch

from oumi.core.constants import LABEL_IGNORE_INDEX
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types import Conversation


def tokenize_for_completions_only_training_with_template(
    tokenizer: BaseTokenizer, conversation: Conversation
) -> dict:
    """Tokenize a conversation for completions-only training with a template."""
    results = tokenizer.apply_chat_template(
        conversation=conversation,  # type: ignore
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )

    # results = cast(dict, results)

    assistant_tokens_mask = results.pop("assistant_masks")

    results["labels"] = [
        LABEL_IGNORE_INDEX if mask else token_id
        for mask, token_id in zip(assistant_tokens_mask, results["input_ids"])
    ]

    return results


def tokenize_for_completions_only_training_with_prefix(
    tokenizer: BaseTokenizer,
    conversation: Conversation,
    response_template: str,
    instruction_template: str,
    response_token_ids: list[int],
    instruction_token_ids: list[int],
) -> dict:
    """Tokenize a conversation for completions-only training with a prefix."""
    prompt = tokenizer.apply_chat_template(
        conversation=conversation,  # type: ignore
        tokenize=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
    )
    examples = [prompt]
    batch = tokenizer(examples, truncation=True, padding=False, return_tensors="pt")

    labels = batch["input_ids"].clone()
    batch["labels"] = labels

    for i in range(len(examples)):
        response_token_ids_idxs = []
        human_token_ids_idxs = []

        cond = np.atleast_1d(batch["labels"][i] == response_token_ids[0])

        for assistant_idx in np.where(cond)[0]:
            # find the indexes of the start of a response.
            if (
                response_token_ids
                == batch["labels"][i][
                    assistant_idx : assistant_idx + len(response_token_ids)
                ].tolist()
            ):
                response_token_ids_idxs.append(assistant_idx + len(response_token_ids))

        if len(response_token_ids_idxs) == 0:
            warnings.warn(
                f"Could not find response key `{response_template}` in the "
                f'following instance: {tokenizer.decode(batch["input_ids"][i])} '
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            batch["labels"][i, :] = LABEL_IGNORE_INDEX

        human_token_ids = instruction_token_ids
        for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
            # find the indexes of the start of a human answer.
            if (
                human_token_ids
                == batch["labels"][i][
                    human_idx : human_idx + len(human_token_ids)
                ].tolist()
            ):
                human_token_ids_idxs.append(human_idx)

        if len(human_token_ids_idxs) == 0:
            warnings.warn(
                f"Could not find instruction key `{instruction_template}` in the "
                f'following instance: {tokenizer.decode(batch["input_ids"][i])} '
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            batch["labels"][i, :] = LABEL_IGNORE_INDEX

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
                batch["labels"][i, start:end] = LABEL_IGNORE_INDEX
            else:
                batch["labels"][i, :end] = LABEL_IGNORE_INDEX

        if len(response_token_ids_idxs) < len(human_token_ids_idxs):
            batch["labels"][i, human_token_ids_idxs[-1] :] = LABEL_IGNORE_INDEX

    return {k: v[0] for k, v in batch.items()}


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
