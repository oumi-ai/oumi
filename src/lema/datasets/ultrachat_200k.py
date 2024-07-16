"""Handling Huggingface/ultrachat_200k in the context of SFT via trl library.

https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

Significant portion of code was copied from:
https://github.com/huggingface/alignment-handbook/blob/main/src/alignment/data.py#L28
"""

from typing import Callable, Dict, Literal

from transformers import PreTrainedTokenizerBase

import lema.utils.logging


def maybe_insert_system_message(messages, tokenizer):
    """Inserts an empty system message to prepend a chat dialogue.

    An empty message will not be added if the role corresponding to the first message
    of the input `messages` is already set to `system', or if the chat_template does
    not seem to support system messages in general.

    Note: A system message is typically used to ground the higher-level purpose the LLM
    has in the context of the messages. E.g., the LLM a prompted to be a hungry pirate.
    Adding empty prompts can be beneficial to homogenize a dataset where some dialogues
    do have explicit such system messages, and others do not.

    Args:
        messages (List[Dict]): Each item of is a dict mapping the `content` of the
            message and the `role` of the one relayed it.
        tokenizer (PreTrainedTokenizerBase): the tokenizer used to process the messages.
    """
    if messages[0]["role"] == "system":  # skip if it explicitly exists
        return

    chat_template = tokenizer.chat_template

    # confirm the jinja template supports a system message before inserting
    # TODO: this function can be reused by more datasets; to be moved in a broader scope
    # about chat_templates
    # TODO: Investigate which templates (models) are eligible for system-messages
    # NOTE: below <|im_start|> covers ChatML template and it is a hack that will be
    # be fixed when we repackage the templates logic at a broader score.
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})
    else:
        lema.utils.logging.logger.warning(
            "Requested to add an empty system message using a template."
        )


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation"],
    auto_insert_empty_system_msg: bool = True,
):
    """Applies the chat template carried by the tokenizer to the input example.

    Args:
        example (Dict): Mapping `messages` to a List containing the (ordered) messages
            exchanged within a single chat dialogue.
            Each item of example["messages"] is a dict mapping the `content` of the
            message and the `role` of the one relayed it.
            E.g., role == 'user' or role == 'assistant'.
        tokenizer (PreTrainedTokenizerBase): the tokenizer to be used to process
            the example.
        task (Literal[str]): The task type the example is used in.
            "sft": i.e., for training purposes.
            "generation": i.e., for inference purposes.
        auto_insert_empty_system_msg (bool, optional): To add or not an empty
            system message at the beginning of the formatted chat. Defaults to True.

    Raises:
        NotImplementedError: Currently only the `sft` task mode is supported.
        ValueError: if requested `task` is not in "sft" or "generation"

    Returns:
        Dict: It adds a `text` key in the input `example` dictionary, mapped to a string
        carrying the `messages` to the tokenizer's chat format.
    """
    if task in ["generation"]:
        raise NotImplementedError("currently only sft implementation is supported.")

    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(task == "generation"),
        )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided "
            "task is one of ['sft', 'generation']"
        )
    return example


def trl_sft_ultrachat_200k_preprocessor_fn(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[..., Dict]:
    """Builds a preprocessing function for a TRL SFT (chat) trainer."""

    def prompt_generation_fn(samples) -> dict:
        results = apply_chat_template(
            samples, tokenizer=tokenizer, task="sft", auto_insert_empty_system_msg=True
        )
        return results

    return prompt_generation_fn
