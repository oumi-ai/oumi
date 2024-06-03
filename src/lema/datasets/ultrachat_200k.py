"""Handling Huggingface/ultrachat_200k in the context of SFT via trl library.

https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

Significant portion of code was copied from:
https://github.com/huggingface/alignment-handbook/blob/main/src/alignment/data.py#L28
"""

from typing import Callable, Literal

from transformers import PreTrainedTokenizerBase


def maybe_insert_system_message(messages, tokenizer):
    """Insert a system message to start the chat dialogue.

    Args:
        messages (_type_): _description_
        tokenizer (_type_): _description_
    """
    if messages[0]["role"] == "system":
        return

    chat_template = tokenizer.chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation"],
    auto_insert_empty_system_msg: bool = True,
):
    """Apply the chat template carried by the tokenizer to the input example.

    Args:
        example (_type_): _description_
        tokenizer (_type_): _description_
        task (Literal[]): _description_
        auto_insert_empty_system_msg (bool, optional): _description_. Defaults to True.

    Raises:
        NotImplementedError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if task in ["generation"]:
        raise NotImplementedError("currently only sft implementation is supported")

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
            f"Task {task} not supported, please ensure that the provided \
            task is one of ['sft', 'generation']"
        )
    return example


def trl_sft_ultrachat_200k_preprocessor_fn(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable:
    """Build a preprocessing function for a TRL SFT (chat) trainer."""

    def prompt_generation_fn(samples) -> dict:
        results = apply_chat_template(
            samples, tokenizer=tokenizer, task="sft", auto_insert_empty_system_msg=True
        )
        return results

    return prompt_generation_fn
