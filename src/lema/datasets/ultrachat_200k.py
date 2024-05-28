"""Handling Huggingface/ultrachat_200k in the context of SFT via trl library.

https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

Significant portion of code was copied from:
https://github.com/huggingface/alignment-handbook/blob/main/src/alignment/data.py#L28
"""

from typing import Literal

# DEFAULT_CHAT_TEMPLATE =\
# "{% for message in messages %}\n{% if message['role'] == 'user' %}\
# \n{{ '<|user|>\n' + message['content'] + eos_token }}\
# \n{% elif message['role'] == 'system' %}\
# \n{{ '<|system|>\n' + message['content'] + eos_token }}\
# \n{% elif message['role'] == 'assistant' %}\
# \n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\
# \n{% endif %}\n{% if loop.last and \
# add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"  # noqa


def maybe_insert_system_message(messages, tokenizer):
    """_summary_.

    Args:
        messages (_type_): _description_
        tokenizer (_type_): _description_
    """
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
    auto_insert_empty_system_msg: bool = True,
):
    """_summary_.

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
    if task in ["generation", "rm", "dpo"]:
        raise NotImplementedError("currently only sft implementation is supported")

    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! \
                Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided \
            task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example


# TODO
# def trl_sft_chat_preprocessor_fn(
#     tokenizer: PreTrainedTokenizerBase,
# ) -> Callable:
#     """Build a preprocessing function for the TRL SFT trainer.
#     """
#     def prompt_generation_fn(samples) -> dict:
#     return prompt_generation_fn
