from typing import Callable, Dict

from transformers import PreTrainedTokenizerBase

from lema.datasets.common import apply_chat_template


def _convert_to_lema_format(example: dict) -> dict:
    """Converts the input example to the LeMa format."""
    messages = []
    metadata = {}

    # Use default alpaca system prompt
    system_prompt = (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request."
    )

    # Use default aplaca user prompt template
    if example.get("input") is not None and len(example["input"]) > 0:
        # This example has both an instruction and a user input.
        user_prompt = """{instruction}\n\n### Input:\n{input}""".format(
            instruction=example["instruction"], input=example["input"]
        )
    else:
        user_prompt = example["instruction"]

    # Create message list
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": example["output"]})

    return {"messages": messages, "metadata": metadata}


def alpaca_preprocessing_fn(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[..., Dict]:
    """Builds a preprocessing function for the Alpaca dataset.

    Dataset: https://huggingface.co/datasets/tatsu-lab/alpaca
    """

    def prompt_generation_fn(sample) -> dict:
        sample = _convert_to_lema_format(sample)
        results = apply_chat_template(sample, tokenizer=tokenizer, task="sft")
        return results

    return prompt_generation_fn
