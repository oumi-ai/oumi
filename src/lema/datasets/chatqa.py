from typing import Callable, Dict

from transformers import PreTrainedTokenizerBase

from lema.datasets.common import apply_chat_template


def _convert_to_lema_format(example: dict) -> dict:
    """Converts the input example to the LeMa format."""
    messages = example["messages"].copy()
    metadata = {}

    for response in example["answers"]:
        messages.append({"role": "assistant", "content": response})

    return {"messages": messages, "metadata": metadata}


def chatqa_preprocessor_fn(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[..., Dict]:
    """Builds a preprocessing function for a TRL SFT (chat) trainer."""

    def prompt_generation_fn(sample) -> dict:
        sample = _convert_to_lema_format(sample)
        results = apply_chat_template(sample, tokenizer=tokenizer, task="sft")
        return results

    return prompt_generation_fn
