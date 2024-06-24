from typing import Literal

from transformers import PreTrainedTokenizerBase


def apply_chat_template(
    samples: dict,
    tokenizer: PreTrainedTokenizerBase,
    task: Literal["sft", "generation"],
) -> dict:
    """Applies the chat template carried by the tokenizer to the input example.

    Args:
        samples (Dict): Mapping `messages` to a List containing the (ordered) messages
            exchanged within a single chat dialogue.
            Each item of example["messages"] is a dict mapping the `content` of the
            message and the `role` of the one relayed it.
            E.g., role == 'user' or role == 'assistant'.
        tokenizer (PreTrainedTokenizerBase): the tokenizer to be used to process
            the example.
        task (Literal[str]): The task type the example is used in.
            "sft": i.e., for training purposes.
            "generation": i.e., for inference purposes.

    Raises:
        NotImplementedError: Currently only the `sft` task mode is supported.
        ValueError: if requested `task` is not in "sft" or "generation"

    Returns:
        Dict: It adds a `text` key in the input `example` dictionary, mapped to a string
        carrying the `messages` to the tokenizer's chat format.
    """
    if task in ["sft", "generation"]:
        messages = samples["messages"]

        samples["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(task == "generation"),
        )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided "
            "task is one of ['sft', 'generation']"
        )
    return samples
