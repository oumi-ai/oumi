from typing import Literal, Optional

from oumi.core.tokenizers import BaseTokenizer

MESSAGES_KEY = "messages"
METADATA_KEY = "metadata"

_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"
_ROLE_KEY = "role"
_CONTENT_KEY = "content"


def convert_prompt_response_to_chat_example(
    prompt: str, response: str, system_instruction: Optional[str] = None
) -> dict:
    """Converts prompt, response, and system instruction into one chat example."""
    messages = [
        {_ROLE_KEY: _USER_ROLE, _CONTENT_KEY: prompt},
        {_ROLE_KEY: _ASSISTANT_ROLE, _CONTENT_KEY: response},
    ]

    if system_instruction is not None:
        messages = [
            {_ROLE_KEY: _SYSTEM_ROLE, _CONTENT_KEY: system_instruction}
        ] + messages
    return {MESSAGES_KEY: messages, METADATA_KEY: {}}


def apply_chat_template(
    samples: dict,
    tokenizer: BaseTokenizer,
    task: Literal["sft", "generation"],
) -> dict:
    """Applies the chat template carried by the tokenizer to the input example.

    Args:
        samples (Dict): Mapping `messages` to a List containing the (ordered) messages
            exchanged within a single chat dialogue.
            Each item of example["messages"] is a dict mapping the `content` of the
            message and the `role` of the one relayed it.
            E.g., role == 'user' or role == 'assistant'.
        tokenizer (BaseTokenizer): the tokenizer to be used to process
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
