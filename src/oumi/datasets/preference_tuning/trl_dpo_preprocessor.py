from typing import Callable, Dict

from oumi.core.tokenizers import BaseTokenizer

_PROMPT_KEY = "prompt"
_CHOSEN_KEY = "chosen"
_REJECTED_KEY = "rejected"

_ROLE = "role"
_CONTENT = "content"
_ASSISTANT = "assistant"


def trl_dpo_chat_preprocessor_fn(
    tokenizer: BaseTokenizer,
) -> Callable[..., Dict]:
    """Builds a preprocessing function for the TRL DPO trainer.

    DPOTrainer expects prompts, as well as the chosen and rejected responses
    for each prompt.
    """
    return _convert_to_oumi_format


def _extract_from_chat_format(sample):
    # Get the last 'assistant' turn in the chat.
    for turn in sample[::-1]:
        if turn[_ROLE] == _ASSISTANT:
            return turn[_CONTENT]

    raise ValueError("No chat turn was found with an 'assistant' role.")


def _convert_to_oumi_format(samples: dict) -> dict:
    prompt = samples[_PROMPT_KEY]
    chosen_chat = samples[_CHOSEN_KEY]
    rejected_chat = samples[_REJECTED_KEY]

    chosen_chat_response = _extract_from_chat_format(chosen_chat)
    rejected_chat_response = _extract_from_chat_format(rejected_chat)

    return {
        _PROMPT_KEY: prompt,
        _CHOSEN_KEY: chosen_chat_response,
        _REJECTED_KEY: rejected_chat_response,
    }
