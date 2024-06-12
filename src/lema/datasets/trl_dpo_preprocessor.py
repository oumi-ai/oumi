from typing import Callable

from transformers import PreTrainedTokenizerBase

_PROMPT_KEY = "prompt"
_CHOSEN_KEY = "chosen"
_REJECTED_KEY = "rejected"

_ROLE = "role"
_CONTENT = "content"
_ASSISTANT = "assistant"


def trl_dpo_chat_preprocessor_fn(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable:
    """Builds a preprocessing function for the TRL DPO trainer.

    DPOTrainer expects prompts, as well as the chosen and rejected responses
    for each prompt.
    """

    def prompt_generation_fn(samples) -> dict:
        prompt = samples[_PROMPT_KEY]
        chosen_chat = samples[_CHOSEN_KEY]
        rejected_chat = samples[_REJECTED_KEY]

        results = {
            _PROMPT_KEY: [],
            _CHOSEN_KEY: [],
            _REJECTED_KEY: [],
        }

        for prompt_sample, chosen_sample, rejected_sample in zip(
            prompt, chosen_chat, rejected_chat
        ):
            results[_PROMPT_KEY].append(prompt_sample)

            chosen_sample_response = _extract_from_chat_format(chosen_sample)
            rejected_sample_response = _extract_from_chat_format(rejected_sample)

            results[_CHOSEN_KEY].append(chosen_sample_response)
            results[_REJECTED_KEY].append(rejected_sample_response)

        return results

    return prompt_generation_fn


def _extract_from_chat_format(sample):
    # Get the last 'assistant' turn in the chat.
    for turn in sample[::-1]:
        if turn[_ROLE] == _ASSISTANT:
            return turn[_CONTENT]

    raise ValueError("No chat turn was found with an 'assistant' role.")
