from typing import Callable, Dict, Optional

from transformers import PreTrainedTokenizerBase

from lema.datasets.common import (
    apply_chat_template,
    convert_prompt_response_to_chat_example,
)


class PromptResponseSftPreprocessorFactory:
    """Constructs the preprocessing function for datasets of prompt-response pairs."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """Construct the factory with a single tokenizer."""
        self.tokenizer = tokenizer

    def get_preprocessing_fn(
        self,
        prompt_key: str = "inputs",
        response_key: str = "targets",
        system_instruction: Optional[str] = None,
    ) -> Callable[..., Dict]:
        """Builds a preprocessing function for single-turn prompt-response datasets.

        One example is the aya dataset: https://huggingface.co/datasets/CohereForAI/aya_dataset
        """

        def prompt_generation_fn(sample) -> dict:
            sample = self._convert_to_lema_format(
                sample, prompt_key, response_key, system_instruction
            )
            results = apply_chat_template(sample, tokenizer=self.tokenizer, task="sft")
            return results

        return prompt_generation_fn

    def _convert_to_lema_format(
        self,
        example: dict,
        prompt_key: str,
        response_key: str,
        system_instruction: Optional[str] = None,
    ) -> dict:
        """Converts the input example to the LeMa format."""
        prompt = example[prompt_key]
        response = example[response_key]

        return convert_prompt_response_to_chat_example(
            prompt=prompt, response=response, system_instruction=system_instruction
        )
