from typing import Any, Dict, List

import trl

from oumi.core.tokenizers.base_tokenizer import BaseTokenizer

_INPUT_IDS_KEY = "input_ids"


class TextCompletionsCollatorWithPadding:
    def __init__(
        self, tokenizer: BaseTokenizer, instruction_prefix: str, response_prefix: str
    ):
        """Custom collator for text LLM training.

        Args:
        tokenizer: The tokenizer used for encoding the data.
        instruction_prefix: The prefix marking the beginning of the user instruction.
        response_prefix: The prefix marking the beginning of the assistant response.
        """
        self._default_collator = trl.DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template=instruction_prefix,
            response_template=response_prefix,
        )

        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer doesn't define `pad_token_id`.")

    def _collate(self, inputs: List[Any]) -> Dict[str, Any]:
        result = self._default_collator(inputs)
        return result

    def __call__(self, batch) -> Dict[str, Any]:
        """Pads to the longest length present in the batch.

        Args:
            batch: List of batch items.

        Returns:
            Dict[str, torch.Tensor]: Processed batch.
        """
        for item in batch:
            if _INPUT_IDS_KEY not in item:
                raise ValueError(
                    f"Item doesn't contain '{_INPUT_IDS_KEY}' key. "
                    f"Available keys: {item.keys()}"
                )

        # Collate batch prompts.
        collated_text_inputs = self._collate(batch)

        return collated_text_inputs
