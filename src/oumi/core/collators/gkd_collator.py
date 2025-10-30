# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling


class GkdCollator:
    """Data collator for GKD training with Oumi's pre-tokenized datasets.

    This collator wraps a standard collator and:
    1. Marks the prompt/completion boundary in labels
    2. Adds the "prompts" field required by GKDTrainer's compute_loss method
    """

    def __init__(
        self, base_collator: DataCollatorForLanguageModeling, prompt_ratio: float = 0.4
    ):
        """Initialize the GKD collator.

        Args:
            base_collator: The base collator to use for standard tokenization.
            prompt_ratio: Fraction of sequence that is prompt (default: 0.4).
        """
        self.base_collator = base_collator
        self.prompt_ratio = prompt_ratio

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate features and add prompts field for GKD.

        Args:
            features: List of examples with input_ids and attention_mask.

        Returns:
            Collated batch with input_ids, attention_mask, labels, and prompts.
        """
        # Use base collator to get standard batch
        batch = self.base_collator(features)

        # FIRST: Mark prompt/completion boundary by masking prompt tokens in labels
        # This must happen BEFORE extracting prompts
        for i in range(len(batch["labels"])):
            seq_length = len(batch["labels"][i])
            prompt_length = int(seq_length * self.prompt_ratio)

            # Ensure at least 10 completion tokens
            min_completion = 10
            if seq_length - prompt_length < min_completion:
                prompt_length = max(0, seq_length - min_completion)

            # Mask prompt tokens
            batch["labels"][i][:prompt_length] = -100

        # THEN: Extract prompts (tokens before first completion token)
        prompts = []
        for idx, (input_ids, labels) in enumerate(
            zip(batch["input_ids"], batch["labels"])
        ):
            # Find where completions start (first non -100 token in labels)
            label_mask = labels != -100

            if label_mask.any():
                first_completion_idx = label_mask.nonzero(as_tuple=True)[0][0].item()
                # Prompts include everything UP TO the first completion token
                # This means tokens at indices [0, first_completion_idx)
                prompt = input_ids[:first_completion_idx]
            else:
                # If no labels (shouldn't happen in SFT), use first half as prompt
                prompt_len = len(input_ids) // 2
                prompt = input_ids[:prompt_len]

            prompts.append(prompt)

        # Pad prompts to same length
        if len(prompts) > 0:
            max_prompt_len = max(len(p) for p in prompts)
            padded_prompts = []
            for prompt in prompts:
                if len(prompt) < max_prompt_len:
                    # Pad on the left (prepend padding)
                    padding = torch.full(
                        (max_prompt_len - len(prompt),),
                        self.base_collator.tokenizer.pad_token_id,
                        dtype=prompt.dtype,
                    )
                    prompt = torch.cat([padding, prompt])
                padded_prompts.append(prompt)

            batch["prompts"] = torch.stack(padded_prompts)
        else:
            # Empty prompts - create empty tensor
            batch["prompts"] = torch.tensor([], dtype=batch["input_ids"].dtype)

        return batch
