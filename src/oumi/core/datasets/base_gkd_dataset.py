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

"""Base dataset for Generalized Knowledge Distillation (GKD) training."""

from typing import Any

from datasets import Dataset


class BaseGkdDataset:
    """Wrapper for SFT datasets to work with GKD training.

    GKD requires knowing where the prompt ends and completion begins.
    This wrapper takes an SFT dataset and modifies the labels to mark
    prompt tokens with -100, allowing GKD to:
    1. Extract prompts for on-policy generation
    2. Compute loss only on completion tokens

    The wrapper assumes the dataset has a 'conversation' structure where:
    - User messages are prompts
    - Assistant messages are completions
    """

    @staticmethod
    def convert_sft_to_gkd(
        dataset: Dataset,
        prompt_completion_ratio: float = 0.5,
    ) -> Dataset:
        """Convert an SFT dataset to GKD format by marking prompt/completion boundary.

        Args:
            dataset: SFT dataset with 'input_ids' and 'labels' fields.
            prompt_completion_ratio: Ratio of sequence that is prompt (default: 0.5).
                For example, 0.5 means first 50% is prompt, last 50% is completion.
                This is a heuristic when we can't determine the actual boundary.

        Returns:
            Modified dataset with labels properly masked for GKD.
        """

        def mark_prompt_completion_boundary(example: dict[str, Any]) -> dict[str, Any]:
            """Mark the prompt/completion boundary in labels.

            For GKD, we need to:
            1. Keep input_ids and attention_mask as-is (full sequence)
            2. Modify labels to be -100 for prompt tokens, actual IDs for completion

            Since we don't have the original conversation structure after tokenization,
            we use a heuristic: assume the first X% is prompt, rest is completion.
            """
            input_ids = example["input_ids"]
            labels = example.get(
                "labels",
                input_ids.copy() if isinstance(input_ids, list) else input_ids.clone(),
            )

            # Calculate prompt length based on ratio
            seq_length = len(input_ids)
            prompt_length = int(seq_length * prompt_completion_ratio)

            # Ensure we have at least some completion tokens
            min_completion_length = 10
            if seq_length - prompt_length < min_completion_length:
                prompt_length = max(0, seq_length - min_completion_length)

            # Mask prompt tokens in labels
            if isinstance(labels, list):
                # Convert to list if needed
                labels = labels.copy()
                for i in range(prompt_length):
                    labels[i] = -100
            else:
                # Tensor

                labels = labels.clone()
                labels[:prompt_length] = -100

            example["labels"] = labels
            return example

        # Apply the transformation to all examples
        modified_dataset = dataset.map(
            mark_prompt_completion_boundary,
            desc="Marking prompt/completion boundaries for GKD",
        )

        return modified_dataset

    @staticmethod
    def convert_sft_to_gkd_with_metadata(
        dataset: Dataset,
        prompt_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
    ) -> Dataset:
        """Convert an SFT dataset to GKD format using original metadata.

        This is more accurate than the ratio-based approach when the original
        prompt/completion structure is available in the dataset.

        Args:
            dataset: SFT dataset with metadata fields.
            prompt_key: Key for the instruction/prompt field.
            input_key: Key for additional input context (optional).
            output_key: Key for the output/completion field.

        Returns:
            Modified dataset with proper prompt/completion boundaries.

        Note:
            This method requires the dataset to still have the original text fields.
            If the dataset is already fully tokenized without metadata, use
            convert_sft_to_gkd() instead.
        """

        def mark_boundary_from_metadata(example: dict[str, Any]) -> dict[str, Any]:
            """Mark prompt/completion using original text metadata."""
            # This would require re-tokenizing with proper masking
            # For now, this is a placeholder for future enhancement
            raise NotImplementedError(
                "Metadata-based conversion not yet implemented. "
                "Use convert_sft_to_gkd() with prompt_completion_ratio instead."
            )

        return dataset.map(mark_boundary_from_metadata)
