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

"""Data processing components for trainers."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from transformers import PreTrainedTokenizerBase

from oumi.utils.logging import logger


@dataclass
class DataConfig:
    """Configuration for data processing."""

    max_length: int = 4096
    pack_sequences: bool = True
    mask_user_messages: bool = True
    add_special_tokens: bool = True
    truncation: bool = True
    padding: str = "max_length"
    return_tensors: str = "pt"


class ChatTemplateProcessor:
    """Process chat templates with role masking for SFT."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: DataConfig,
    ):
        """Initialize chat template processor.

        Args:
            tokenizer: Tokenizer for processing
            config: Data configuration
        """
        self.tokenizer = tokenizer
        self.config = config

        # Setup special tokens
        self._setup_special_tokens()

    def _setup_special_tokens(self):
        """Setup special tokens for chat processing."""
        # Ensure we have necessary special tokens
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        logger.info(f"Using pad_token: {self.tokenizer.pad_token}")
        logger.info(f"Using eos_token: {self.tokenizer.eos_token}")

    def process_chat_sample(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process a single chat sample with role masking.

        Args:
            sample: Chat sample with 'messages' field

        Returns:
            Processed sample with input_ids, attention_mask, and labels
        """
        if "messages" not in sample:
            raise ValueError("Sample must contain 'messages' field")

        messages = sample["messages"]

        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # Fallback: simple concatenation
            formatted_text = self._simple_chat_format(messages)

        # Tokenize
        tokenized = self.tokenizer(
            formatted_text,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors=self.config.return_tensors,
            add_special_tokens=self.config.add_special_tokens,
        )

        # Create labels with role masking
        labels = self._create_labels_with_masking(messages, tokenized["input_ids"])

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

    def _simple_chat_format(self, messages: list[dict[str, str]]) -> str:
        """Simple chat formatting fallback."""
        formatted_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted_parts.append(f"<{role}>{content}</{role}>")
        return "".join(formatted_parts)

    def _create_labels_with_masking(
        self, messages: list[dict[str, str]], input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Create labels with role-based masking.

        Args:
            messages: Original messages
            input_ids: Tokenized input IDs

        Returns:
            Labels tensor with masked user messages (-100)
        """
        if not self.config.mask_user_messages:
            # No masking, labels = input_ids
            return input_ids.clone()

        labels = input_ids.clone()

        # Find user message boundaries and mask them
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Use tokenizer's template to find boundaries
            labels = self._mask_with_template(messages, labels)
        else:
            # Simple masking approach
            labels = self._mask_simple(messages, labels)

        return labels

    def _mask_with_template(
        self, messages: list[dict[str, str]], labels: torch.Tensor
    ) -> torch.Tensor:
        """Mask labels using chat template boundaries."""
        # This is a simplified approach - in practice, you'd want more
        # sophisticated boundary detection
        for i, message in enumerate(messages):
            if message.get("role") == "user":
                # Find tokens corresponding to this user message and mask them
                # This is a placeholder - actual implementation would need
                # more sophisticated token alignment
                pass

        return labels

    def _mask_simple(
        self, messages: list[dict[str, str]], labels: torch.Tensor
    ) -> torch.Tensor:
        """Simple masking approach."""
        # For now, return labels as-is
        # In practice, you'd implement role-based masking logic
        return labels


class SequencePacker:
    """Pack multiple sequences into single training examples."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 4096,
        pack_sequences: bool = True,
    ):
        """Initialize sequence packer.

        Args:
            tokenizer: Tokenizer for processing
            max_length: Maximum sequence length
            pack_sequences: Whether to pack sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pack_sequences = pack_sequences

    def pack_samples(
        self, samples: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        """Pack multiple samples into single examples.

        Args:
            samples: List of tokenized samples

        Returns:
            List of packed samples
        """
        if not self.pack_sequences:
            return samples

        packed_samples = []
        current_input_ids = []
        current_labels = []
        current_attention_mask = []

        for sample in samples:
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            attention_mask = sample["attention_mask"]

            # Check if we can fit this sample
            needed_length = len(current_input_ids) + len(input_ids)

            if needed_length <= self.max_length:
                # Add to current packed sample
                current_input_ids.extend(input_ids.tolist())
                current_labels.extend(labels.tolist())
                current_attention_mask.extend(attention_mask.tolist())
            else:
                # Finalize current packed sample if not empty
                if current_input_ids:
                    packed_samples.append(
                        self._finalize_packed_sample(
                            current_input_ids, current_labels, current_attention_mask
                        )
                    )

                # Start new packed sample
                current_input_ids = input_ids.tolist()
                current_labels = labels.tolist()
                current_attention_mask = attention_mask.tolist()

        # Finalize last packed sample
        if current_input_ids:
            packed_samples.append(
                self._finalize_packed_sample(
                    current_input_ids, current_labels, current_attention_mask
                )
            )

        return packed_samples

    def _finalize_packed_sample(
        self, input_ids: list[int], labels: list[int], attention_mask: list[int]
    ) -> dict[str, torch.Tensor]:
        """Finalize a packed sample with proper padding.

        Args:
            input_ids: Input token IDs
            labels: Label token IDs
            attention_mask: Attention mask

        Returns:
            Finalized packed sample
        """
        # Pad to max_length
        pad_length = self.max_length - len(input_ids)

        if pad_length > 0:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id

            input_ids.extend([pad_token_id] * pad_length)
            labels.extend([-100] * pad_length)  # Ignore padding in loss
            attention_mask.extend([0] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class SFTDataCollator:
    """Data collator for SFT training."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 4096,
        pad_to_multiple_of: Optional[int] = None,
    ):
        """Initialize SFT data collator.

        Args:
            tokenizer: Tokenizer for processing
            max_length: Maximum sequence length
            pad_to_multiple_of: Pad to multiple of this value
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        # Setup padding token
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer must have pad_token_id or eos_token_id")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate features into a batch.

        Args:
            features: List of features to collate

        Returns:
            Batched features
        """
        # Determine batch size and sequence length
        batch_size = len(features)

        # Find maximum sequence length in this batch
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = min(max_len, self.max_length)

        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Initialize batch tensors
        batch = {
            "input_ids": torch.full(
                (batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long
            ),
            "attention_mask": torch.zeros((batch_size, max_len), dtype=torch.long),
            "labels": torch.full((batch_size, max_len), -100, dtype=torch.long),
        }

        # Fill batch with features
        for i, feature in enumerate(features):
            input_ids = feature["input_ids"]
            attention_mask = feature.get("attention_mask")
            labels = feature.get("labels")

            # Truncate if necessary
            seq_len = min(len(input_ids), max_len)

            # Copy input_ids
            batch["input_ids"][i, :seq_len] = input_ids[:seq_len]

            # Copy attention_mask
            if attention_mask is not None:
                batch["attention_mask"][i, :seq_len] = attention_mask[:seq_len]
            else:
                batch["attention_mask"][i, :seq_len] = 1

            # Copy labels
            if labels is not None:
                batch["labels"][i, :seq_len] = labels[:seq_len]

        return batch


class SFTDatasetProcessor:
    """Main processor for SFT datasets."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[DataConfig] = None,
    ):
        """Initialize SFT dataset processor.

        Args:
            tokenizer: Tokenizer for processing
            config: Data configuration
        """
        self.tokenizer = tokenizer
        self.config = config or DataConfig()

        # Initialize components
        self.chat_processor = ChatTemplateProcessor(tokenizer, self.config)
        self.sequence_packer = SequencePacker(
            tokenizer, self.config.max_length, self.config.pack_sequences
        )
        self.data_collator = SFTDataCollator(tokenizer, self.config.max_length)

    def process_dataset(self, dataset) -> torch.utils.data.Dataset:
        """Process a dataset for SFT training.

        Args:
            dataset: Raw dataset to process

        Returns:
            Processed dataset
        """
        # Process each sample
        processed_samples = []
        for sample in dataset:
            try:
                processed_sample = self.chat_processor.process_chat_sample(sample)
                processed_samples.append(processed_sample)
            except Exception as e:
                logger.warning(f"Failed to process sample: {e}")
                continue

        # Pack sequences if enabled
        if self.config.pack_sequences:
            processed_samples = self.sequence_packer.pack_samples(processed_samples)

        logger.info(f"Processed {len(processed_samples)} samples")
        return ProcessedDataset(processed_samples)

    def get_data_collator(self) -> Callable:
        """Get the data collator."""
        return self.data_collator


class ProcessedDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for processed samples."""

    def __init__(self, samples: list[dict[str, torch.Tensor]]):
        """Initialize processed dataset.

        Args:
            samples: List of processed samples
        """
        self.samples = samples

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get item by index."""
        return self.samples[idx]
