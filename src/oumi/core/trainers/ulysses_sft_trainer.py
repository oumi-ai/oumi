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

"""Ulysses Sequence Parallelism enabled SFT Trainer.

This module provides a custom SFTTrainer that integrates Ulysses sequence
parallelism for training on extremely long sequences.
"""

from typing import Callable, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainingArguments,
)
from trl import SFTTrainer

from oumi.utils.logging import logger


class UlyssesSFTTrainer(SFTTrainer):
    """SFT Trainer with Ulysses sequence parallelism support.

    This trainer extends TRL's SFTTrainer to support Ulysses sequence parallelism,
    enabling training on extremely long sequences by sharding them across multiple GPUs.
    """

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, torch.nn.Module]] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[list] = None,
        optimizers: tuple = (None, None),
        preprocess_logits_for_metrics: Optional[Callable] = None,
        sequence_parallel_size: int = 1,
        **kwargs,
    ):
        """Initialize UlyssesSFTTrainer.

        Args:
            model: Model to train
            args: Training arguments
            data_collator: Data collator function
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            processing_class: Tokenizer or processor
            compute_loss_func: Custom loss computation function
            compute_metrics: Metrics computation function
            callbacks: Training callbacks
            optimizers: Optimizer and scheduler tuple
            preprocess_logits_for_metrics: Logits preprocessing function
            sequence_parallel_size: Number of GPUs for sequence parallelism
            **kwargs: Additional arguments passed to SFTTrainer
        """
        self.sequence_parallel_size = sequence_parallel_size

        # Store original model for setup
        self._original_model = model

        # Initialize parent SFTTrainer first
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs,
        )

        # Setup Ulysses SP after parent initialization
        if self.sequence_parallel_size > 1:
            self._setup_ulysses_sp()

    def _setup_ulysses_sp(self):
        """Setup Ulysses sequence parallelism for model and data loaders."""
        if self.model is None:
            logger.warning("Model is None, cannot setup Ulysses SP")
            return

        logger.info(
            f"Setting up Ulysses SP with sequence_parallel_size={self.sequence_parallel_size}"
        )

        # Setup model with Ulysses SP
        # Note: We don't need to patch the dataloader here since TRL handles
        # data loading internally. The Ulysses SP attention patches will handle
        # the sequence sharding during forward passes.

        try:
            from oumi.core.parallel.ulysses_sp import UlyssesSPAttentionHF

            UlyssesSPAttentionHF.register_with_transformers(
                self.model, self.sequence_parallel_size
            )
            logger.info("Ulysses SP attention patches applied successfully")
        except Exception as e:
            logger.error(f"Failed to setup Ulysses SP: {e}")
            raise

    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader, potentially with Ulysses SP adaptations.

        Returns:
            Training DataLoader, adapted for sequence parallelism if enabled
        """
        # Get the standard training dataloader from parent
        dataloader = super().get_train_dataloader()

        # For TRL trainers, we rely on the attention-level sharding
        # rather than dataloader-level sharding, since TRL has complex
        # data processing pipelines that are better left intact

        if self.sequence_parallel_size > 1:
            logger.info(
                f"Using Ulysses SP with sequence_parallel_size={self.sequence_parallel_size}. "
                "Sequence sharding will be handled at the attention level."
            )

        return dataloader

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Get evaluation dataloader.

        Args:
            eval_dataset: Optional evaluation dataset

        Returns:
            Evaluation DataLoader
        """
        # For evaluation, we use the standard dataloader since Ulysses SP
        # is primarily for training very long sequences
        return super().get_eval_dataloader(eval_dataset)

    @classmethod
    def from_config(
        cls,
        model: PreTrainedModel,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        args: Optional[TrainingArguments] = None,
        sequence_parallel_size: int = 1,
        **kwargs,
    ) -> "UlyssesSFTTrainer":
        """Create UlyssesSFTTrainer from configuration.

        Args:
            model: Model to train
            processing_class: Tokenizer or processor
            args: Training arguments
            sequence_parallel_size: Number of GPUs for sequence parallelism
            **kwargs: Additional arguments

        Returns:
            Configured UlyssesSFTTrainer instance
        """
        return cls(
            model=model,
            processing_class=processing_class,
            args=args,
            sequence_parallel_size=sequence_parallel_size,
            **kwargs,
        )
