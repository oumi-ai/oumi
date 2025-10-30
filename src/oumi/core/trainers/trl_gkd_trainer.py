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

from typing import Any, Callable, Optional, Union

import torch
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    EvalPrediction,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from trl import GKDTrainer
from trl.trainer.gkd_config import GKDConfig
from trl.trainer.sft_config import SFTConfig


class TrlGkdTrainer(GKDTrainer):
    """Wrapper around TRL's GKDTrainer to handle Oumi's pre-tokenized datasets.

    This wrapper overrides the dataset preparation and collator setup to work with
    Oumi's pre-tokenized datasets instead of TRL's expected ChatML format.

    The key difference from the base GKDTrainer is that we skip the ChatML collator
    creation when the dataset is already tokenized and use the provided collator instead.
    """

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, torch.nn.Module, str]] = None,
        teacher_model: Union[PreTrainedModel, torch.nn.Module, str] = None,
        args: Optional[GKDConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        peft_config: Optional[Any] = None,
        formatting_func: Optional[Callable] = None,
    ):
        """Initialize GKD trainer, using provided collator for pre-tokenized datasets."""
        # Set remove_unused_columns=False
        args.remove_unused_columns = False

        # Check if dataset is already tokenized (has input_ids)
        is_processed = False
        if train_dataset is not None:
            column_names = list(next(iter(train_dataset)).keys())
            is_processed = "input_ids" in column_names

        # Handle collator based on dataset type
        if not is_processed:
            # Dataset needs ChatML conversion - use TRL's collator
            from trl.trainer.utils import DataCollatorForChatML

            data_collator = DataCollatorForChatML(
                tokenizer=processing_class, max_length=args.max_length
            )
        elif is_processed:
            # Dataset is tokenized - wrap the collator to add "prompts" field
            from oumi.core.collators.gkd_collator import GkdCollator

            # If no collator provided, create default DataCollatorForLanguageModeling
            if data_collator is None:
                from transformers import DataCollatorForLanguageModeling

                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=processing_class, mlm=False
                )

            data_collator = GkdCollator(data_collator)

        # Note: Prompt/completion boundary marking is handled by GkdCollator at collation time
        # No need to modify the dataset here

        # Call SFTTrainer's __init__ (skip GKDTrainer's __init__)
        super(GKDTrainer, self).__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        # Set GKD-specific attributes from config (copied from GKDTrainer.__init__)
        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd

        # Load teacher model (copied from GKDTrainer.__init__)
        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
            )
        else:
            # Make a copy to avoid modifying the original config dict
            teacher_model_init_kwargs = args.teacher_model_init_kwargs.copy()
            if "torch_dtype" in teacher_model_init_kwargs:
                dtype_value = teacher_model_init_kwargs["torch_dtype"]
                if dtype_value not in ["auto", None]:
                    # Convert string to torch.dtype only in our local copy
                    teacher_model_init_kwargs["torch_dtype"] = getattr(
                        torch, dtype_value
                    )

        if isinstance(teacher_model, str):
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model, **teacher_model_init_kwargs
            )

        # Prepare teacher model with accelerator
        if self.is_deepspeed_enabled:
            from transformers.integrations import prepare_deepspeed

            teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            teacher_model = self.accelerator.prepare_model(
                teacher_model, evaluation_mode=True
            )

        self.teacher_model = teacher_model
        self.teacher_model.eval()

        # Create generation config (copied from GKDTrainer.__init__)
        from transformers import GenerationConfig

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
        )

        if hasattr(self.model.generation_config, "eos_token_id"):
            self.generation_config.eos_token_id = (
                self.model.generation_config.eos_token_id
            )

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Any,
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        """Override dataset preparation to skip ChatML conversion for pre-tokenized data.

        Oumi datasets are already tokenized, so we skip TRL's dataset preparation
        which expects raw conversation data in "messages" format.

        Args:
            dataset: The dataset to prepare.
            processing_class: Tokenizer or processor.
            args: SFT training arguments.
            packing: Whether to use packing.
            formatting_func: Optional formatting function.
            dataset_name: Name of the dataset.

        Returns:
            The dataset unchanged, since it's already prepared by Oumi.
        """
        # Check if dataset is already tokenized
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        if is_processed:
            # Dataset is already tokenized by Oumi, return as-is
            return dataset

        # If not tokenized, fall back to parent's preparation
        return super()._prepare_dataset(
            dataset, processing_class, args, packing, formatting_func, dataset_name
        )
