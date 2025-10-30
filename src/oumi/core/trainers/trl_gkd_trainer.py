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
    """Wrapper around TRL's GKDTrainer for on-policy knowledge distillation.

    This trainer uses DistillationDataset which provides prompts in TRL's
    conversational format. TRL handles tokenization and properly marks
    prompt/completion boundaries using the conversation structure.

    Key Features:
    - Proper prompt/response boundaries (no heuristics!)
    - TRL's ChatML collator handles tokenization
    - Chat templates properly applied
    - On-policy knowledge distillation from teacher to student

    Example usage:
        ```yaml
        data:
          train:
            datasets:
              - dataset_name: "distillation"
                dataset_path: "prompts.jsonl"  # {"prompts": "text"}
        training:
          trainer_type: "TRL_GKD"
          gkd:
            teacher_model_name_or_path: "larger-model"
            lmbda: 0.5
            beta: 0.5
            temperature: 1.0
        ```

    The DistillationDataset wraps prompts in conversation format:
        ```python
        {
            "messages": [
                {"role": "user", "content": "prompt text"},
                {"role": "assistant", "content": ""}  # Placeholder
            ]
        }
        ```

    TRL extracts the prompt using `messages[:-1]` (all but last message),
    then uses teacher/student distributions for distillation.
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
        """Initialize GKD trainer for distillation training.

        Expects DistillationDataset which provides untokenized conversations
        in TRL's format. TRL handles tokenization and boundary detection.
        """
        # Check dataset format
        if train_dataset is not None and len(train_dataset) > 0:
            first_example = train_dataset[0]
            has_messages = "messages" in first_example
            is_tokenized = "input_ids" in first_example

            if not has_messages or is_tokenized:
                raise ValueError(
                    "GKD training requires untokenized datasets with 'messages' field. "
                    "Use DistillationDataset which provides prompts in conversation format:\n"
                    "  data:\n"
                    "    train:\n"
                    "      datasets:\n"
                    "        - dataset_name: 'distillation'\n"
                    "          dataset_path: 'prompts.jsonl'\n"
                    "\n"
                    "Create prompts.jsonl with format: {\"prompts\": \"your prompt here\"}"
                )

        # Set remove_unused_columns=False to preserve "messages" field
        # TRL will tokenize but needs messages for the collator
        args.remove_unused_columns = False

        # Call GKDTrainer's __init__ - it handles everything properly
        super().__init__(
            model=model,
            teacher_model=teacher_model,
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

