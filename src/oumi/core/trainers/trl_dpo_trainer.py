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

import copy
import importlib.metadata
import json
from typing import Any

from trl import DPOTrainer

_TOKENIZED_DPO_COLUMN_SETS = (
    frozenset(("prompt_ids", "chosen_ids", "rejected_ids")),
    frozenset(("prompt_input_ids", "chosen_input_ids", "rejected_input_ids")),
)
_OUMI_PROMPT_COLUMN = "messages"
_TRL_PROMPT_COLUMN = "prompt"
_TOOLS_COLUMN = "tools"


def _deserialize_tool_call_arguments(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Decode JSON tool arguments without mutating the source messages."""
    has_serialized_arguments = any(
        isinstance((tool_call.get("function") or {}).get("arguments"), str)
        for message in messages
        for tool_call in message.get("tool_calls") or []
    )
    if not has_serialized_arguments:
        return messages

    decoded_messages = copy.deepcopy(messages)
    for message in decoded_messages:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    return decoded_messages


class TrlDpoTrainer(DPOTrainer):
    """Light wrapper supporting raw and Oumi-tokenized DPO datasets."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initializes the TrlDpoTrainer."""
        self._precompute_engine = None
        super().__init__(*args, **kwargs)

    def _precompute_ref_logps(self, dataset, name, batch_size):
        """Prepare the policy once for FSDP precompute and training."""
        if (
            self.is_fsdp_enabled
            and self.ref_model is None
            and self._precompute_engine is None
        ):
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            if getattr(fsdp_plugin, "fsdp_version", 1) == 2:
                self.create_optimizer()
                self.model, self.optimizer = self.accelerator.prepare(
                    self.model, self.optimizer
                )
            else:
                self.model = self.accelerator.prepare(self.model)
            self.model_wrapped = self.model
            self._precompute_engine = self.model.eval()

        return super()._precompute_ref_logps(dataset, name, batch_size)

    def _prepare_for_training(
        self, max_steps, train_dataloader, resume_from_checkpoint
    ):
        """Reuse the FSDP policy prepared for reference scoring."""
        if self._precompute_engine is None or not self.is_fsdp_enabled:
            return super()._prepare_for_training(
                max_steps, train_dataloader, resume_from_checkpoint
            )

        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        model = self.accelerator.prepare_model(self._precompute_engine)
        if self.optimizer is None:
            self.optimizer = self.create_optimizer()
        self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        self.create_scheduler(num_training_steps=max_steps)

        self.model = self.model_wrapped = self._precompute_engine = model
        parallelism_config = getattr(self.accelerator, "parallelism_config", None)
        if (
            parallelism_config is not None
            and parallelism_config.sp_backend == "deepspeed"
            and parallelism_config.sp_enabled
        ):
            train_dataloader = self.accelerator.deepspeed_ulysses_dl_adapter(
                train_dataloader, model
            )

        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)
            self._load_optimizer_and_scheduler(resume_from_checkpoint)
            self._load_scaler(resume_from_checkpoint)

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        model.train()
        return model, train_dataloader

    def _tokenize(self, processing_class, input, **kwargs):
        """Decode serialized tool arguments immediately before rendering."""
        if isinstance(input, list):
            input = _deserialize_tool_call_arguments(input)
        return super()._tokenize(  # pyright: ignore[reportAttributeAccessIssue]
            processing_class, input, **kwargs
        )

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        """Prepare raw datasets while preserving Oumi-tokenized datasets."""
        column_names = frozenset(dataset.column_names or ())
        if any(
            tokenized_columns <= column_names
            for tokenized_columns in _TOKENIZED_DPO_COLUMN_SETS
        ):
            return dataset

        if _TOOLS_COLUMN in column_names and not callable(
            getattr(DPOTrainer, "_tokenize", None)
        ):
            raise RuntimeError(
                "Structured DPO datasets with tools require TRL 1.0 or newer "
                f"(installed: {importlib.metadata.version('trl')}). "
                "Upgrade with: pip install --upgrade 'trl>=1.0'"
            )

        if (
            _OUMI_PROMPT_COLUMN in column_names
            and _TRL_PROMPT_COLUMN not in column_names
        ):
            dataset = dataset.rename_column(_OUMI_PROMPT_COLUMN, _TRL_PROMPT_COLUMN)

        return super()._prepare_dataset(dataset, processing_class, args, dataset_name)
