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
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from trl import DPOTrainer
from trl.trainer.utils import hash_module

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
        ref_log_probs_cache_dir: str | None = None,
        **kwargs,
    ):
        """Initializes the TrlDpoTrainer."""
        self._ref_log_probs_cache_dir = (
            Path(ref_log_probs_cache_dir).expanduser()
            if ref_log_probs_cache_dir
            else None
        )
        super().__init__(*args, **kwargs)

    def _precompute_ref_logps(self, dataset, name, batch_size):
        """Precompute reference log probabilities with an optional stable cache."""
        if self._ref_log_probs_cache_dir is None:
            return super()._precompute_ref_logps(dataset, name, batch_size)

        model_hash = hash_module(cast(torch.nn.Module, self.ref_model or self.model))
        dataset_hash = Hasher.hash(dataset.data.table.replace_schema_metadata(None))
        fingerprint = Hasher.hash((dataset_hash, model_hash))
        cache_file = self._ref_log_probs_cache_dir / f"{name}-{fingerprint}.npz"

        if cache_file.exists():
            with np.load(cache_file) as loaded:
                ref_chosen_logps = loaded["ref_chosen_logps"]
                ref_rejected_logps = loaded["ref_rejected_logps"]
        else:
            dataloader = DataLoader(
                cast(torch.utils.data.Dataset, dataset),
                batch_size=batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                shuffle=False,
            )
            data_loader = self.accelerator.prepare(dataloader)
            chosen_batches = []
            rejected_batches = []
            for padded_batch in tqdm(
                iterable=data_loader,
                desc=f"Computing reference log probs for {name} dataset",
            ):
                chosen_logps, rejected_logps = self.compute_ref_log_probs(padded_batch)
                chosen_logps, rejected_logps = self.accelerator.gather_for_metrics(
                    (chosen_logps, rejected_logps)
                )
                chosen_batches.append(chosen_logps.cpu())
                rejected_batches.append(rejected_logps.cpu())

            ref_chosen_logps = torch.cat(chosen_batches).float().numpy()
            ref_rejected_logps = torch.cat(rejected_batches).float().numpy()
            if self.accelerator.is_main_process:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                temporary_cache_file = cache_file.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temporary_cache_file,
                    ref_chosen_logps=ref_chosen_logps,
                    ref_rejected_logps=ref_rejected_logps,
                )
                temporary_cache_file.replace(cache_file)
            self.accelerator.wait_for_everyone()

        dataset = dataset.add_column(name="ref_chosen_logps", column=ref_chosen_logps)
        return dataset.add_column(
            name="ref_rejected_logps",
            column=ref_rejected_logps,
            new_fingerprint=fingerprint,
        )

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
