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
from typing import Any

import numpy as np
import torch
from accelerate.utils import broadcast_object_list, is_peft_model
from datasets.fingerprint import Hasher
from packaging.specifiers import SpecifierSet
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.integrations.fsdp import update_fsdp_plugin_peft
from trl import DPOTrainer

# The FSDP precompute overrides below copy private TRL and Transformers methods that
# are unchanged across these ranges. Re-check them before widening a range.
_FSDP_PRECOMPUTE_VERSIONS = {"transformers": ">=5.3,<5.17", "trl": ">=1.0,<1.7"}

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
        self._precompute_model_hash = None
        super().__init__(*args, **kwargs)

    def _precompute_ref_logps(self, dataset, name, batch_size):
        """Precompute FSDP reference scores with a rank-consistent cache key."""
        if not self.is_fsdp_enabled or self.ref_model is not None:
            return super()._precompute_ref_logps(dataset, name, batch_size)

        # TODO: Remove the FSDP precompute overrides once Oumi's minimum TRL version
        # prepares FSDP policies for reference precompute itself (proposed for FSDP1
        # and FSDP2 in https://github.com/huggingface/trl/pull/6527).
        self._check_fsdp_precompute_support()

        # Local: hash_module first ships in TRL 0.29, below Oumi's TRL floor.
        from trl.trainer.utils import hash_module

        if self._precompute_model_hash is None:
            # With FSDP CPU-RAM-efficient loading, only rank 0 holds the real weights
            # until FSDP wraps the model, so hash there and share the result.
            model_hash = [
                hash_module(self.model) if self.accelerator.is_main_process else None
            ]
            broadcast_object_list(model_hash, from_process=0)
            self._precompute_model_hash = model_hash[0]
        fingerprint = Hasher.hash((dataset._fingerprint, self._precompute_model_hash))
        cache_file = Path(
            dataset._get_cache_file_path(fingerprint).removesuffix(".arrow") + ".npz"
        )
        if cache_file.exists():
            loaded = np.load(cache_file)
            ref_chosen_logps = loaded["ref_chosen_logps"]
            ref_rejected_logps = loaded["ref_rejected_logps"]
        else:
            dataloader = DataLoader(
                dataset,  # pyright: ignore[reportArgumentType]
                batch_size=batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                shuffle=False,
            )
            data_loader = self.accelerator.prepare(dataloader)
            self._prepare_policy_for_ref_logps()
            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(
                iterable=data_loader,
                desc=f"Computing reference log probs for {name} dataset",
            ):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(
                    padded_batch
                )
                ref_chosen_logp, ref_rejected_logp = (
                    self.accelerator.gather_for_metrics(
                        (ref_chosen_logp, ref_rejected_logp)
                    )
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

            ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()
            if self.accelerator.is_main_process:
                np.savez_compressed(
                    cache_file,
                    ref_chosen_logps=ref_chosen_logps,
                    ref_rejected_logps=ref_rejected_logps,
                )
            self.accelerator.wait_for_everyone()

        dataset = dataset.add_column("ref_chosen_logps", ref_chosen_logps)
        return dataset.add_column(
            "ref_rejected_logps",
            ref_rejected_logps,
            new_fingerprint=fingerprint,
        )

    def _check_fsdp_precompute_support(self) -> None:
        """Reject FSDP setups the precompute overrides were not verified against."""
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        if getattr(fsdp_plugin, "fsdp_version", 1) != 1:
            raise RuntimeError(
                "Precomputed DPO reference log probabilities currently support "
                "FSDP1 only."
            )

        for package, supported in _FSDP_PRECOMPUTE_VERSIONS.items():
            installed = importlib.metadata.version(package)
            if not SpecifierSet(supported).contains(installed, prereleases=True):
                raise RuntimeError(
                    "FSDP with precomputed DPO reference log probabilities requires "
                    f"{package}{supported} (installed: {installed})."
                )

    def _prepare_policy_for_ref_logps(self) -> None:
        """Prepare the FSDP1 policy once, immediately before reference scoring."""
        if self._precompute_engine is not None:
            return

        if is_peft_model(self.model):
            update_fsdp_plugin_peft(self.model, self.accelerator)

        self.model = self.accelerator.prepare(self.model)
        self.model_wrapped = self.model
        self._precompute_engine = self.model.eval()

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

        # Trainer.train() calls accelerator.free_memory() first. Preparing the
        # wrapped policy again only re-registers it, which clip_grad_norm_ and
        # save_state need; it does not wrap the policy a second time.
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
