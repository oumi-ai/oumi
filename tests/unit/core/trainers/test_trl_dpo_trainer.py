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
import importlib
import json
import re
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

from oumi.core.trainers.trl_dpo_trainer import TrlDpoTrainer


def _mock_prepare_inputs() -> tuple[PreTrainedTokenizerBase, DPOConfig]:
    return cast(PreTrainedTokenizerBase, MagicMock()), cast(DPOConfig, MagicMock())


def _tool_call(arguments: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "arguments": json.dumps(arguments),
        },
    }


def _fsdp_trainer(fsdp_version: int = 1) -> Any:
    trainer: Any = object.__new__(TrlDpoTrainer)
    trainer.is_fsdp_enabled = True
    trainer.ref_model = None
    trainer.model = trainer.model_wrapped = torch.nn.Linear(1, 1)
    trainer._precompute_engine = None
    trainer._precompute_model_hash = None
    trainer.args = SimpleNamespace(
        dataloader_num_workers=0, dataloader_pin_memory=False
    )
    trainer.data_collator = lambda rows: {"x": torch.tensor([r["x"] for r in rows])}
    trainer.accelerator = SimpleNamespace(
        state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=fsdp_version)),
        prepare=MagicMock(side_effect=lambda obj: obj),
        gather_for_metrics=lambda value: value,
        is_main_process=True,
        wait_for_everyone=lambda: None,
    )
    return trainer


def _reference_forward(trainer, batch):
    assert not trainer.model.training
    scores = batch["x"].float()
    return scores, -scores


def _saved_dataset(path, values):
    Dataset.from_dict({"x": values}).save_to_disk(path)
    return Dataset.load_from_disk(path)


def test_precompute_ref_logps_prepares_fsdp_policy_once(tmp_path):
    trainer = _fsdp_trainer()
    raw_model = trainer.model
    fsdp_model = torch.nn.Sequential(raw_model)
    events = []

    def prepare(obj):
        if isinstance(obj, DataLoader):
            return obj
        assert obj is raw_model
        events.append("prepare")
        return fsdp_model

    def reference_forward(trainer, batch):
        assert trainer.model is fsdp_model
        events.append("forward")
        return _reference_forward(trainer, batch)

    trainer.accelerator.prepare.side_effect = prepare
    with (
        patch("trl.trainer.utils.hash_module", return_value="hash") as hash_module,
        patch(
            "oumi.core.trainers.trl_dpo_trainer.broadcast_object_list",
            side_effect=lambda objects, from_process: objects,
        ),
        patch.object(
            DPOTrainer,
            "compute_ref_log_probs",
            autospec=True,
            side_effect=reference_forward,
        ),
    ):
        results = [
            trainer._precompute_ref_logps(
                _saved_dataset(tmp_path / name, [1, 2, 3]), name, 2
            )
            for name in ("train", "eval")
        ]

    hash_module.assert_called_once_with(raw_model)
    assert events == ["prepare"] + ["forward"] * 4
    assert trainer.model is trainer.model_wrapped is trainer._precompute_engine
    for result in results:
        assert result["ref_chosen_logps"] == [1.0, 2.0, 3.0]
        assert result["ref_rejected_logps"] == [-1.0, -2.0, -3.0]


def test_precompute_ref_logps_reuses_cached_scores_without_preparing(tmp_path):
    first_trainer = _fsdp_trainer()
    second_trainer = _fsdp_trainer()
    second_trainer.model.load_state_dict(first_trainer.model.state_dict())
    _saved_dataset(tmp_path, [1, 2])

    with (
        patch(
            "oumi.core.trainers.trl_dpo_trainer.broadcast_object_list",
            side_effect=lambda objects, from_process: objects,
        ),
        patch.object(
            DPOTrainer,
            "compute_ref_log_probs",
            autospec=True,
            side_effect=_reference_forward,
        ) as reference_forward,
    ):
        first_trainer._precompute_ref_logps(
            Dataset.load_from_disk(tmp_path), "train", 2
        )
        result = second_trainer._precompute_ref_logps(
            Dataset.load_from_disk(tmp_path), "train", 2
        )

    assert reference_forward.call_count == 1
    assert second_trainer.accelerator.prepare.call_count == 0
    assert second_trainer._precompute_engine is None
    assert result["ref_chosen_logps"] == [1.0, 2.0]
    assert result["ref_rejected_logps"] == [-1.0, -2.0]


def test_precompute_ref_logps_uses_rank_zero_model_hash(tmp_path):
    trainer = _fsdp_trainer()
    trainer.accelerator.is_main_process = False
    dataset = _saved_dataset(tmp_path, [1])

    def broadcast(objects, from_process):
        assert objects == [None] and from_process == 0
        objects[0] = "rank-0-hash"
        return objects

    with (
        patch("trl.trainer.utils.hash_module") as hash_module,
        patch(
            "oumi.core.trainers.trl_dpo_trainer.broadcast_object_list",
            side_effect=broadcast,
        ),
        patch.object(
            DPOTrainer,
            "compute_ref_log_probs",
            autospec=True,
            side_effect=_reference_forward,
        ),
    ):
        result = trainer._precompute_ref_logps(dataset, "train", 1)

    hash_module.assert_not_called()
    assert result._fingerprint == Hasher.hash((dataset._fingerprint, "rank-0-hash"))


def test_precompute_ref_logps_delegates_without_fsdp():
    trainer = object.__new__(TrlDpoTrainer)
    trainer.is_fsdp_enabled = False
    trainer.ref_model = None
    dataset = MagicMock()
    precomputed_dataset = MagicMock()

    with patch.object(
        DPOTrainer,
        "_precompute_ref_logps",
        autospec=True,
        return_value=precomputed_dataset,
    ) as precompute:
        result = trainer._precompute_ref_logps(dataset, "train", 1)

    assert result is precomputed_dataset
    precompute.assert_called_once_with(trainer, dataset, "train", 1)


@pytest.mark.parametrize(
    ("fsdp_version", "versions", "error"),
    [
        (2, {}, "support FSDP1 only"),
        (1, {"transformers": "5.2.0"}, "requires transformers>=5.3,<5.17"),
        (1, {"transformers": "5.17.0"}, "requires transformers>=5.3,<5.17"),
        (1, {"trl": "1.7.0"}, "requires trl>=1.0,<1.7"),
    ],
)
def test_precompute_ref_logps_rejects_unverified_fsdp_setups(
    fsdp_version, versions, error
):
    trainer = _fsdp_trainer(fsdp_version)
    installed = {"transformers": "5.10.1", "trl": "1.6.0", **versions}

    with (
        patch(
            "oumi.core.trainers.trl_dpo_trainer.importlib.metadata.version",
            side_effect=installed.__getitem__,
        ),
        patch("trl.trainer.utils.hash_module") as hash_module,
        pytest.raises(RuntimeError, match=re.escape(error)),
    ):
        trainer._precompute_ref_logps(MagicMock(), "train", 1)

    hash_module.assert_not_called()
    trainer.accelerator.prepare.assert_not_called()


@pytest.mark.parametrize(
    ("transformers_version", "trl_version"),
    [("5.3.0", "1.4.0"), ("5.6.0", "1.6.0"), ("5.10.1", "1.6.0")],
)
def test_check_fsdp_precompute_support_accepts_deployed_versions(
    transformers_version, trl_version
):
    installed = {"transformers": transformers_version, "trl": trl_version}

    with patch(
        "oumi.core.trainers.trl_dpo_trainer.importlib.metadata.version",
        side_effect=installed.__getitem__,
    ):
        _fsdp_trainer()._check_fsdp_precompute_support()


def test_prepare_policy_configures_peft_before_fsdp():
    trainer = _fsdp_trainer()
    raw_model = trainer.model
    update_peft = MagicMock()

    def prepare(model):
        update_peft.assert_called_once_with(model, trainer.accelerator)
        return model

    trainer.accelerator.prepare.side_effect = prepare

    with (
        patch(
            "oumi.core.trainers.trl_dpo_trainer.is_peft_model",
            return_value=True,
        ),
        patch(
            "transformers.integrations.fsdp.update_fsdp_plugin_peft",
            update_peft,
        ),
    ):
        trainer._prepare_policy_for_ref_logps()

    trainer.accelerator.prepare.assert_called_once_with(raw_model)


def test_module_imports_without_update_fsdp_plugin_peft(monkeypatch):
    import transformers.integrations.fsdp as hf_fsdp

    monkeypatch.delattr(hf_fsdp, "update_fsdp_plugin_peft", raising=False)
    monkeypatch.delitem(sys.modules, "oumi.core.trainers.trl_dpo_trainer")

    importlib.import_module("oumi.core.trainers.trl_dpo_trainer")


@pytest.mark.parametrize("has_optimizer", [True, False])
def test_prepare_for_training_reuses_precompute_engine(has_optimizer):
    trainer = object.__new__(TrlDpoTrainer)
    model = MagicMock()
    optimizer = MagicMock()
    scheduler = MagicMock()
    train_dataloader = MagicMock()
    trainer.is_fsdp_enabled = True
    trainer._precompute_engine = model
    trainer.model = trainer.model_wrapped = model
    trainer.optimizer = optimizer if has_optimizer else None
    trainer.lr_scheduler = cast(Any, scheduler)
    trainer._created_lr_scheduler = False
    trainer.accelerator = cast(
        Any,
        SimpleNamespace(
            prepare_model=MagicMock(return_value=model),
            prepare_optimizer=MagicMock(return_value=optimizer),
            parallelism_config=None,
        ),
    )
    trainer.create_optimizer = MagicMock(return_value=optimizer)
    trainer.create_scheduler = MagicMock()
    trainer.callback_handler = cast(Any, SimpleNamespace())

    result = trainer._prepare_for_training(4, train_dataloader, None)

    trainer.accelerator.prepare_model.assert_called_once_with(model)
    trainer.accelerator.prepare_optimizer.assert_called_once_with(optimizer)
    if has_optimizer:
        trainer.create_optimizer.assert_not_called()
    else:
        trainer.create_optimizer.assert_called_once_with()
    trainer.create_scheduler.assert_called_once_with(num_training_steps=4)
    model.train.assert_called_once_with()
    assert result == (model, train_dataloader)
    assert trainer.callback_handler.model is model


class _CapturingProcessingClass:
    eos_token = "</s>"

    def __init__(self):
        self.rendered_messages: list[list[dict[str, Any]]] = []

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **kwargs
    ) -> dict[str, list[int]]:
        self.rendered_messages.append(copy.deepcopy(messages))
        return {"input_ids": list(range(len(messages) + 1))}


@pytest.mark.parametrize(
    "column_names",
    [
        ["prompt_ids", "chosen_ids", "rejected_ids"],
        ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids"],
    ],
)
def test_prepare_dataset_preserves_tokenized_datasets(column_names):
    trainer = object.__new__(TrlDpoTrainer)
    dataset = MagicMock(column_names=column_names)
    processing_class, args = _mock_prepare_inputs()

    with patch.object(DPOTrainer, "_prepare_dataset", autospec=True) as prepare:
        result = trainer._prepare_dataset(dataset, processing_class, args, "train")

    assert result is dataset
    prepare.assert_not_called()


def test_prepare_dataset_delegates_raw_datasets_to_trl():
    trainer = object.__new__(TrlDpoTrainer)
    dataset = MagicMock(column_names=["prompt", "chosen", "rejected"])
    prepared_dataset = MagicMock()
    processing_class, args = _mock_prepare_inputs()

    with patch.object(
        DPOTrainer,
        "_prepare_dataset",
        autospec=True,
        return_value=prepared_dataset,
    ) as prepare:
        result = trainer._prepare_dataset(dataset, processing_class, args, "train")

    assert result is prepared_dataset
    prepare.assert_called_once_with(trainer, dataset, processing_class, args, "train")


def test_prepare_dataset_maps_oumi_prompt_column_to_trl():
    trainer = object.__new__(TrlDpoTrainer)
    dataset = MagicMock(column_names=["messages", "chosen", "rejected"])
    renamed_dataset = MagicMock()
    dataset.rename_column.return_value = renamed_dataset
    processing_class, args = _mock_prepare_inputs()

    with patch.object(DPOTrainer, "_prepare_dataset", autospec=True) as prepare:
        trainer._prepare_dataset(dataset, processing_class, args, "train")

    dataset.rename_column.assert_called_once_with("messages", "prompt")
    prepare.assert_called_once_with(
        trainer, renamed_dataset, processing_class, args, "train"
    )


def test_prepare_dataset_rejects_tools_when_trl_lacks_tokenize_hook():
    trainer = object.__new__(TrlDpoTrainer)
    dataset = MagicMock(column_names=["prompt", "chosen", "rejected", "tools"])
    processing_class, args = _mock_prepare_inputs()

    with (
        patch.object(DPOTrainer, "_tokenize", None),
        pytest.raises(RuntimeError, match="require TRL 1.0 or newer"),
    ):
        trainer._prepare_dataset(dataset, processing_class, args, "train")


def test_tokenize_decodes_tool_arguments_without_mutating_input():
    trainer = object.__new__(TrlDpoTrainer)
    trainer._is_vlm = False
    processing_class = MagicMock()
    processing_class.apply_chat_template.return_value = {"input_ids": [1]}
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call({"case_id": "X"})],
        }
    ]
    original_messages = copy.deepcopy(messages)

    trainer._tokenize(processing_class, messages)

    assert messages == original_messages
    rendered_messages = processing_class.apply_chat_template.call_args.args[0]
    assert rendered_messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "case_id": "X"
    }


def test_tokenize_preserves_text_only_messages():
    trainer = object.__new__(TrlDpoTrainer)
    messages = [{"role": "assistant", "content": "Done."}]
    processing_class = MagicMock()

    with patch.object(DPOTrainer, "_tokenize", autospec=True) as tokenize:
        trainer._tokenize(processing_class, messages)

    assert tokenize.call_args.args[2] is messages


def test_prepare_dataset_preserves_disjoint_tool_argument_schemas():
    rows = [
        {
            "prompt": [{"role": "user", "content": "Find my record."}],
            "chosen": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_tool_call({"case_id": "X"})],
                }
            ],
            "rejected": [{"role": "assistant", "content": "I cannot help."}],
            "tools": json.dumps([{"type": "function", "function": {}}]),
        },
        {
            "prompt": [{"role": "user", "content": "Find my flight."}],
            "chosen": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _tool_call({"flight": "AA1", "seats": 2}),
                    ],
                }
            ],
            "rejected": [{"role": "assistant", "content": "I cannot help."}],
            "tools": json.dumps([{"type": "function", "function": {}}]),
        },
    ]
    dataset = Dataset.from_list(rows)
    trainer = object.__new__(TrlDpoTrainer)
    trainer._is_vlm = False
    trainer._tokenizer = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        eos_token="</s>"
    )
    processing_class = _CapturingProcessingClass()
    args = SimpleNamespace(dataset_num_proc=None)

    trainer._prepare_dataset(
        dataset,
        cast(PreTrainedTokenizerBase, processing_class),
        cast(DPOConfig, args),
        "train",
    )

    rendered_tool_arguments = [
        message["tool_calls"][0]["function"]["arguments"]
        for messages in processing_class.rendered_messages
        for message in messages
        if message.get("tool_calls")
    ]
    assert rendered_tool_arguments == [
        {"case_id": "X"},
        {"flight": "AA1", "seats": 2},
    ]
    assert dataset[0]["chosen"][0]["tool_calls"][0]["function"]["arguments"] == (
        '{"case_id": "X"}'
    )
