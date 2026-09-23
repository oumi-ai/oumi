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
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
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


@pytest.mark.parametrize("is_fsdp_enabled", [True, False])
def test_precompute_ref_logps_prepares_policy_once_for_fsdp(is_fsdp_enabled):
    trainer = object.__new__(TrlDpoTrainer)
    trainer.is_fsdp_enabled = is_fsdp_enabled
    trainer.ref_model = None
    trainer.model = MagicMock()
    trainer.model.eval.return_value = trainer.model
    trainer.model_wrapped = trainer.model
    trainer._precompute_engine = None
    trainer.optimizer = None
    trainer.accelerator = cast(
        Any,
        SimpleNamespace(
            state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=1)),
            prepare=MagicMock(return_value=trainer.model),
        ),
    )
    dataset = MagicMock()
    prepared_dataset = MagicMock()

    with patch.object(
        DPOTrainer,
        "_precompute_ref_logps",
        autospec=True,
        return_value=prepared_dataset,
    ) as precompute:
        result = trainer._precompute_ref_logps(dataset, "train", 1)

    assert result is prepared_dataset
    precompute.assert_called_once_with(trainer, dataset, "train", 1)
    if is_fsdp_enabled:
        trainer.accelerator.prepare.assert_called_once_with(trainer.model)
        assert trainer._precompute_engine is trainer.model
    else:
        trainer.accelerator.prepare.assert_not_called()
        assert trainer._precompute_engine is None


def test_prepare_for_training_reuses_precompute_engine():
    trainer = object.__new__(TrlDpoTrainer)
    model = MagicMock()
    optimizer = MagicMock()
    scheduler = MagicMock()
    train_dataloader = MagicMock()
    trainer.is_fsdp_enabled = True
    trainer._precompute_engine = model
    trainer.model = trainer.model_wrapped = model
    trainer.optimizer = optimizer
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
    trainer.create_optimizer = MagicMock()
    trainer.create_scheduler = MagicMock()
    trainer.callback_handler = cast(Any, SimpleNamespace())

    result = trainer._prepare_for_training(4, train_dataloader, None)

    trainer.accelerator.prepare_model.assert_called_once_with(model)
    trainer.accelerator.prepare_optimizer.assert_called_once_with(optimizer)
    trainer.create_optimizer.assert_not_called()
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
