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

from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from transformers import PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

from oumi.core.trainers.trl_dpo_trainer import TrlDpoTrainer


def _mock_prepare_inputs() -> tuple[PreTrainedTokenizerBase, DPOConfig]:
    return cast(PreTrainedTokenizerBase, MagicMock()), cast(DPOConfig, MagicMock())


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
