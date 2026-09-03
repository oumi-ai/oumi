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

import json
from unittest.mock import patch

import pandas as pd

from oumi.core.trainers.verl_grpo_trainer import VerlGrpoTrainer
from oumi.core.types.conversation import Conversation, Role
from oumi.datasets.grpo.letter_count import LetterCountGrpoDataset


def _raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "conversation_id": "oumi_letter_count_0",
                "messages": [
                    {
                        "content": "How many 'r's are in 'strawberry'?",
                        "role": "user",
                    }
                ],
                "metadata": {
                    "letter": "r",
                    "letter_count_integer": 3,
                    "word": "strawberry",
                },
            }
        ]
    )


def test_default_output_remains_native_grpo_format():
    with patch.object(LetterCountGrpoDataset, "_load_data", return_value=_raw_data()):
        dataset = LetterCountGrpoDataset(split="train")

    row = dataset[0]
    assert set(row) == {"prompt", "letter_count"}
    assert row["letter_count"] == 3


def test_return_conversations_preserves_metadata_for_verl():
    with patch.object(LetterCountGrpoDataset, "_load_data", return_value=_raw_data()):
        dataset = LetterCountGrpoDataset(
            split="train",
            return_conversations=True,
        )

    hf_dataset = dataset.to_hf(return_iterable=True)
    row = next(iter(hf_dataset))
    assert set(row) == {"conversation_json"}
    conversation = Conversation.from_json(row["conversation_json"])
    assert conversation.metadata["letter_count_integer"] == 3
    assert [message.role for message in conversation.messages] == [
        Role.SYSTEM,
        Role.USER,
    ]

    verl_row = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        row, 0, dataset.dataset_name, "train"
    )
    assert verl_row["reward_model"]["ground_truth"] == ""
    assert json.loads(verl_row["extra_info"]["metadata"])["letter_count_integer"] == 3
