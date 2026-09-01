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

from trl import DPOTrainer

_TOKENIZED_DPO_COLUMN_SETS = (
    frozenset(("prompt_ids", "chosen_ids", "rejected_ids")),
    frozenset(("prompt_input_ids", "chosen_input_ids", "rejected_input_ids")),
)
_OUMI_PROMPT_COLUMN = "messages"
_TRL_PROMPT_COLUMN = "prompt"


class TrlDpoTrainer(DPOTrainer):
    """Light wrapper supporting raw and Oumi-tokenized DPO datasets."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initializes the TrlDpoTrainer."""
        super().__init__(*args, **kwargs)

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        """Prepare raw datasets while preserving Oumi-tokenized datasets."""
        column_names = frozenset(dataset.column_names or ())
        if any(
            tokenized_columns <= column_names
            for tokenized_columns in _TOKENIZED_DPO_COLUMN_SETS
        ):
            return dataset

        if (
            _OUMI_PROMPT_COLUMN in column_names
            and _TRL_PROMPT_COLUMN not in column_names
        ):
            dataset = dataset.rename_column(_OUMI_PROMPT_COLUMN, _TRL_PROMPT_COLUMN)

        return super()._prepare_dataset(dataset, processing_class, args, dataset_name)
