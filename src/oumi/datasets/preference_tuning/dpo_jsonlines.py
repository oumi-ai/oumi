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

import pandas as pd
from typing_extensions import override

from oumi.core.datasets.base_dpo_dataset import BaseDpoDataset
from oumi.core.registry import register_dataset
from oumi.utils.io_utils import load_jsonlines
from oumi.utils.packaging import is_trl_v0_29_or_later

_KEYS = ("prompt", "chosen", "rejected")


@register_dataset("text_dpo_jsonl")
class TextDpoJsonlinesDataset(BaseDpoDataset):
    """DPO dataset over pre-rendered text rows in JSONL.

    Each row carries `prompt`, `chosen`, and `rejected` as plain strings that
    already went through a chat template. Rows are tokenized directly; the
    chat template is NOT applied again. Use this when prompts contain tool
    calls or other structure the caller rendered exactly once.
    """

    default_dataset = "text_dpo_jsonl"

    def __init__(
        self,
        *,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
        data: list[dict] | None = None,
        **kwargs,
    ):
        """Initializes the dataset from a JSONL file or an in-memory list."""
        if dataset_path is not None and data is not None:
            raise ValueError("Only one of dataset_path or data must be provided")
        if data is not None:
            rows = data
        elif dataset_path is not None:
            rows = load_jsonlines(dataset_path)
        else:
            raise ValueError("Either dataset_path or data must be provided")

        self._rows = pd.DataFrame(rows)

        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path, **kwargs)

    @override
    def _load_data(self) -> pd.DataFrame:
        return self._rows

    @override
    def transform(self, sample: dict) -> dict:
        """Tokenize one pre-rendered row without applying the chat template."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer is required to process a sample.")
        ids: dict[str, list[int]] = {}
        for key in _KEYS:
            text = sample[key]
            if not isinstance(text, str):
                raise ValueError(
                    f"text_dpo_jsonl rows carry pre-rendered strings; "
                    f"'{key}' is {type(text).__name__}"
                )
            ids[key] = list(
                self._tokenizer(text, add_special_tokens=False)["input_ids"]
            )
        eos = cast("int | None", self._tokenizer.eos_token_id)
        for key in ("chosen", "rejected"):
            if eos is not None and (not ids[key] or ids[key][-1] != eos):
                ids[key].append(eos)
        if is_trl_v0_29_or_later():
            return {
                "prompt_ids": ids["prompt"],
                "chosen_ids": ids["chosen"],
                "rejected_ids": ids["rejected"],
            }
        return {
            "prompt_input_ids": ids["prompt"],
            "chosen_input_ids": ids["chosen"],
            "rejected_input_ids": ids["rejected"],
        }
