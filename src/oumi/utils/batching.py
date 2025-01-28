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

from typing import TypeVar

T = TypeVar("T")


def batch(dataset: list[T], batch_size: int) -> list[list[T]]:
    """Batches the provided dataset.

    Args:
        dataset: The dataset to batch, which is a flat list of items.
        batch_size: The desired size of each batch.

    Returns:
        A list of batches. Each batch is a list of `batch_size` items, assuming that
        the dataset's size is a multiple of `batch_size`. Otherwise, the last batch to
        be included will contain less items than `batch_size`.
    """
    batches = []
    for dataset_index in range(0, len(dataset), batch_size):
        batches.append(dataset[dataset_index : dataset_index + batch_size])
    return batches


def unbatch(dataset: list[list[T]]) -> list[T]:
    """Unbatches (flatten) the provided dataset."""
    return [item for batch in dataset for item in batch]
