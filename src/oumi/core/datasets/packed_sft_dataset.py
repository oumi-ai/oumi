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

from typing import Optional

import torch
from tqdm import tqdm
from typing_extensions import override

from oumi.core.constants import LABEL_IGNORE_INDEX
from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.datasets.base_sft_dataset import BaseSftDataset
from oumi.utils.logging import logger


class PackedSftDataset(BaseMapDataset):
    """A dataset that packs samples from a base SFT dataset to maximize efficiency."""

    def __init__(
        self,
        base_dataset: BaseSftDataset,
        max_seq_len: int,
        show_progress: bool = True,
        split_samples: bool = False,
        concat_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        enable_padding: bool = True,
        **kwargs,
    ):
        """Initialize the PackedSftDataset.

        Args:
            base_dataset: The base SFT dataset to pack samples from.
            max_seq_len: Maximum sequence length for packed samples.
            show_progress: Whether to show progress bar during packing.
                Defaults to True.
            split_samples: Whether to split samples that are longer than max_seq_len.
                If False, samples longer than max_seq_len will be skipped.
                Defaults to False.
            concat_token_id: Token ID to use for concatenating samples.
                If None, samples will be concatenated without a separator token.
                Defaults to None.
            pad_token_id: Token ID to use for padding.
                Required if enable_padding is True. Defaults to None.
            enable_padding: Whether to pad sequences to max_seq_len.
                If True, pad_token_id must be provided. Defaults to True.
            **kwargs: Additional arguments passed to BaseMapDataset.
        """
        super().__init__(**kwargs, dataset_name=base_dataset.dataset_name)

        self.base_dataset = base_dataset

        self._max_seq_len = max_seq_len
        self._disable_tqdm = not show_progress
        self._split_samples = split_samples
        self._concat_token_id = concat_token_id
        self._pad_token_id = pad_token_id
        self._enable_padding = enable_padding
        self._data: list[dict[str, torch.Tensor]] = []

        if self._enable_padding and self._pad_token_id is None:
            raise ValueError(
                "`pad_token_id` must be provided if `enable_padding` is True"
            )

        self._check_dataset_compatibility()
        self._load_data()

    @override
    def _load_data(self) -> None:
        """Pack the base dataset into constant-length samples."""
        buffer = self._get_empty_buffer()

        iterator = range(len(self.base_dataset))

        for idx in tqdm(
            iterator,
            desc="Packing dataset",
            dynamic_ncols=True,
            disable=self._disable_tqdm,
        ):
            sample = self.base_dataset[idx]
            sample_len = len(sample["input_ids"])

            if sample_len > self._max_seq_len and not self._split_samples:
                # We can't split samples, and the sample is too long to fit in
                # the context window. There is no way to handle this sample
                logger.warning(
                    f"Dataset sample is too long ({sample_len} > {self._max_seq_len}). "
                    "Please set `split_samples=True` or increase `max_seq_len`. "
                    "This sample will be skipped."
                )
                continue

            if (
                self._get_potential_sample_len(sample=sample, buffer=buffer)
                == self._max_seq_len
            ):
                # Done with the current buffer, we need to create a new pack
                self._append_sample_to_buffer(sample=sample, buffer=buffer)
                self._append_packed_sample_to_dataset(buffer)
                buffer = self._get_empty_buffer()
                continue
            elif (
                self._get_potential_sample_len(sample=sample, buffer=buffer)
                < self._max_seq_len
            ):
                # We still have space in the buffer, so we add the sample to it
                # and keep going
                self._append_sample_to_buffer(sample=sample, buffer=buffer)
                continue

            # We don't have space in the buffer, so we need to create a new pack
            if self._split_samples:
                self._append_sample_to_buffer(sample=sample, buffer=buffer)

                while self._get_sample_len(buffer) >= self._max_seq_len:
                    finished_sample, buffer = self._split_sample(
                        buffer, cutoff=self._max_seq_len
                    )
                    self._append_packed_sample_to_dataset(finished_sample)
            else:
                # We're not allow to split samples, but buffer + sample is too large
                if self._get_sample_len(buffer) == 0:
                    self._append_sample_to_buffer(sample=sample, buffer=buffer)
                    self._append_packed_sample_to_dataset(buffer)
                else:
                    self._append_packed_sample_to_dataset(buffer)
                    buffer = self._get_empty_buffer()
                    self._append_sample_to_buffer(sample=sample, buffer=buffer)

        # Handle remaining samples in buffer
        if self._get_sample_len(buffer) > 0:
            if self._split_samples:
                while self._get_sample_len(buffer) > 0:
                    finished_sample, buffer = self._split_sample(
                        buffer, cutoff=self._max_seq_len
                    )
                    self._append_packed_sample_to_dataset(finished_sample)
            else:
                self._append_packed_sample_to_dataset(buffer)

    @override
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a pack from the dataset by index."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for PackedSftDataset")
        return self._data[idx]

    @override
    def transform(self, example: dict) -> dict:
        """No-op transform."""
        return example

    #
    # Private methods
    #
    def _append_packed_sample_to_dataset(self, buffer: dict[str, list]) -> None:
        """Creates a fixed length training sample from the buffer and add to dataset."""
        buffer_len = self._get_sample_len(buffer)

        if buffer_len > self._max_seq_len:
            raise ValueError(
                "Buffer is too long "
                f"({buffer_len} >= {self._max_seq_len}). "
                "Please increase `max_seq_len`."
            )

        # Convert lists to tensors
        sample = {k: torch.tensor(v, dtype=torch.long) for k, v in buffer.items()}

        # Pad if needed
        if self._enable_padding and self._pad_token_id is not None:
            if buffer_len < self._max_seq_len:
                pad_length = self._max_seq_len - buffer_len
                for name, value in sample.items():
                    if name == "labels":
                        pad_value = LABEL_IGNORE_INDEX
                    else:
                        pad_value = self._pad_token_id

                    sample[name] = torch.cat(
                        [
                            sample[name],
                            torch.full(
                                (pad_length,), fill_value=pad_value, dtype=torch.long
                            ),
                        ]
                    )

        self._data.append(sample)

    def _append_sample_to_buffer(
        self, sample: dict[str, list], buffer: dict[str, list]
    ) -> None:
        """Append a single training sample to the buffer.

        If concat token is enabled, and if and only if we actually concatenate
        two samples, we add the concat token in between the two samples
        """
        if len(sample["input_ids"]) == 0:
            # Nothing to add
            return

        should_add_concat_token = self._concat_token_id is not None

        if len(buffer["input_ids"]) == 0:
            # Buffer is empty, so we're not concatenating two different samples
            # no need to add concat token
            should_add_concat_token = False

        if should_add_concat_token:
            buffer["input_ids"].append(self._concat_token_id)
            buffer["labels"].append(LABEL_IGNORE_INDEX)  # exclude from loss

        buffer["input_ids"].extend(sample["input_ids"])
        buffer["labels"].extend(sample["labels"])

    def _split_sample(
        self, sample: dict[str, list], cutoff: int
    ) -> tuple[dict[str, list], dict[str, list]]:
        """Split a sample into two parts at the cutoff point.

        Args:
            sample: Dictionary containing lists to split
            cutoff: Index at which to split the lists

        Returns:
            Tuple of two dictionaries containing the split lists
        """
        first_half = {k: v[:cutoff] for k, v in sample.items()}
        second_half = {k: v[cutoff:] for k, v in sample.items()}
        return first_half, second_half

    def _get_empty_buffer(self) -> dict[str, list]:
        """Get an empty buffer with all required fields."""
        return {
            "input_ids": [],
            "labels": [],
        }

    def _get_sample_len(self, buffer: dict[str, list]) -> int:
        """Get the length of the samples in the buffer."""
        return len(buffer["input_ids"])

    def _get_potential_sample_len(
        self, sample: dict[str, list], buffer: dict[str, list]
    ) -> int:
        """Get the length of the samples in the buffer."""
        buffer_len = self._get_sample_len(buffer)
        sample_len = self._get_sample_len(sample)

        # In case we don't need to add a concat token
        if self._concat_token_id is None or buffer_len == 0 or sample_len == 0:
            return buffer_len + sample_len

        # In case we need to add a concat token
        return buffer_len + sample_len + 1

    def _check_dataset_compatibility(self) -> None:
        """Check the base dataset for errors."""
        if len(self.base_dataset) == 0:
            raise ValueError("Base dataset is empty. Cannot pack empty dataset.")

        keys = set(self.base_dataset[0].keys())

        if "input_ids" not in keys:
            raise ValueError("Base dataset must contain 'input_ids' key.")

        if "labels" not in keys:
            raise ValueError("Base dataset must contain 'labels' key.")

        if set(keys) != {"input_ids", "labels"}:
            logger.warning(
                "Base dataset contains additional keys. "
                "Only 'input_ids' and 'labels' will be used."
            )
